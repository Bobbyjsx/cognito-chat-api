import logging
from typing import Any

import httpx
from fastapi import HTTPException

from app.billing.models import BillingPlan
from app.core.config import settings

logger = logging.getLogger(__name__)


class PaystackProvider:
    def __init__(self):
        self.base_url = (settings.paystack_base_url or "https://api.paystack.co").rstrip("/")
        self.secret_key = (settings.paystack_secret_key or "").strip()

    def _ensure_configured(self, plan: BillingPlan | None = None) -> None:
        if not self.secret_key:
            raise HTTPException(status_code=503, detail="Billing provider is not configured")
        if plan is not None and not (plan.provider_plan_code or "").strip():
            raise HTTPException(status_code=503, detail="Billing plan is not configured")

    @property
    def headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.secret_key}", "Content-Type": "application/json"}

    async def initialize_subscription_checkout(
        self, email: str, plan: BillingPlan, user_id: str, callback_url: str | None = None
    ) -> dict[str, Any]:
        self._ensure_configured(plan)
        body: dict[str, Any] = {
            "email": email,
            "amount": plan.amount,
            "plan": plan.provider_plan_code,
            "metadata": {"user_id": str(user_id), "plan_id": plan.id},
        }
        if callback_url:
            body["callback_url"] = callback_url
        try:
            async with httpx.AsyncClient(timeout=20.0) as client:
                response = await client.post(
                    f"{self.base_url}/transaction/initialize",
                    headers=self.headers,
                    json=body,
                )
        except HTTPException:
            raise
        except httpx.HTTPError as exc:
            logger.exception("Paystack checkout request failed")
            raise HTTPException(status_code=502, detail="Unable to start checkout") from exc

        if response.status_code >= 400:
            detail = "Unable to start checkout"
            try:
                message = response.json().get("message")
                if isinstance(message, str) and message.strip():
                    detail = message.strip()
            except Exception:  # noqa: S110
                pass
            logger.warning("Paystack initialize failed status=%s", response.status_code)
            raise HTTPException(status_code=502, detail=detail)

        payload = (response.json() or {}).get("data") or {}
        if not payload.get("authorization_url") or not payload.get("reference"):
            raise HTTPException(status_code=502, detail="Unable to start checkout")
        return payload

    async def verify_transaction(self, reference: str) -> dict[str, Any]:
        ref = (reference or "").strip()
        if not ref:
            raise HTTPException(status_code=400, detail="Reference is required")
        async with httpx.AsyncClient(timeout=20.0) as client:
            response = await client.get(f"{self.base_url}/transaction/verify/{ref}", headers=self.headers)
            if response.status_code == 404:
                raise HTTPException(status_code=404, detail="Transaction reference not found")
            response.raise_for_status()
            data = response.json().get("data")
            if not isinstance(data, dict):
                raise HTTPException(status_code=502, detail="Unexpected Paystack verification response")
            return data

    async def get_subscription(self, subscription_code: str) -> dict[str, Any]:
        code = (subscription_code or "").strip()
        if not code:
            raise HTTPException(status_code=409, detail="Subscription is not fully activated yet")
        async with httpx.AsyncClient(timeout=20.0) as client:
            response = await client.get(f"{self.base_url}/subscription/{code}", headers=self.headers)
            response.raise_for_status()
            data = response.json().get("data")
            if not isinstance(data, dict):
                raise HTTPException(status_code=502, detail="Unexpected Paystack subscription response")
            return data

    async def find_customer_subscription(self, customer_code: str) -> dict[str, Any] | None:
        code = (customer_code or "").strip()
        if not code:
            return None
        try:
            async with httpx.AsyncClient(timeout=20.0) as client:
                customer_res = await client.get(
                    f"{self.base_url}/customer/{code}",
                    headers=self.headers,
                )
                if customer_res.status_code >= 400:
                    return None
                customer_data = customer_res.json().get("data")
                if not customer_data or "id" not in customer_data:
                    return None

                customer_id = customer_data["id"]

                response = await client.get(
                    f"{self.base_url}/subscription",
                    headers=self.headers,
                    params={"customer": customer_id},
                )
                if response.status_code >= 400:
                    return None
                data = response.json().get("data")
        except httpx.HTTPError:
            return None
        if isinstance(data, list) and data:
            return data[0] if isinstance(data[0], dict) else None
        if isinstance(data, dict):
            return data
        return None

    async def cancel_subscription(self, subscription_code: str, email_token: str) -> bool:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/subscription/disable",
                headers=self.headers,
                json={"code": subscription_code, "token": email_token},
            )
            if response.status_code == 404:
                try:
                    data = response.json()
                    if "already inactive" in data.get("message", "") or data.get("code") == "not_found":
                        return True
                except Exception:  # noqa: S110
                    pass
            response.raise_for_status()
            return response.json()["status"]

    async def enable_subscription(self, subscription_code: str, email_token: str) -> bool:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/subscription/enable",
                headers=self.headers,
                json={"code": subscription_code, "token": email_token},
            )
            if response.status_code in (400, 404):
                try:
                    data = response.json()
                    msg = str(data.get("message", "")).lower()
                    if (
                        "already active" in msg
                        or "already enabled" in msg
                        or "cannot be enabled" in msg
                        or data.get("status") is True
                    ):
                        return True
                except Exception:  # noqa: S110
                    pass
            response.raise_for_status()
            return response.json().get("status", True)

    async def fetch_customer(self, email: str) -> dict[str, Any] | None:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{self.base_url}/customer/{email}", headers=self.headers)
            if response.status_code == 404:
                return None
            response.raise_for_status()
            return response.json()["data"]

    async def create_subscription(
        self, customer_code: str, plan_code: str, authorization_code: str, start_date: str | None = None
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "customer": customer_code,
            "plan": plan_code,
            "authorization": authorization_code,
        }
        if start_date:
            body["start_date"] = start_date

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/subscription",
                headers=self.headers,
                json=body,
            )
            response.raise_for_status()
            return response.json()["data"]
