from typing import Any

import httpx

from app.billing.models import BillingPlan
from app.core.config import settings


class PaystackProvider:
    def __init__(self):
        self.base_url = settings.paystack_base_url
        self.secret_key = settings.paystack_secret_key

    @property
    def headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.secret_key}", "Content-Type": "application/json"}

    async def initialize_subscription_checkout(self, email: str, plan: BillingPlan, user_id: str) -> dict[str, Any]:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/transaction/initialize",
                headers=self.headers,
                json={
                    "email": email,
                    "amount": plan.amount,
                    "plan": plan.provider_plan_code,
                    "metadata": {"user_id": str(user_id), "plan_id": plan.id},
                },
            )
            response.raise_for_status()
            data = response.json()
            return data["data"]

    async def get_subscription(self, subscription_code: str) -> dict[str, Any]:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{self.base_url}/subscription/{subscription_code}", headers=self.headers)
            response.raise_for_status()
            return response.json()["data"]

    async def cancel_subscription(self, subscription_code: str, email_token: str) -> bool:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/subscription/disable",
                headers=self.headers,
                json={"code": subscription_code, "token": email_token},
            )
            response.raise_for_status()
            return response.json()["status"]

    async def enable_subscription(self, subscription_code: str, email_token: str) -> bool:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/subscription/enable",
                headers=self.headers,
                json={"code": subscription_code, "token": email_token},
            )
            response.raise_for_status()
            return response.json()["status"]

    async def fetch_customer(self, email: str) -> dict[str, Any] | None:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{self.base_url}/customer/{email}", headers=self.headers)
            if response.status_code == 404:
                return None
            response.raise_for_status()
            return response.json()["data"]
