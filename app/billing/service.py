from fastapi import HTTPException

from app.billing.models import BillingPlan, SubscriptionStatus
from app.billing.providers.paystack import PaystackProvider
from app.billing.repository import SubscriptionRepository
from app.billing.schemas import CheckoutResponse, PlanSchema, SubscriptionSchema
from app.core.config import settings


class BillingService:
    def __init__(self, repository: SubscriptionRepository, provider: PaystackProvider):
        self.repository = repository
        self.provider = provider

    def _get_plans(self):
        return {
            "go": BillingPlan(
                id="go_monthly",
                product="cognito",
                tier="go",
                interval="monthly",
                amount=599900,  # kobo
                currency="NGN",
                provider="paystack",
                provider_plan_code=settings.go_paystack_plan_code,
            ),
            "premium": BillingPlan(
                id="premium_monthly",
                product="cognito",
                tier="premium",
                interval="monthly",
                amount=999900,  # kobo
                currency="NGN",
                provider="paystack",
                provider_plan_code=settings.premium_paystack_plan_code,
            ),
        }

    def get_plans(self) -> list[PlanSchema]:
        plans = self._get_plans()
        return [
            PlanSchema(
                id=plan.id,
                tier=plan.tier,
                interval=plan.interval,
                amount=plan.amount // 100,  # return NGN not kobo for frontend
                currency=plan.currency,
            )
            for plan in plans.values()
        ]

    @staticmethod
    def _sanitize_callback_url(url: str | None) -> str | None:
        if not url:
            return None
        from urllib.parse import urlparse

        parsed = urlparse(url.strip())
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            return None
        if parsed.path.rstrip("/") != "/settings/billing":
            return None
        return f"{parsed.scheme}://{parsed.netloc}/settings/billing"

    async def create_checkout(
        self, plan_key: str, user_id: str, email: str, callback_url: str | None = None
    ) -> CheckoutResponse:
        from app.core.identity import is_usable_email

        plans = self._get_plans()
        plan = plans.get(plan_key.lower())
        if not plan:
            raise HTTPException(status_code=400, detail="Invalid plan selected")
        if not is_usable_email(email):
            raise HTTPException(
                status_code=400,
                detail="A valid email is required to start checkout. Please sign in again.",
            )

        checkout_data = await self.provider.initialize_subscription_checkout(
            email.strip(),
            plan,
            user_id,
            callback_url=self._sanitize_callback_url(callback_url),
        )

        return CheckoutResponse(
            checkout_url=checkout_data["authorization_url"],
            reference=checkout_data["reference"],
        )

    async def verify_and_activate_payment(self, reference: str, user_id: str) -> SubscriptionSchema:
        ref = (reference or "").strip()
        if not ref:
            raise HTTPException(status_code=400, detail="Missing reference")

        tx_data = await self.provider.verify_transaction(ref)
        if not tx_data or tx_data.get("status") != "success":
            raise HTTPException(status_code=400, detail="Payment was not successful or cannot be verified")

        metadata = tx_data.get("metadata") or {}
        tx_user_id = metadata.get("user_id")
        if not tx_user_id or str(tx_user_id) != str(user_id):
            raise HTTPException(status_code=403, detail="Transaction does not belong to this user")

        customer_code = (tx_data.get("customer") or {}).get("customer_code")
        if customer_code and not tx_data.get("subscription_code"):
            sub_data = await self.provider.find_customer_subscription(customer_code)
            if sub_data:
                tx_data["subscription_code"] = sub_data.get("subscription_code")

        from app.api.webhooks.paystack import _handle_charge_success

        await _handle_charge_success(tx_data, self.repository)

        updated = await self.get_user_subscription(user_id)
        if not updated:
            raise HTTPException(status_code=500, detail="Failed to load updated subscription")
        return updated

    async def get_user_subscription(self, user_id: str) -> SubscriptionSchema | None:
        sub = await self.repository.get_by_user_id(user_id)
        if not sub:
            return None

        scheduled = None
        if sub.scheduled_change:
            from app.billing.schemas import ScheduledChangeSchema

            status_val = (
                sub.scheduled_change.status.value
                if hasattr(sub.scheduled_change.status, "value")
                else str(sub.scheduled_change.status)
            )
            scheduled = ScheduledChangeSchema(
                target_tier=sub.scheduled_change.target_tier,
                effective_at=sub.scheduled_change.effective_at,
                status=status_val,
            )

        return SubscriptionSchema(
            tier=sub.tier,
            status=sub.status,
            interval=sub.interval,
            amount=sub.amount // 100,  # NGN
            currency=sub.currency,
            current_period_end=sub.current_period_end,
            cancel_at_period_end=sub.cancel_at_period_end,
            scheduled_change=scheduled,
        )

    async def cancel_subscription(self, user_id: str, email: str) -> bool:
        sub = await self.repository.get_by_user_id(user_id)
        if not sub:
            raise HTTPException(status_code=400, detail="No active subscription to cancel")
        if sub.cancel_at_period_end and not sub.scheduled_change:
            return True
        if sub.status not in [SubscriptionStatus.ACTIVE, SubscriptionStatus.CANCELLED]:
            raise HTTPException(status_code=400, detail="No active subscription to cancel")

        sub_code = (sub.provider_subscription_code or "").strip()
        email_token = None
        paystack_sub: dict | None = None
        if sub_code:
            paystack_sub = await self.provider.get_subscription(sub_code)
        elif sub.provider_customer_code:
            paystack_sub = await self.provider.find_customer_subscription(sub.provider_customer_code)
            if paystack_sub:
                sub_code = str(paystack_sub.get("subscription_code") or "").strip()
                if sub_code:
                    sub.provider_subscription_code = sub_code
                    await self.repository.save(sub)

        if isinstance(paystack_sub, dict):
            email_token = paystack_sub.get("email_token")

        if not sub.cancel_at_period_end:
            if not sub_code or not email_token:
                raise HTTPException(
                    status_code=409,
                    detail="Subscription is still activating. Try again in a moment.",
                )
            await self.provider.cancel_subscription(sub_code, email_token)

        # Also cancel any pending scheduled downgrade
        if sub.scheduled_change:
            future_status = getattr(sub.scheduled_change, "status", None)
            if hasattr(future_status, "value"):
                future_status = future_status.value
            if future_status == "pending":
                future_sub_code = sub.scheduled_change.provider_subscription_code
                if future_sub_code:
                    try:
                        future_sub = await self.provider.get_subscription(future_sub_code)
                        future_email_token = future_sub.get("email_token") if future_sub else None
                        if future_email_token:
                            await self.provider.cancel_subscription(future_sub_code, future_email_token)
                    except Exception as e:
                        import logging

                        logging.getLogger(__name__).warning(
                            "Failed to cancel future sub during full cancellation %s: %s", future_sub_code, e
                        )
            sub.scheduled_change = None

        sub.cancel_at_period_end = True
        sub.status = SubscriptionStatus.CANCELLED
        await self.repository.save(sub)

        return True

    async def schedule_downgrade(self, user_id: str, target_plan_key: str) -> dict:
        sub = await self.repository.get_by_user_id(user_id)
        if not sub:
            raise HTTPException(status_code=400, detail="No active subscription to downgrade")

        # Allow downgrade if active OR if it's cancelled but hasn't expired yet
        if sub.status not in [SubscriptionStatus.ACTIVE, SubscriptionStatus.CANCELLED]:
            raise HTTPException(status_code=400, detail="Subscription is not active")

        # Check if already downgrading
        if sub.scheduled_change and sub.scheduled_change.status == "pending":
            if sub.scheduled_change.target_tier == target_plan_key:
                return {
                    "current_plan": sub.tier,
                    "scheduled_plan": sub.scheduled_change.target_tier,
                    "effective_at": sub.scheduled_change.effective_at,
                }
            else:
                raise HTTPException(status_code=400, detail="Another downgrade is already pending")

        plans = self._get_plans()
        target_plan = plans.get(target_plan_key.lower())
        if not target_plan:
            raise HTTPException(status_code=400, detail="Invalid plan selected")

        if sub.tier == target_plan.tier:
            raise HTTPException(status_code=400, detail="Already on this plan")

        # Basic check to ensure it's actually a downgrade (could be extended)
        current_plan = plans.get(sub.tier)
        if current_plan and current_plan.amount <= target_plan.amount:
            raise HTTPException(status_code=400, detail="Target plan is not a downgrade")

        # 1. We must disable the current subscription renewal
        sub_code = (sub.provider_subscription_code or "").strip()
        email_token = None
        paystack_sub: dict | None = None

        if sub_code:
            paystack_sub = await self.provider.get_subscription(sub_code)
        elif sub.provider_customer_code:
            paystack_sub = await self.provider.find_customer_subscription(sub.provider_customer_code)
            if paystack_sub:
                sub_code = str(paystack_sub.get("subscription_code") or "").strip()
                if sub_code:
                    sub.provider_subscription_code = sub_code
                    await self.repository.save(sub)

        if isinstance(paystack_sub, dict):
            email_token = paystack_sub.get("email_token")
            authorization_code = (paystack_sub.get("authorization") or {}).get("authorization_code")
        else:
            authorization_code = None

        if not sub_code or not email_token or not authorization_code:
            raise HTTPException(
                status_code=409,
                detail="Subscription is still activating or missing authorization. Try again in a moment.",
            )

        was_already_cancelled = sub.status == SubscriptionStatus.CANCELLED or bool(sub.cancel_at_period_end)

        # Disable current subscription if it's currently active on Paystack
        if not was_already_cancelled or (paystack_sub and paystack_sub.get("status") == "active"):
            try:
                await self.provider.cancel_subscription(sub_code, email_token)
            except Exception as e:
                import logging

                logging.getLogger(__name__).warning("Current sub %s disable warning: %s", sub_code, e)

        # 2. Determine effective date for future Go subscription
        from datetime import datetime, timezone

        from app.billing.models import ScheduledChange, ScheduledChangeStatus

        now = datetime.now(timezone.utc)
        effective_date = None
        if sub.current_period_end and sub.current_period_end > now:
            effective_date = sub.current_period_end
        elif paystack_sub and paystack_sub.get("next_payment_date"):
            from app.api.webhooks.paystack import _parse_dt

            parsed_dt = _parse_dt(paystack_sub.get("next_payment_date"))
            if parsed_dt and parsed_dt > now:
                effective_date = parsed_dt
                sub.current_period_end = parsed_dt

        if not effective_date:
            effective_date = now

        # Ensure exact ISO 8601 UTC string without microseconds for Paystack (YYYY-MM-DDTHH:MM:SSZ)
        effective_date_utc = effective_date.astimezone(timezone.utc).replace(microsecond=0)
        start_date_iso = effective_date_utc.strftime("%Y-%m-%dT%H:%M:%SZ")

        try:
            future_sub = await self.provider.create_subscription(
                customer_code=sub.provider_customer_code,
                plan_code=target_plan.provider_plan_code,
                authorization_code=authorization_code,
                start_date=start_date_iso,
            )
            future_sub_code = future_sub.get("subscription_code")

            sub.scheduled_change = ScheduledChange(
                target_tier=target_plan.tier,
                effective_at=effective_date_utc,
                status=ScheduledChangeStatus.PENDING,
                provider_subscription_code=future_sub_code,
                was_already_cancelled=was_already_cancelled,
            )
            if was_already_cancelled:
                sub.status = SubscriptionStatus.CANCELLED
                sub.cancel_at_period_end = True
            else:
                sub.status = SubscriptionStatus.ACTIVE
                sub.cancel_at_period_end = True

            await self.repository.save(sub)
        except Exception:
            # If the future sub creation fails, record renewal was stopped
            sub.cancel_at_period_end = True
            await self.repository.save(sub)
            raise HTTPException(status_code=502, detail="Failed to schedule downgrade. Current renewal was stopped.")

        return {
            "current_plan": sub.tier,
            "scheduled_plan": target_plan.tier,
            "effective_at": effective_date_utc,
        }

    async def cancel_downgrade(self, user_id: str) -> bool:
        sub = await self.repository.get_by_user_id(user_id)
        if not sub:
            raise HTTPException(status_code=400, detail="No active subscription")

        if not sub.scheduled_change or sub.scheduled_change.status != "pending":
            raise HTTPException(status_code=400, detail="No pending downgrade to cancel")

        was_already_cancelled = getattr(sub.scheduled_change, "was_already_cancelled", False)

        # 1. Cancel the future Go subscription on Paystack
        future_sub_code = sub.scheduled_change.provider_subscription_code
        if future_sub_code:
            try:
                future_sub = await self.provider.get_subscription(future_sub_code)
                email_token = future_sub.get("email_token") if future_sub else None
                if email_token:
                    await self.provider.cancel_subscription(future_sub_code, email_token)
            except Exception as e:
                # Log but continue, we must handle current subscription
                import logging

                logging.getLogger(__name__).warning(
                    "Failed to cancel future Go subscription %s: %s", future_sub_code, e
                )

        # 2. Re-enable current Premium subscription ONLY if it was NOT already cancelled before!
        if not was_already_cancelled:
            current_sub_code = sub.provider_subscription_code
            if current_sub_code:
                try:
                    current_sub = await self.provider.get_subscription(current_sub_code)
                    if current_sub and current_sub.get("status") == "active":
                        # Already active on Paystack, no need to call enable
                        pass
                    else:
                        email_token = current_sub.get("email_token") if current_sub else None
                        if email_token:
                            await self.provider.enable_subscription(current_sub_code, email_token)
                except Exception as e:
                    import logging

                    logging.getLogger(__name__).warning(
                        "Could not re-enable current subscription %s on Paystack (continuing): %s", current_sub_code, e
                    )

            sub.status = SubscriptionStatus.ACTIVE
            sub.cancel_at_period_end = False
        else:
            # It was already cancelled before! Revert back to CANCELLED state with cancel_at_period_end = True
            sub.status = SubscriptionStatus.CANCELLED
            sub.cancel_at_period_end = True

        # 3. Clear scheduled change
        sub.scheduled_change = None
        await self.repository.save(sub)

        return True
