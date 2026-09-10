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

    async def create_checkout(self, plan_key: str, user_id: str, email: str) -> CheckoutResponse:
        plans = self._get_plans()
        plan = plans.get(plan_key.lower())
        if not plan:
            raise HTTPException(status_code=400, detail="Invalid plan selected")

        checkout_data = await self.provider.initialize_subscription_checkout(email, plan, user_id)

        return CheckoutResponse(
            authorization_url=checkout_data["authorization_url"], reference=checkout_data["reference"]
        )

    async def get_user_subscription(self, user_id: str) -> SubscriptionSchema | None:
        sub = await self.repository.get_by_user_id(user_id)
        if not sub:
            return None

        return SubscriptionSchema(
            tier=sub.tier,
            status=sub.status,
            interval=sub.interval,
            amount=sub.amount // 100,  # NGN
            currency=sub.currency,
            current_period_end=sub.current_period_end,
            cancel_at_period_end=sub.cancel_at_period_end,
        )

    async def cancel_subscription(self, user_id: str, email: str) -> bool:
        sub = await self.repository.get_by_user_id(user_id)
        if not sub or sub.status != SubscriptionStatus.ACTIVE:
            raise HTTPException(status_code=400, detail="No active subscription to cancel")

        paystack_sub = await self.provider.get_subscription(sub.provider_subscription_code)
        email_token = paystack_sub["email_token"]

        success = await self.provider.cancel_subscription(sub.provider_subscription_code, email_token)
        if success:
            sub.cancel_at_period_end = True
            sub.status = SubscriptionStatus.CANCELLED
            await self.repository.save(sub)

        return success
