from datetime import datetime

from pydantic import BaseModel

from app.billing.models import SubscriptionStatus


class CheckoutRequest(BaseModel):
    plan: str  # "go" or "premium"


class CheckoutResponse(BaseModel):
    authorization_url: str
    reference: str


class PlanSchema(BaseModel):
    id: str
    tier: str
    interval: str
    amount: int
    currency: str


class PlansResponse(BaseModel):
    plans: list[PlanSchema]


class SubscriptionSchema(BaseModel):
    tier: str
    status: SubscriptionStatus
    interval: str
    amount: int
    currency: str
    current_period_end: datetime | None
    cancel_at_period_end: bool
