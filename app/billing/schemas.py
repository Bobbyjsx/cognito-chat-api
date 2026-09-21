from datetime import datetime

from pydantic import BaseModel

from app.billing.models import SubscriptionStatus


class CheckoutRequest(BaseModel):
    plan: str  # "go" or "premium"
    callback_url: str | None = None


class CheckoutResponse(BaseModel):
    checkout_url: str
    reference: str


class PlanSchema(BaseModel):
    id: str
    tier: str
    interval: str
    amount: int
    currency: str


class PlansResponse(BaseModel):
    plans: list[PlanSchema]


class ScheduledChangeSchema(BaseModel):
    target_tier: str
    effective_at: datetime
    status: str


class SubscriptionSchema(BaseModel):
    tier: str
    status: SubscriptionStatus
    interval: str
    amount: int
    currency: str
    current_period_end: datetime | None
    cancel_at_period_end: bool
    scheduled_change: ScheduledChangeSchema | None = None


class DowngradeRequest(BaseModel):
    plan: str  # e.g. "go"


class DowngradeResponse(BaseModel):
    current_plan: str
    scheduled_plan: str
    effective_at: datetime
