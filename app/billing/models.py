from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field


class SubscriptionStatus(str, Enum):
    PENDING = "pending"
    ACTIVE = "active"
    PAST_DUE = "past_due"
    CANCELLED = "cancelled"
    EXPIRED = "expired"


class BillingPlan(BaseModel):
    id: str
    product: str
    tier: str
    interval: str
    amount: int
    currency: str
    provider: str
    provider_plan_code: str


class SubscriptionDB(BaseModel):
    id: str
    user_id: str
    product: str = "cognito"
    tier: str
    status: SubscriptionStatus
    interval: str
    amount: int
    currency: str
    provider: str = "paystack"

    provider_customer_code: str
    provider_subscription_code: str
    provider_plan_code: str

    current_period_start: datetime | None = None
    current_period_end: datetime | None = None
    cancel_at_period_end: bool = False

    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
