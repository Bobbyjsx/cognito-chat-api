import json
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from app.billing.entitlements import has_minimum_tier
from app.billing.models import SubscriptionDB, SubscriptionStatus
from app.billing.service import BillingService
from app.main import app


def test_plans_resolve():
    service = BillingService(repository=None, provider=None)
    plans = service._get_plans()
    assert plans["go"].amount == 599900
    assert plans["premium"].amount == 999900


def test_entitlements():
    free_sub = None
    go_sub = SubscriptionDB(
        user_id="user1",
        tier="go",
        status=SubscriptionStatus.ACTIVE,
        interval="monthly",
        amount=599900,
        currency="NGN",
        provider_customer_code="CUS",
        provider_subscription_code="SUB",
        provider_plan_code="PLN",
    )
    premium_sub = SubscriptionDB(
        user_id="user2",
        tier="premium",
        status=SubscriptionStatus.ACTIVE,
        interval="monthly",
        amount=999900,
        currency="NGN",
        provider_customer_code="CUS",
        provider_subscription_code="SUB",
        provider_plan_code="PLN",
    )

    assert has_minimum_tier(go_sub, "go")
    assert not has_minimum_tier(go_sub, "premium")
    assert has_minimum_tier(premium_sub, "go")
    assert has_minimum_tier(premium_sub, "premium")
    assert not has_minimum_tier(free_sub, "go")
    assert not has_minimum_tier(free_sub, "premium")

    expired_sub = go_sub.model_copy()
    expired_sub.status = SubscriptionStatus.EXPIRED
    assert not has_minimum_tier(expired_sub, "go")

    cancelled_sub = go_sub.model_copy()
    cancelled_sub.status = SubscriptionStatus.CANCELLED
    cancelled_sub.current_period_end = datetime.now(timezone.utc) + timedelta(days=5)
    assert has_minimum_tier(cancelled_sub, "go")

    cancelled_sub.current_period_end = datetime.now(timezone.utc) - timedelta(days=5)
    assert not has_minimum_tier(cancelled_sub, "go")


@pytest.mark.asyncio
async def test_webhook_signature():
    client = TestClient(app)

    payload = {"event": "charge.success", "data": {}}
    body = json.dumps(payload).encode("utf-8")

    # Missing signature
    resp = client.post("/webhooks/paystack", content=body)
    assert resp.status_code == 400

    # Invalid signature
    resp = client.post("/webhooks/paystack", content=body, headers={"x-paystack-signature": "invalid"})
    assert resp.status_code == 400

    # We would need to mock settings.paystack_webhook_secret or rely on defaults for proper test
