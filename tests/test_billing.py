"""
Comprehensive billing tests for the Cognito Paystack billing integration.

Coverage:
  - Plan configuration (amounts, tiers, kobo conversion)
  - Entitlements (hierarchy, expired/cancelled states)
  - Billing API endpoints (plans, status, checkout, cancel)
  - Webhook signature verification (reject invalid, accept valid)
  - Webhook idempotency (duplicate events)
  - All Paystack event types:
      charge.success
      subscription.create
      subscription.enable
      subscription.disable
      invoice.payment_failed
      informational events (no state change)
      unknown events (no crash)
  - DB persistence (Firestore emulator)
  - Security: client cannot override amount / plan code
"""

import hashlib
import hmac
import json
import os
import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ──────────────────────────────────────────────────────────────────────────────
# Fixtures & helpers
# ──────────────────────────────────────────────────────────────────────────────

GO_PLAN_CODE = os.environ.get("GO_PAYSTACK_PLAN_CODE", "PLN_3v65wje2l7kin47")
PREMIUM_PLAN_CODE = os.environ.get("PREMIUM_PAYSTACK_PLAN_CODE", "PLN_sr1h7zo9qogclvb")
USER_ID = "test-user-billing-001"
USER_EMAIL = "billing@test.com"


TEST_SECRET = os.environ.get("PAYSTACK_SECRET_KEY", "")


def _sign(body: bytes, secret: str = TEST_SECRET) -> str:
    return hmac.new(secret.encode(), body, hashlib.sha512).hexdigest()


def _webhook(client, payload: dict, secret: str = TEST_SECRET):
    body = json.dumps(payload).encode()
    sig = _sign(body, secret)
    return client.post(
        "/webhooks/paystack",
        content=body,
        headers={"x-paystack-signature": sig, "Content-Type": "application/json"},
    )


def _charge_success_payload(
    user_id: str = USER_ID,
    plan_id: str = "go_monthly",
    plan_code: str = GO_PLAN_CODE,
    amount: int = 599900,
    ref: str | None = None,
) -> dict:
    ref = ref or f"ref_{uuid.uuid4().hex}"
    return {
        "event": "charge.success",
        "data": {
            "id": f"charge_{uuid.uuid4().hex}",
            "reference": ref,
            "amount": amount,
            "currency": "NGN",
            "paid_at": datetime.now(timezone.utc).isoformat(),
            "plan": {"plan_code": plan_code},
            "customer": {"customer_code": "CUS_test001", "email": USER_EMAIL},
            "metadata": {"user_id": user_id, "plan_id": plan_id},
        },
    }


def _subscription_create_payload(
    user_id: str = USER_ID,
    sub_code: str = "SUB_test001",
    plan_code: str = GO_PLAN_CODE,
) -> dict:
    return {
        "event": "subscription.create",
        "data": {
            "id": f"sub_{uuid.uuid4().hex}",
            "subscription_code": sub_code,
            "next_payment_date": (datetime.now(timezone.utc) + timedelta(days=30)).isoformat(),
            "plan": {"plan_code": plan_code},
            "customer": {
                "customer_code": "CUS_test001",
                "email": USER_EMAIL,
                "metadata": {"user_id": user_id},
            },
        },
    }


def _subscription_disable_payload(sub_code: str = "SUB_test001") -> dict:
    return {
        "event": "subscription.disable",
        "data": {
            "id": f"dis_{uuid.uuid4().hex}",
            "subscription_code": sub_code,
        },
    }


def _subscription_enable_payload(sub_code: str = "SUB_test001") -> dict:
    return {
        "event": "subscription.enable",
        "data": {
            "id": f"en_{uuid.uuid4().hex}",
            "subscription_code": sub_code,
            "next_payment_date": (datetime.now(timezone.utc) + timedelta(days=30)).isoformat(),
        },
    }


def _invoice_failed_payload(sub_code: str = "SUB_test001") -> dict:
    return {
        "event": "invoice.payment_failed",
        "data": {
            "id": f"inv_{uuid.uuid4().hex}",
            "subscription": {"subscription_code": sub_code},
        },
    }


# ──────────────────────────────────────────────────────────────────────────────
# AUTH helpers (re-use the conftest client fixture)
# ──────────────────────────────────────────────────────────────────────────────


def _signup_login(client, email: str = USER_EMAIL, pw: str = "pass123") -> str:
    client.post("/auth/signup", json={"email": email, "password": pw})
    resp = client.post("/auth/login", json={"email": email, "password": pw})
    return resp.json()["access_token"]


def _auth(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


# ──────────────────────────────────────────────────────────────────────────────
# 1. Plan configuration
# ──────────────────────────────────────────────────────────────────────────────


def test_synthetic_identity_email_is_not_usable():
    from app.core.identity import email_from_mapping, is_usable_email

    assert not is_usable_email("")
    assert not is_usable_email("not-an-email")
    assert not is_usable_email("user@auth.identity")
    assert not is_usable_email("b0dd30c5-be96-4d77-b011-9955179cfa32@auth.identity")
    assert is_usable_email("person@gmail.com")
    assert email_from_mapping({"email": "person@gmail.com"}) == "person@gmail.com"
    assert email_from_mapping({"user": {"email": "person@gmail.com"}}) == "person@gmail.com"
    assert email_from_mapping({"email": "x@auth.identity"}) is None


@pytest.mark.asyncio
async def test_create_checkout_rejects_synthetic_email():
    from fastapi import HTTPException

    from app.billing.service import BillingService

    provider = MagicMock()
    provider.initialize_subscription_checkout = AsyncMock()
    service = BillingService(None, provider)  # type: ignore[arg-type]
    with pytest.raises(HTTPException) as exc_info:
        await service.create_checkout("premium", USER_ID, "abc@auth.identity")
    assert exc_info.value.status_code == 400
    provider.initialize_subscription_checkout.assert_not_called()


def test_sanitize_callback_url_allows_billing_return_only():
    from app.billing.service import BillingService

    assert (
        BillingService._sanitize_callback_url("http://localhost:3000/settings/billing")
        == "http://localhost:3000/settings/billing"
    )
    assert BillingService._sanitize_callback_url("https://evil.example/phish") is None
    assert BillingService._sanitize_callback_url("javascript:alert(1)") is None


def test_go_plan_amount_is_5999_ngn():
    from app.billing.service import BillingService

    plans = BillingService(None, None)._get_plans()  # type: ignore[arg-type]
    assert plans["go"].amount == 599900  # kobo
    assert plans["go"].currency == "NGN"
    assert plans["go"].provider_plan_code == GO_PLAN_CODE


def test_premium_plan_amount_is_9999_ngn():
    from app.billing.service import BillingService

    plans = BillingService(None, None)._get_plans()  # type: ignore[arg-type]
    assert plans["premium"].amount == 999900  # kobo
    assert plans["premium"].currency == "NGN"
    assert plans["premium"].provider_plan_code == PREMIUM_PLAN_CODE


def test_plans_endpoint_returns_ngn_not_kobo(client):
    resp = client.get("/billing/plans")
    assert resp.status_code == 200
    plans = {p["tier"]: p for p in resp.json()["plans"]}
    assert plans["go"]["amount"] == 5999
    assert plans["premium"]["amount"] == 9999
    assert plans["go"]["currency"] == "NGN"
    assert "token_limit_6h" not in plans["go"]
    assert "token_limit_weekly" not in plans["go"]
    assert "token_limit_6h" not in plans["premium"]


# ──────────────────────────────────────────────────────────────────────────────
# 2. Entitlements
# ──────────────────────────────────────────────────────────────────────────────


def _make_sub(tier: str, status, period_end_offset_days: int = 30):
    from app.billing.models import SubscriptionDB

    return SubscriptionDB(
        id=str(uuid.uuid4()),
        user_id="u1",
        tier=tier,
        status=status,
        interval="monthly",
        amount=599900,
        currency="NGN",
        provider_customer_code="CUS",
        provider_subscription_code="SUB",
        provider_plan_code=GO_PLAN_CODE,
        current_period_end=datetime.now(timezone.utc) + timedelta(days=period_end_offset_days),
    )


def test_free_user_has_no_go_entitlement():
    from app.billing.entitlements import has_minimum_tier

    assert not has_minimum_tier(None, "go")


def test_free_user_has_no_premium_entitlement():
    from app.billing.entitlements import has_minimum_tier

    assert not has_minimum_tier(None, "premium")


def test_go_user_has_go_entitlement():
    from app.billing.entitlements import has_minimum_tier
    from app.billing.models import SubscriptionStatus

    sub = _make_sub("go", SubscriptionStatus.ACTIVE)
    assert has_minimum_tier(sub, "go")


def test_go_user_lacks_premium_entitlement():
    from app.billing.entitlements import has_minimum_tier
    from app.billing.models import SubscriptionStatus

    sub = _make_sub("go", SubscriptionStatus.ACTIVE)
    assert not has_minimum_tier(sub, "premium")


def test_premium_user_satisfies_go_entitlement():
    from app.billing.entitlements import has_minimum_tier
    from app.billing.models import SubscriptionStatus

    sub = _make_sub("premium", SubscriptionStatus.ACTIVE)
    assert has_minimum_tier(sub, "go")
    assert has_minimum_tier(sub, "premium")


def test_expired_subscription_loses_entitlement():
    from app.billing.entitlements import has_minimum_tier
    from app.billing.models import SubscriptionStatus

    sub = _make_sub("premium", SubscriptionStatus.EXPIRED)
    assert not has_minimum_tier(sub, "go")
    assert not has_minimum_tier(sub, "premium")


def test_past_due_subscription_loses_entitlement():
    from app.billing.entitlements import has_minimum_tier
    from app.billing.models import SubscriptionStatus

    sub = _make_sub("premium", SubscriptionStatus.PAST_DUE)
    assert not has_minimum_tier(sub, "premium")


def test_cancelled_with_future_period_end_retains_access():
    from app.billing.entitlements import has_minimum_tier
    from app.billing.models import SubscriptionStatus

    sub = _make_sub("go", SubscriptionStatus.CANCELLED, period_end_offset_days=5)
    assert has_minimum_tier(sub, "go")


def test_cancelled_with_past_period_end_loses_access():
    from app.billing.entitlements import has_minimum_tier
    from app.billing.models import SubscriptionStatus

    sub = _make_sub("go", SubscriptionStatus.CANCELLED, period_end_offset_days=-1)
    assert not has_minimum_tier(sub, "go")


# ──────────────────────────────────────────────────────────────────────────────
# 3. Billing API endpoints
# ──────────────────────────────────────────────────────────────────────────────


def test_billing_status_unauthenticated(client):
    resp = client.get("/billing")
    assert resp.status_code == 401


def test_billing_status_free_user(client):
    token = _signup_login(client, "free@test.com")
    resp = client.get("/billing", headers=_auth(token))
    assert resp.status_code == 200
    data = resp.json()
    assert data["tier"] == "free"


def test_checkout_invalid_plan_rejected(client):
    token = _signup_login(client, "checkout_bad@test.com")
    resp = client.post("/billing/checkout", json={"plan": "enterprise"}, headers=_auth(token))
    assert resp.status_code == 400


def test_checkout_unauthenticated_rejected(client):
    resp = client.post("/billing/checkout", json={"plan": "go"})
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_cancel_without_subscription_code_backfills_from_customer():
    from app.billing.models import SubscriptionDB, SubscriptionStatus
    from app.billing.service import BillingService

    sub = SubscriptionDB(
        id="sub1",
        user_id=USER_ID,
        tier="premium",
        status=SubscriptionStatus.ACTIVE,
        interval="monthly",
        amount=999900,
        currency="NGN",
        provider_customer_code="CUS_live",
        provider_subscription_code="",
        provider_plan_code=PREMIUM_PLAN_CODE,
    )
    repo = MagicMock()
    repo.get_by_user_id = AsyncMock(return_value=sub)
    repo.save = AsyncMock(return_value=sub)
    provider = MagicMock()
    provider.find_customer_subscription = AsyncMock(
        return_value={"subscription_code": "SUB_live", "email_token": "etok"}
    )
    provider.cancel_subscription = AsyncMock(return_value=True)

    service = BillingService(repo, provider)
    assert await service.cancel_subscription(USER_ID, USER_EMAIL) is True
    provider.cancel_subscription.assert_awaited_once_with("SUB_live", "etok")


@pytest.mark.asyncio
async def test_checkout_without_paystack_secret_returns_503():
    from fastapi import HTTPException

    from app.billing.models import BillingPlan
    from app.billing.providers.paystack import PaystackProvider

    provider = PaystackProvider()
    provider.secret_key = ""
    plan = BillingPlan(
        id="go_monthly",
        product="cognito",
        tier="go",
        interval="monthly",
        amount=599900,
        currency="NGN",
        provider="paystack",
        provider_plan_code=GO_PLAN_CODE,
    )
    with pytest.raises(HTTPException) as exc_info:
        await provider.initialize_subscription_checkout("a@b.com", plan, USER_ID)
    assert exc_info.value.status_code == 503


def test_checkout_go_calls_paystack(client):
    """Checkout initializes with Paystack and returns checkout_url."""
    token = _signup_login(client, "checkout_go@test.com")

    mock_provider = MagicMock()
    mock_provider.initialize_subscription_checkout = AsyncMock(
        return_value={"authorization_url": "https://checkout.paystack.com/abc", "reference": "ref_abc"}
    )

    with patch("app.router.billing.PaystackProvider", return_value=mock_provider):
        resp = client.post("/billing/checkout", json={"plan": "go"}, headers=_auth(token))

    assert resp.status_code == 200
    assert resp.json()["checkout_url"] == "https://checkout.paystack.com/abc"
    assert resp.json()["reference"] == "ref_abc"


def test_checkout_premium_calls_paystack(client):
    token = _signup_login(client, "checkout_premium@test.com")

    mock_provider = MagicMock()
    mock_provider.initialize_subscription_checkout = AsyncMock(
        return_value={"authorization_url": "https://checkout.paystack.com/xyz", "reference": "ref_xyz"}
    )

    with patch("app.router.billing.PaystackProvider", return_value=mock_provider):
        resp = client.post("/billing/checkout", json={"plan": "premium"}, headers=_auth(token))

    assert resp.status_code == 200


def test_client_cannot_override_amount(client):
    """Any amount or plan_code fields from the client must be ignored."""
    token = _signup_login(client, "hacker@test.com")

    mock_provider = MagicMock()
    mock_provider.initialize_subscription_checkout = AsyncMock(
        return_value={"authorization_url": "https://paystack.com/x", "reference": "ref_x"}
    )

    with patch("app.router.billing.PaystackProvider", return_value=mock_provider):
        resp = client.post(
            "/billing/checkout",
            # Attempt to inject amount / plan_code – should be ignored
            json={"plan": "go", "amount": 1, "plan_code": "PLN_FAKE", "price": 0},
            headers=_auth(token),
        )

    assert resp.status_code == 200
    # Provider must have been called with the server-resolved plan (Go = 599900 kobo)
    call_args = mock_provider.initialize_subscription_checkout.call_args
    plan_arg = call_args[0][1]  # positional arg: plan (BillingPlan)
    assert plan_arg.amount == 599900
    assert plan_arg.provider_plan_code == GO_PLAN_CODE


# ──────────────────────────────────────────────────────────────────────────────
# 4. Webhook signature security
# ──────────────────────────────────────────────────────────────────────────────


def test_webhook_missing_signature_rejected(client):
    body = json.dumps({"event": "charge.success", "data": {}}).encode()
    resp = client.post("/webhooks/paystack", content=body)
    assert resp.status_code == 400


def test_webhook_wrong_signature_rejected(client):
    body = json.dumps({"event": "charge.success", "data": {}}).encode()
    resp = client.post(
        "/webhooks/paystack",
        content=body,
        headers={"x-paystack-signature": "deadbeef", "Content-Type": "application/json"},
    )
    assert resp.status_code == 400


def test_webhook_tampered_body_rejected(client):
    """Sign original body, then tamper before sending."""
    original = json.dumps({"event": "charge.success", "data": {}}).encode()
    sig = _sign(original)
    tampered = json.dumps({"event": "charge.success", "data": {"injected": True}}).encode()
    resp = client.post(
        "/webhooks/paystack",
        content=tampered,
        headers={"x-paystack-signature": sig, "Content-Type": "application/json"},
    )
    assert resp.status_code == 400


def test_webhook_valid_signature_accepted(client):
    # An event we don't act on (informational)
    resp = _webhook(client, {"event": "customeridentification", "data": {"id": "e1"}})
    assert resp.status_code == 200
    assert resp.json()["status"] == "success"


# ──────────────────────────────────────────────────────────────────────────────
# 5. Idempotency
# ──────────────────────────────────────────────────────────────────────────────


def test_duplicate_charge_success_is_idempotent(client):
    """Sending the same charge.success twice must not create duplicate records."""
    payload = _charge_success_payload(user_id=f"idem-{uuid.uuid4().hex}")
    _webhook(client, payload)
    _webhook(client, payload)  # duplicate

    # Verify no exception and a single record in db would be validated via DB test below


def test_unknown_event_does_not_crash(client):
    resp = _webhook(client, {"event": "paystack.future_event_type", "data": {"id": "x1"}})
    assert resp.status_code == 200
    assert resp.json()["status"] == "success"


def test_informational_events_accepted(client):
    for event in ["invoice.create", "invoice.update", "subscription.expiry_cards", "customeridentification"]:
        resp = _webhook(client, {"event": event, "data": {"id": f"info_{uuid.uuid4().hex}"}})
        assert resp.status_code == 200, f"Failed for event: {event}"


# ──────────────────────────────────────────────────────────────────────────────
# 6. DB persistence – full event lifecycle (requires Firestore emulator)
# ──────────────────────────────────────────────────────────────────────────────


def _direct_db():
    """Return a synchronous Firestore client pointing at the emulator."""
    from google.cloud import firestore

    return firestore.Client(project="test-project")


def _get_sub_by_user(user_id: str):
    db = _direct_db()
    docs = (
        db.collection("subscriptions")
        .where("user_id", "==", user_id)
        .order_by("created_at", direction="DESCENDING")
        .limit(1)
        .get()
    )
    if not docs:
        return None
    d = docs[0].to_dict()
    d["id"] = docs[0].id
    return d


def _get_event(event_id: str):
    db = _direct_db()
    doc = db.collection("payment_events").document(event_id).get()
    return doc.to_dict() if doc.exists else None


def test_charge_success_creates_active_subscription_in_db(client):
    uid = f"db-cs-{uuid.uuid4().hex}"
    payload = _charge_success_payload(user_id=uid)
    resp = _webhook(client, payload)
    assert resp.status_code == 200

    sub = _get_sub_by_user(uid)
    assert sub is not None, "Subscription not persisted after charge.success"
    assert sub["user_id"] == uid
    assert sub["status"] == "active"
    assert sub["tier"] == "go"
    assert sub["provider"] == "paystack"
    assert sub["provider_customer_code"] == "CUS_test001"
    assert sub["provider_plan_code"] == GO_PLAN_CODE
    assert sub["current_period_end"] is not None


def test_charge_success_premium_creates_premium_subscription(client):
    uid = f"db-cs-prem-{uuid.uuid4().hex}"
    payload = _charge_success_payload(
        user_id=uid,
        plan_id="premium_monthly",
        plan_code=PREMIUM_PLAN_CODE,
        amount=999900,
    )
    resp = _webhook(client, payload)
    assert resp.status_code == 200

    sub = _get_sub_by_user(uid)
    assert sub is not None
    assert sub["tier"] == "premium"
    assert sub["status"] == "active"


def test_charge_success_amount_wins_over_mismatched_plan_code(client):
    """A ₦9,999 charge must be Premium even if Paystack echoes the Go plan code."""
    uid = f"db-cs-mismatch-{uuid.uuid4().hex}"
    payload = _charge_success_payload(
        user_id=uid,
        plan_id="premium_monthly",
        plan_code=GO_PLAN_CODE,
        amount=999900,
    )
    resp = _webhook(client, payload)
    assert resp.status_code == 200
    sub = _get_sub_by_user(uid)
    assert sub is not None
    assert sub["tier"] == "premium"
    assert sub["amount"] == 999900


def test_subscription_create_before_charge_success_keeps_sub_code(client):
    uid = f"db-race-{uuid.uuid4().hex}"
    cus = f"CUS_{uuid.uuid4().hex[:8]}"
    sub_code = f"SUB_{uuid.uuid4().hex[:8]}"

    create = _subscription_create_payload(sub_code=sub_code)
    create["data"]["customer"] = {"customer_code": cus, "email": USER_EMAIL, "metadata": {}}
    assert _webhook(client, create).status_code == 200

    charge = _charge_success_payload(
        user_id=uid,
        plan_id="premium_monthly",
        plan_code=PREMIUM_PLAN_CODE,
        amount=999900,
    )
    charge["data"]["customer"]["customer_code"] = cus
    assert _webhook(client, charge).status_code == 200

    sub = _get_sub_by_user(uid)
    assert sub is not None
    assert sub["tier"] == "premium"
    assert sub["provider_subscription_code"] == sub_code


def test_subscription_create_links_sub_code_to_user(client):
    uid = f"db-sc-{uuid.uuid4().hex}"
    sub_code = f"SUB_{uuid.uuid4().hex[:8]}"

    # First charge.success to create the record
    _webhook(client, _charge_success_payload(user_id=uid))

    # Then subscription.create to link the code
    resp = _webhook(client, _subscription_create_payload(user_id=uid, sub_code=sub_code))
    assert resp.status_code == 200

    sub = _get_sub_by_user(uid)
    assert sub is not None
    assert sub["provider_subscription_code"] == sub_code
    assert sub["status"] == "active"


def test_subscription_disable_marks_cancelled(client):
    uid = f"db-dis-{uuid.uuid4().hex}"
    sub_code = f"SUB_{uuid.uuid4().hex[:8]}"

    _webhook(client, _charge_success_payload(user_id=uid))
    _webhook(client, _subscription_create_payload(user_id=uid, sub_code=sub_code))
    resp = _webhook(client, _subscription_disable_payload(sub_code=sub_code))
    assert resp.status_code == 200

    sub = _get_sub_by_user(uid)
    assert sub is not None
    assert sub["status"] == "cancelled"
    assert sub["cancel_at_period_end"] is True


def test_subscription_enable_reactivates_cancelled(client):
    uid = f"db-en-{uuid.uuid4().hex}"
    sub_code = f"SUB_{uuid.uuid4().hex[:8]}"

    _webhook(client, _charge_success_payload(user_id=uid))
    _webhook(client, _subscription_create_payload(user_id=uid, sub_code=sub_code))
    _webhook(client, _subscription_disable_payload(sub_code=sub_code))
    # Verify cancelled
    assert _get_sub_by_user(uid)["status"] == "cancelled"

    resp = _webhook(client, _subscription_enable_payload(sub_code=sub_code))
    assert resp.status_code == 200

    sub = _get_sub_by_user(uid)
    assert sub["status"] == "active"
    assert sub["cancel_at_period_end"] is False


def test_invoice_payment_failed_sets_past_due(client):
    uid = f"db-pf-{uuid.uuid4().hex}"
    sub_code = f"SUB_{uuid.uuid4().hex[:8]}"

    _webhook(client, _charge_success_payload(user_id=uid))
    _webhook(client, _subscription_create_payload(user_id=uid, sub_code=sub_code))

    resp = _webhook(client, _invoice_failed_payload(sub_code=sub_code))
    assert resp.status_code == 200

    sub = _get_sub_by_user(uid)
    assert sub is not None
    assert sub["status"] == "past_due"


def test_full_renewal_failure_then_cancel_lifecycle(client):
    """Simulate: success → sub linked → renewal fails → disabled by Paystack."""
    uid = f"db-full-{uuid.uuid4().hex}"
    sub_code = f"SUB_{uuid.uuid4().hex[:8]}"

    _webhook(client, _charge_success_payload(user_id=uid))
    _webhook(client, _subscription_create_payload(user_id=uid, sub_code=sub_code))
    _webhook(client, _invoice_failed_payload(sub_code=sub_code))

    assert _get_sub_by_user(uid)["status"] == "past_due"

    # Paystack exhausts retries → sends subscription.disable
    _webhook(client, _subscription_disable_payload(sub_code=sub_code))

    sub = _get_sub_by_user(uid)
    assert sub["status"] == "cancelled"
    assert sub["cancel_at_period_end"] is True


def test_duplicate_event_is_idempotent_in_db(client):
    """Sending the same charge.success twice must result in exactly one DB record."""
    uid = f"db-idem-{uuid.uuid4().hex}"
    payload = _charge_success_payload(user_id=uid)

    _webhook(client, payload)
    _webhook(client, payload)  # exact duplicate

    # Verify only one subscription document
    db = _direct_db()
    docs = db.collection("subscriptions").where("user_id", "==", uid).get()
    assert len(docs) == 1

    # Verify the event is marked processed
    charge_id = payload["data"]["id"]
    event_id = f"paystack_{charge_id}_charge.success"
    event_doc = _get_event(event_id)
    assert event_doc is not None, "Event was not recorded as processed"


def test_payment_event_marked_processed(client):
    """payment_events/{eventId} must be written after first processing."""
    uid = f"db-ev-{uuid.uuid4().hex}"
    payload = _charge_success_payload(user_id=uid)
    charge_id = payload["data"]["id"]

    _webhook(client, payload)

    event_id = f"paystack_{charge_id}_charge.success"
    doc = _get_event(event_id)
    assert doc is not None
    assert "processed_at" in doc


# ──────────────────────────────────────────────────────────────────────────────
# 7. Billing status endpoint reflects webhook state
# ──────────────────────────────────────────────────────────────────────────────


def _real_user_id(client, token: str) -> str:
    """Return the actual Firestore user ID for a logged-in user."""
    resp = client.get("/auth/me", headers=_auth(token))
    assert resp.status_code == 200
    return str(resp.json()["id"])


def test_billing_status_reflects_active_subscription(client):
    email = f"status-active-{uuid.uuid4().hex}@test.com"
    token = _signup_login(client, email)
    uid = _real_user_id(client, token)

    # Inject subscription via webhook using the real user ID
    payload = _charge_success_payload(user_id=uid)
    _webhook(client, payload)

    resp = client.get("/billing", headers=_auth(token))
    assert resp.status_code == 200
    data = resp.json()
    assert data["tier"] == "go"
    assert data["status"] == "active"
    assert data["amount"] == 5999


def test_billing_status_reflects_cancelled_subscription(client):
    email = f"status-cancel-{uuid.uuid4().hex}@test.com"
    token = _signup_login(client, email)
    uid = _real_user_id(client, token)
    sub_code = f"SUB_{uuid.uuid4().hex[:8]}"

    _webhook(client, _charge_success_payload(user_id=uid))
    _webhook(client, _subscription_create_payload(user_id=uid, sub_code=sub_code))
    _webhook(client, _subscription_disable_payload(sub_code=sub_code))

    resp = client.get("/billing", headers=_auth(token))
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "cancelled"
    assert data["cancel_at_period_end"] is True


def test_billing_status_reflects_past_due(client):
    email = f"status-pd-{uuid.uuid4().hex}@test.com"
    token = _signup_login(client, email)
    uid = _real_user_id(client, token)
    sub_code = f"SUB_{uuid.uuid4().hex[:8]}"

    _webhook(client, _charge_success_payload(user_id=uid))
    _webhook(client, _subscription_create_payload(user_id=uid, sub_code=sub_code))
    _webhook(client, _invoice_failed_payload(sub_code=sub_code))

    resp = client.get("/billing", headers=_auth(token))
    assert resp.status_code == 200
    assert resp.json()["status"] == "past_due"


# ──────────────────────────────────────────────────────────────────────────────
# Upgrade / Downgrade Refinements
# ──────────────────────────────────────────────────────────────────────────────


def test_upgrade_to_premium_cancels_old_go_renewal(client):
    uid = f"db-upgrade-{uuid.uuid4().hex}"
    cus = f"CUS_{uuid.uuid4().hex[:8]}"
    old_sub = f"SUB_oldgo_{uuid.uuid4().hex[:8]}"
    new_sub = f"SUB_newprem_{uuid.uuid4().hex[:8]}"

    # Setup initial Go subscription
    create = _subscription_create_payload(user_id=uid, sub_code=old_sub, plan_code=GO_PLAN_CODE)
    create["data"]["customer"] = {"customer_code": cus, "email": USER_EMAIL, "metadata": {"user_id": uid}}
    assert _webhook(client, create).status_code == 200

    # User upgrades to Premium
    charge = _charge_success_payload(
        user_id=uid,
        plan_id="premium_monthly",
        plan_code=PREMIUM_PLAN_CODE,
        amount=999900,
    )
    charge["data"]["customer"]["customer_code"] = cus
    charge["data"]["subscription_code"] = new_sub

    mock_provider = MagicMock()
    mock_provider.get_subscription = AsyncMock(return_value={"email_token": "token"})
    mock_provider.cancel_subscription = AsyncMock(return_value=True)

    with patch("app.billing.providers.paystack.PaystackProvider", return_value=mock_provider):
        assert _webhook(client, charge).status_code == 200

    # It should have cancelled the OLD subscription!
    mock_provider.cancel_subscription.assert_called_once_with(old_sub, "token")

    # DB should reflect Premium
    sub = _get_sub_by_user(uid)
    assert sub["tier"] == "premium"
    assert sub["provider_subscription_code"] == new_sub


def test_downgrade_to_go_defers_transition(client):
    uid = f"db-downgrade-{uuid.uuid4().hex}"
    cus = f"CUS_{uuid.uuid4().hex[:8]}"
    prem_sub = f"SUB_prem_{uuid.uuid4().hex[:8]}"

    # Setup initial Premium subscription
    create = _subscription_create_payload(user_id=uid, sub_code=prem_sub, plan_code=PREMIUM_PLAN_CODE)
    create["data"]["customer"] = {"customer_code": cus, "email": USER_EMAIL, "metadata": {"user_id": uid}}
    assert _webhook(client, create).status_code == 200

    sub = _get_sub_by_user(uid)
    assert sub["tier"] == "premium"

    # User schedules downgrade to Go
    token = _signup_login(client, USER_EMAIL)
    # mock get_user_by_email to return the right user, or use the token user id
    # Since _signup_login creates a user, let's use its id.
    uid = client.get("/auth/me", headers=_auth(token)).json()["id"]

    # recreate premium sub with real user id
    create = _subscription_create_payload(user_id=uid, sub_code=prem_sub, plan_code=PREMIUM_PLAN_CODE)
    create["data"]["customer"] = {"customer_code": cus, "email": USER_EMAIL, "metadata": {"user_id": uid}}
    assert _webhook(client, create).status_code == 200

    mock_provider = MagicMock()
    mock_provider.get_subscription = AsyncMock(
        return_value={"email_token": "token", "authorization": {"authorization_code": "auth123"}}
    )
    mock_provider.cancel_subscription = AsyncMock(return_value=True)
    mock_provider.create_subscription = AsyncMock(return_value={"subscription_code": "SUB_futurego"})

    with patch("app.router.billing.PaystackProvider", return_value=mock_provider):
        resp = client.post("/billing/subscription/downgrade", json={"plan": "go"}, headers=_auth(token))
        assert resp.status_code == 200

    # It should have cancelled renewal of Premium
    mock_provider.cancel_subscription.assert_called_once_with(prem_sub, "token")
    # It should have created a future Go sub
    mock_provider.create_subscription.assert_called_once()

    # DB should still be Premium, but with scheduled_change
    sub = _get_sub_by_user(uid)
    assert sub["tier"] == "premium"
    assert sub["scheduled_change"]["target_tier"] == "go"
    assert sub["scheduled_change"]["status"] == "pending"
    assert sub["scheduled_change"]["provider_subscription_code"] == "SUB_futurego"


def test_cancel_downgrade_restores_premium(client):
    # Setup user
    token = _signup_login(client, "cancel_dg@test.com")
    uid = client.get("/auth/me", headers=_auth(token)).json()["id"]
    cus = f"CUS_{uuid.uuid4().hex[:8]}"
    prem_sub = f"SUB_prem_{uuid.uuid4().hex[:8]}"

    # Setup Premium
    create = _subscription_create_payload(user_id=uid, sub_code=prem_sub, plan_code=PREMIUM_PLAN_CODE)
    create["data"]["customer"] = {"customer_code": cus, "email": "cancel_dg@test.com", "metadata": {"user_id": uid}}
    _webhook(client, create)

    # Downgrade to Go
    mock_provider = MagicMock()
    mock_provider.get_subscription = AsyncMock(
        return_value={"email_token": "token", "authorization": {"authorization_code": "auth123"}}
    )
    mock_provider.cancel_subscription = AsyncMock(return_value=True)
    mock_provider.create_subscription = AsyncMock(return_value={"subscription_code": "SUB_futurego"})

    with patch("app.router.billing.PaystackProvider", return_value=mock_provider):
        client.post("/billing/subscription/downgrade", json={"plan": "go"}, headers=_auth(token))

    # Now Cancel the Downgrade
    mock_provider.enable_subscription = AsyncMock(return_value=True)
    with patch("app.router.billing.PaystackProvider", return_value=mock_provider):
        resp = client.post("/billing/subscription/downgrade/cancel", headers=_auth(token))
        assert resp.status_code == 200

    # DB should be Premium with NO scheduled_change
    sub = _get_sub_by_user(uid)
    assert sub["tier"] == "premium"
    assert sub.get("scheduled_change") is None

    # Future Go should be cancelled, Premium should be re-enabled
    mock_provider.cancel_subscription.assert_called_with("SUB_futurego", "token")
    mock_provider.enable_subscription.assert_called_with(prem_sub, "token")


def test_downgrade_from_cancelled_succeeds(client):
    # Setup user
    token = _signup_login(client, "cancel_then_dg@test.com")
    uid = client.get("/auth/me", headers=_auth(token)).json()["id"]
    cus = f"CUS_{uuid.uuid4().hex[:8]}"
    prem_sub = f"SUB_prem_{uuid.uuid4().hex[:8]}"

    # Setup Premium
    create = _subscription_create_payload(user_id=uid, sub_code=prem_sub, plan_code=PREMIUM_PLAN_CODE)
    create["data"]["customer"] = {
        "customer_code": cus,
        "email": "cancel_then_dg@test.com",
        "metadata": {"user_id": uid},
    }
    _webhook(client, create)

    # Cancel Premium
    mock_provider = MagicMock()
    mock_provider.get_subscription = AsyncMock(
        return_value={"email_token": "token", "authorization": {"authorization_code": "auth123"}}
    )
    mock_provider.cancel_subscription = AsyncMock(return_value=True)
    with patch("app.router.billing.PaystackProvider", return_value=mock_provider):
        resp = client.post("/billing/subscription/cancel", headers=_auth(token))
        assert resp.status_code == 200

    # Ensure it's cancelled
    sub = _get_sub_by_user(uid)
    assert sub["status"] == "cancelled"
    assert sub["cancel_at_period_end"] is True

    # Now downgrade to Go
    mock_provider.create_subscription = AsyncMock(return_value={"subscription_code": "SUB_futurego"})
    with patch("app.router.billing.PaystackProvider", return_value=mock_provider):
        resp = client.post("/billing/subscription/downgrade", json={"plan": "go"}, headers=_auth(token))
        assert resp.status_code == 200

    # DB should preserve cancelled status, but record scheduled_change
    sub = _get_sub_by_user(uid)
    assert sub["status"] == "cancelled"
    assert sub["cancel_at_period_end"] is True
    assert sub["tier"] == "premium"
    assert sub["scheduled_change"]["target_tier"] == "go"
    assert sub["scheduled_change"]["status"] == "pending"

    # Now cancel the downgrade schedule
    mock_provider.enable_subscription = AsyncMock()
    with patch("app.router.billing.PaystackProvider", return_value=mock_provider):
        resp = client.post("/billing/subscription/downgrade/cancel", headers=_auth(token))
        assert resp.status_code == 200

    # DB should still be cancelled (not active!), and scheduled_change is gone
    sub = _get_sub_by_user(uid)
    assert sub["status"] == "cancelled"
    assert sub["cancel_at_period_end"] is True
    assert sub.get("scheduled_change") is None

    # Should NOT have called enable_subscription on Paystack!
    mock_provider.enable_subscription.assert_not_called()


def test_downgrade_to_same_plan_rejected(client):
    # Setup user
    token = _signup_login(client, "same_dg@test.com")
    uid = client.get("/auth/me", headers=_auth(token)).json()["id"]
    cus = f"CUS_{uuid.uuid4().hex[:8]}"
    go_sub = f"SUB_go_{uuid.uuid4().hex[:8]}"

    # Setup Go
    create = _subscription_create_payload(user_id=uid, sub_code=go_sub, plan_code=GO_PLAN_CODE)
    create["data"]["customer"] = {"customer_code": cus, "email": "same_dg@test.com", "metadata": {"user_id": uid}}
    _webhook(client, create)

    # Downgrade to Go
    mock_provider = MagicMock()
    mock_provider.get_subscription = AsyncMock(
        return_value={"email_token": "token", "authorization": {"authorization_code": "auth123"}}
    )
    mock_provider.cancel_subscription = AsyncMock(return_value=True)

    with patch("app.router.billing.PaystackProvider", return_value=mock_provider):
        resp = client.post("/billing/subscription/downgrade", json={"plan": "go"}, headers=_auth(token))
        assert resp.status_code == 400
        assert "Already on this plan" in resp.json()["detail"]


def test_verify_transaction_activates_subscription(client):
    email = f"verify-{uuid.uuid4().hex}@test.com"
    token = _signup_login(client, email)
    uid = _real_user_id(client, token)
    cus = f"CUS_{uuid.uuid4().hex[:8]}"
    ref = f"T_{uuid.uuid4().hex[:12]}"
    sub_code = f"SUB_{uuid.uuid4().hex[:8]}"

    tx_data = {
        "status": "success",
        "reference": ref,
        "amount": 599900,
        "currency": "NGN",
        "paid_at": "2026-09-20T18:00:00.000Z",
        "plan": GO_PLAN_CODE,
        "metadata": {"user_id": uid, "plan_id": "go_monthly"},
        "customer": {"customer_code": cus, "email": email},
    }

    mock_provider = MagicMock()
    mock_provider.verify_transaction = AsyncMock(return_value=tx_data)
    mock_provider.find_customer_subscription = AsyncMock(return_value={"subscription_code": sub_code})

    with patch("app.router.billing.PaystackProvider", return_value=mock_provider):
        resp = client.get(f"/billing/verify?reference={ref}", headers=_auth(token))
        assert resp.status_code == 200
        data = resp.json()
        assert data["tier"] == "go"
        assert data["status"] == "active"
        assert data["amount"] == 5999

    # Verify database
    sub = _get_sub_by_user(uid)
    assert sub is not None
    assert sub["tier"] == "go"
    assert sub["status"] == "active"
    assert sub["provider_subscription_code"] == sub_code


def test_downgrade_idempotency_returns_existing_pending(client):
    """Calling downgrade multiple times for the same plan returns the existing pending change without duplicate sub creation."""
    token = _signup_login(client, "idempotent_dg@test.com")
    uid = client.get("/auth/me", headers=_auth(token)).json()["id"]
    cus = f"CUS_{uuid.uuid4().hex[:8]}"
    prem_sub = f"SUB_prem_{uuid.uuid4().hex[:8]}"

    create = _subscription_create_payload(user_id=uid, sub_code=prem_sub, plan_code=PREMIUM_PLAN_CODE)
    create["data"]["customer"] = {"customer_code": cus, "email": "idempotent_dg@test.com", "metadata": {"user_id": uid}}
    _webhook(client, create)

    mock_provider = MagicMock()
    mock_provider.get_subscription = AsyncMock(
        return_value={"email_token": "token", "authorization": {"authorization_code": "auth123"}}
    )
    mock_provider.cancel_subscription = AsyncMock(return_value=True)
    mock_provider.create_subscription = AsyncMock(return_value={"subscription_code": "SUB_futurego_once"})

    with patch("app.router.billing.PaystackProvider", return_value=mock_provider):
        resp1 = client.post("/billing/subscription/downgrade", json={"plan": "go"}, headers=_auth(token))
        assert resp1.status_code == 200
        assert resp1.json()["scheduled_plan"] == "go"

        # Second call to downgrade to Go
        resp2 = client.post("/billing/subscription/downgrade", json={"plan": "go"}, headers=_auth(token))
        assert resp2.status_code == 200
        assert resp2.json()["scheduled_plan"] == "go"

    # create_subscription must only be called ONCE
    mock_provider.create_subscription.assert_called_once()


def test_cancel_subscription_cancels_pending_future_downgrade(client):
    """When a user fully cancels while a downgrade is scheduled, the future sub on Paystack must also be cancelled."""
    token = _signup_login(client, "cancel_with_dg@test.com")
    uid = client.get("/auth/me", headers=_auth(token)).json()["id"]
    cus = f"CUS_{uuid.uuid4().hex[:8]}"
    prem_sub = f"SUB_prem_{uuid.uuid4().hex[:8]}"

    create = _subscription_create_payload(user_id=uid, sub_code=prem_sub, plan_code=PREMIUM_PLAN_CODE)
    create["data"]["customer"] = {
        "customer_code": cus,
        "email": "cancel_with_dg@test.com",
        "metadata": {"user_id": uid},
    }
    _webhook(client, create)

    mock_provider = MagicMock()
    mock_provider.get_subscription = AsyncMock(
        return_value={"email_token": "token", "authorization": {"authorization_code": "auth123"}}
    )
    mock_provider.cancel_subscription = AsyncMock(return_value=True)
    mock_provider.create_subscription = AsyncMock(return_value={"subscription_code": "SUB_future_cancel"})

    with patch("app.router.billing.PaystackProvider", return_value=mock_provider):
        client.post("/billing/subscription/downgrade", json={"plan": "go"}, headers=_auth(token))

        # Now full cancel
        resp = client.post("/billing/subscription/cancel", headers=_auth(token))
        assert resp.status_code == 200

    # Both current and future subs must have cancel_subscription called
    mock_provider.cancel_subscription.assert_any_call(prem_sub, "token")
    mock_provider.cancel_subscription.assert_any_call("SUB_future_cancel", "token")

    sub = _get_sub_by_user(uid)
    assert sub["status"] == "cancelled"
    assert sub["cancel_at_period_end"] is True
    assert sub.get("scheduled_change") is None


def test_schedule_downgrade_formats_iso_without_microseconds(client):
    """start_date passed to Paystack create_subscription must NOT contain microseconds (Paystack requirement)."""
    token = _signup_login(client, "iso_format@test.com")
    uid = client.get("/auth/me", headers=_auth(token)).json()["id"]
    cus = f"CUS_{uuid.uuid4().hex[:8]}"
    prem_sub = f"SUB_prem_{uuid.uuid4().hex[:8]}"

    create = _subscription_create_payload(user_id=uid, sub_code=prem_sub, plan_code=PREMIUM_PLAN_CODE)
    create["data"]["customer"] = {"customer_code": cus, "email": "iso_format@test.com", "metadata": {"user_id": uid}}
    _webhook(client, create)

    mock_provider = MagicMock()
    mock_provider.get_subscription = AsyncMock(
        return_value={"email_token": "token", "authorization": {"authorization_code": "auth123"}}
    )
    mock_provider.cancel_subscription = AsyncMock(return_value=True)
    mock_provider.create_subscription = AsyncMock(return_value={"subscription_code": "SUB_futurego"})

    with patch("app.router.billing.PaystackProvider", return_value=mock_provider):
        resp = client.post("/billing/subscription/downgrade", json={"plan": "go"}, headers=_auth(token))
        assert resp.status_code == 200

    call_kwargs = mock_provider.create_subscription.call_args.kwargs
    start_date = call_kwargs.get("start_date")
    assert start_date is not None
    # Must end with Z (UTC) and have NO microseconds dot
    assert start_date.endswith("Z")
    assert "." not in start_date


def test_cancel_downgrade_resilient_to_paystack_enable_error(client):
    """If Paystack throws on enable_subscription (e.g. 400 Bad Request already enabled), cancel downgrade must still succeed."""
    token = _signup_login(client, "resilient_cancel@test.com")
    uid = client.get("/auth/me", headers=_auth(token)).json()["id"]
    cus = f"CUS_{uuid.uuid4().hex[:8]}"
    prem_sub = f"SUB_prem_{uuid.uuid4().hex[:8]}"

    create = _subscription_create_payload(user_id=uid, sub_code=prem_sub, plan_code=PREMIUM_PLAN_CODE)
    create["data"]["customer"] = {
        "customer_code": cus,
        "email": "resilient_cancel@test.com",
        "metadata": {"user_id": uid},
    }
    _webhook(client, create)

    mock_provider = MagicMock()
    mock_provider.get_subscription = AsyncMock(
        return_value={"email_token": "token", "authorization": {"authorization_code": "auth123"}}
    )
    mock_provider.cancel_subscription = AsyncMock(return_value=True)
    mock_provider.create_subscription = AsyncMock(return_value={"subscription_code": "SUB_futurego"})

    with patch("app.router.billing.PaystackProvider", return_value=mock_provider):
        client.post("/billing/subscription/downgrade", json={"plan": "go"}, headers=_auth(token))

    # Now cancel downgrade, but mock enable_subscription raising an exception
    mock_provider.enable_subscription = AsyncMock(side_effect=Exception("400 Bad Request already enabled"))
    with patch("app.router.billing.PaystackProvider", return_value=mock_provider):
        resp = client.post("/billing/subscription/downgrade/cancel", headers=_auth(token))
        assert resp.status_code == 200

    sub = _get_sub_by_user(uid)
    assert sub["tier"] == "premium"
    assert sub.get("scheduled_change") is None


def test_verify_transaction_rejects_wrong_user(client):
    """A user cannot verify another user's transaction reference."""
    token1 = _signup_login(client, "user1_verify@test.com")
    token2 = _signup_login(client, "user2_verify@test.com")
    uid1 = _real_user_id(client, token1)

    ref = f"T_{uuid.uuid4().hex[:12]}"
    tx_data = {
        "status": "success",
        "reference": ref,
        "amount": 599900,
        "currency": "NGN",
        "metadata": {"user_id": uid1, "plan_id": "go_monthly"},
        "customer": {"customer_code": "CUS_xyz", "email": "user1_verify@test.com"},
    }

    mock_provider = MagicMock()
    mock_provider.verify_transaction = AsyncMock(return_value=tx_data)

    with patch("app.router.billing.PaystackProvider", return_value=mock_provider):
        # User 2 tries to verify user 1's transaction
        resp = client.get(f"/billing/verify?reference={ref}", headers=_auth(token2))
        assert resp.status_code == 403


def test_verify_transaction_failed_status_rejected(client):
    """Unsuccessful transactions return 400."""
    token = _signup_login(client, "fail_tx@test.com")
    uid = _real_user_id(client, token)
    ref = f"T_{uuid.uuid4().hex[:12]}"

    tx_data = {
        "status": "failed",
        "reference": ref,
        "amount": 599900,
        "metadata": {"user_id": uid, "plan_id": "go_monthly"},
    }

    mock_provider = MagicMock()
    mock_provider.verify_transaction = AsyncMock(return_value=tx_data)

    with patch("app.router.billing.PaystackProvider", return_value=mock_provider):
        resp = client.get(f"/billing/verify?reference={ref}", headers=_auth(token))
        assert resp.status_code == 400
