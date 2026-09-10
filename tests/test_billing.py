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
import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

# ──────────────────────────────────────────────────────────────────────────────
# Fixtures & helpers
# ──────────────────────────────────────────────────────────────────────────────

TEST_SECRET = "sk_test_dc8973dd7d21fd6658833238637ebd7bb48f4bdb"
GO_PLAN_CODE = "PLN_sr1h7zo9qogclvb"
PREMIUM_PLAN_CODE = "PLN_3v65wje2l7kin47"
USER_ID = "test-user-billing-001"
USER_EMAIL = "billing@test.com"


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


def test_checkout_go_calls_paystack(client):
    """Checkout initializes with Paystack and returns authorization_url."""
    token = _signup_login(client, "checkout_go@test.com")

    mock_provider = MagicMock()
    mock_provider.initialize_subscription_checkout = AsyncMock(
        return_value={"authorization_url": "https://checkout.paystack.com/abc", "reference": "ref_abc"}
    )

    with patch("app.router.billing.PaystackProvider", return_value=mock_provider):
        resp = client.post("/billing/checkout", json={"plan": "go"}, headers=_auth(token))

    assert resp.status_code == 200
    assert resp.json()["authorization_url"] == "https://checkout.paystack.com/abc"
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
