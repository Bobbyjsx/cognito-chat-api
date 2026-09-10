"""Paystack webhook receiver.

All Paystack subscription lifecycle events are handled here:

  charge.success          - First payment succeeds → activate subscription
  subscription.create     - Paystack creates a subscription object → link code + dates
  subscription.enable     - Subscription (re)enabled after being disabled → set ACTIVE
  subscription.disable    - Subscription disabled (cancelled) → set CANCELLED
  invoice.create          - Upcoming renewal invoice created (informational)
  invoice.update          - Invoice updated (informational)
  invoice.payment_failed  - Recurring renewal payment failed → set PAST_DUE
  subscription.expiry_cards - Cards expiring soon (informational)
  customeridentification  - Customer ID updated (informational)

Security requirements enforced:
  - HMAC-SHA512 signature verification using PAYSTACK_SECRET_KEY
  - Idempotency: each event is keyed by (provider_event_id, event_type) in Firestore
  - No user-controlled values are trusted for plan resolution
"""

import hashlib
import hmac
import json
import logging
import uuid
from datetime import timezone

import dateutil.parser
from fastapi import APIRouter, Depends, HTTPException, Request
from google.cloud.firestore_v1.async_client import AsyncClient

from app.api.dependencies import get_db
from app.billing.models import SubscriptionDB, SubscriptionStatus
from app.billing.repository import SubscriptionRepository
from app.core.config import settings

router = APIRouter(prefix="/webhooks/paystack", tags=["Webhooks"])
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

PLAN_CODE_TO_TIER: dict[str, str] = {}  # populated lazily at first request


def _plan_code_map() -> dict[str, str]:
    """Resolve Paystack plan codes → internal tier names from config (no DB hit)."""
    if not PLAN_CODE_TO_TIER:
        if settings.go_paystack_plan_code:
            PLAN_CODE_TO_TIER[settings.go_paystack_plan_code] = "go"
        if settings.premium_paystack_plan_code:
            PLAN_CODE_TO_TIER[settings.premium_paystack_plan_code] = "premium"
    return PLAN_CODE_TO_TIER


def _verify_signature(body: bytes, signature: str) -> bool:
    """Return True if HMAC-SHA512 of body matches the Paystack signature header."""
    secret = settings.paystack_webhook_secret or settings.paystack_secret_key
    if not secret:
        logger.error("No Paystack secret configured – cannot verify webhook signature")
        return False
    expected = hmac.new(secret.encode("utf-8"), body, hashlib.sha512).hexdigest()
    return hmac.compare_digest(expected, signature)


def _parse_dt(value: str | None):
    """Parse an ISO-8601 datetime string, returning None on failure."""
    if not value:
        return None
    try:
        dt = dateutil.parser.isoparse(value)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None


# ──────────────────────────────────────────────────────────────────────────────
# Endpoint
# ──────────────────────────────────────────────────────────────────────────────


@router.post("")
async def paystack_webhook(request: Request, db: AsyncClient = Depends(get_db)):
    body = await request.body()
    signature = request.headers.get("x-paystack-signature", "")

    if not signature:
        logger.warning("Paystack webhook received without x-paystack-signature header")
        raise HTTPException(status_code=400, detail="Missing signature")

    if not _verify_signature(body, signature):
        logger.error("Paystack webhook signature verification FAILED – rejecting request")
        raise HTTPException(status_code=400, detail="Invalid signature")

    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    event: str = payload.get("event", "")
    data: dict = payload.get("data", {})

    repo = SubscriptionRepository(db)

    # ── Idempotency ──────────────────────────────────────────────────────────
    # Prefer a stable provider-supplied identifier; fall back to transaction ref.
    raw_id = data.get("id") or data.get("reference") or data.get("subscription_code")
    event_id = f"paystack_{raw_id}_{event}" if raw_id else f"paystack_noid_{uuid.uuid4().hex}_{event}"

    if raw_id:
        already_processed = not await repo.mark_event_processed(event_id)
        if already_processed:
            logger.info("Idempotent skip – event %s already processed", event_id)
            return {"status": "success"}

    logger.info(
        "Processing Paystack webhook",
        extra={
            "event": event,
            "event_id": event_id,
        },
    )

    try:
        match event:
            case "charge.success":
                await _handle_charge_success(data, repo)

            case "subscription.create":
                await _handle_subscription_create(data, repo)

            case "subscription.enable":
                await _handle_subscription_enable(data, repo)

            case "subscription.disable":
                await _handle_subscription_disable(data, repo)

            case "invoice.payment_failed":
                await _handle_invoice_payment_failed(data, repo)

            case "invoice.create" | "invoice.update" | "subscription.expiry_cards" | "customeridentification":
                # Informational events – acknowledged but no state change required
                logger.info("Acknowledged informational Paystack event: %s", event)

            case _:
                logger.info("Unhandled Paystack event type (ignored): %s", event)

    except Exception:
        # Log but return 200 so Paystack doesn't enter an infinite retry loop for
        # application-level errors. Genuine delivery failures should be investigated
        # via structured logs.
        logger.exception("Error processing Paystack webhook event %s (event_id=%s)", event, event_id)

    return {"status": "success"}


# ──────────────────────────────────────────────────────────────────────────────
# Event handlers
# ──────────────────────────────────────────────────────────────────────────────


async def _handle_charge_success(data: dict, repo: SubscriptionRepository) -> None:
    """First payment succeeded.

    Paystack sends this before subscription.create.  We create/update the
    subscription in ACTIVE state and populate what we already know.  The
    subscription_code arrives later via subscription.create.
    """
    metadata = data.get("metadata") or {}
    user_id: str | None = metadata.get("user_id")
    plan_id: str | None = metadata.get("plan_id")

    if not user_id:
        logger.warning("charge.success: no user_id in metadata – cannot map subscription")
        return

    # Resolve plan from the plan_id stored in metadata (set by us at checkout)
    plan_code: str = data.get("plan", {}).get("plan_code") or ""
    tier = _plan_code_map().get(plan_code)

    # Fallback: resolve via plan_id from metadata
    if not tier and plan_id:
        from app.billing.service import BillingService

        svc = BillingService(repository=repo, provider=None)  # type: ignore[arg-type]
        plans = svc._get_plans()
        matched = next((p for p in plans.values() if p.id == plan_id), None)
        if matched:
            plan_code = plan_code or matched.provider_plan_code
            tier = matched.tier

    if not tier:
        logger.error(
            "charge.success: cannot resolve tier for user=%s plan_id=%s plan_code=%s", user_id, plan_id, plan_code
        )
        return

    customer = data.get("customer") or {}
    from datetime import datetime

    from dateutil.relativedelta import relativedelta

    now = datetime.now(timezone.utc)
    period_start = now
    # paid_at from the charge is more accurate if present
    paid_at_str = data.get("paid_at") or data.get("created_at")
    if paid_at_str:
        parsed = _parse_dt(paid_at_str)
        if parsed:
            period_start = parsed

    period_end = period_start + relativedelta(months=1)

    sub = await repo.get_by_user_id(user_id)
    if sub is None:
        sub = SubscriptionDB(
            id="",
            user_id=user_id,
            product="cognito",
            tier=tier,
            status=SubscriptionStatus.ACTIVE,
            interval="monthly",
            amount=data.get("amount") or 0,
            currency=data.get("currency") or "NGN",
            provider="paystack",
            provider_customer_code=customer.get("customer_code") or "",
            provider_subscription_code="",  # filled in by subscription.create
            provider_plan_code=plan_code,
            current_period_start=period_start,
            current_period_end=period_end,
            cancel_at_period_end=False,
        )
    else:
        # Reactivate / upgrade
        sub.status = SubscriptionStatus.ACTIVE
        sub.tier = tier
        sub.provider_plan_code = plan_code or sub.provider_plan_code
        sub.provider_customer_code = customer.get("customer_code") or sub.provider_customer_code
        sub.current_period_start = period_start
        sub.current_period_end = period_end
        sub.cancel_at_period_end = False
        sub.amount = data.get("amount") or sub.amount

    await repo.save(sub)
    logger.info("charge.success: subscription ACTIVE for user=%s tier=%s", user_id, tier)


async def _handle_subscription_create(data: dict, repo: SubscriptionRepository) -> None:
    """Paystack subscription object created.  Attach the subscription_code and
    accurate next payment date to the subscription record."""
    customer = data.get("customer") or {}
    # Paystack places our metadata on the customer object
    metadata = customer.get("metadata") or {}
    user_id: str | None = metadata.get("user_id")

    # Also check top-level metadata (varies by Paystack version)
    if not user_id:
        metadata = data.get("metadata") or {}
        user_id = metadata.get("user_id")

    sub_code: str | None = data.get("subscription_code")

    if not sub_code:
        logger.warning("subscription.create: no subscription_code in payload")
        return

    # Try to find by user_id first
    sub = None
    if user_id:
        sub = await repo.get_by_user_id(user_id)

    # Fallback: the customer_code may already be in Firestore from charge.success
    if sub is None and customer.get("customer_code"):
        sub = await repo.get_by_provider_customer_code(customer["customer_code"])

    if sub is None:
        logger.warning("subscription.create: cannot find subscription for user=%s", user_id)
        return

    sub.provider_subscription_code = sub_code
    next_payment = _parse_dt(data.get("next_payment_date"))
    if next_payment:
        sub.current_period_end = next_payment
    # Ensure active
    sub.status = SubscriptionStatus.ACTIVE
    sub.cancel_at_period_end = False

    await repo.save(sub)
    logger.info("subscription.create: linked sub_code=%s for user=%s", sub_code, sub.user_id)


async def _handle_subscription_enable(data: dict, repo: SubscriptionRepository) -> None:
    """Subscription re-enabled after being disabled → restore ACTIVE state."""
    sub_code: str | None = data.get("subscription_code")
    if not sub_code:
        return

    sub = await repo.get_by_provider_subscription_code(sub_code)
    if sub is None:
        logger.warning("subscription.enable: no subscription found for sub_code=%s", sub_code)
        return

    sub.status = SubscriptionStatus.ACTIVE
    sub.cancel_at_period_end = False

    next_payment = _parse_dt(data.get("next_payment_date"))
    if next_payment:
        sub.current_period_end = next_payment

    await repo.save(sub)
    logger.info("subscription.enable: subscription ACTIVE sub_code=%s", sub_code)


async def _handle_subscription_disable(data: dict, repo: SubscriptionRepository) -> None:
    """Subscription disabled (user or admin cancelled).  Mark CANCELLED with
    cancel_at_period_end so the user retains access until period ends."""
    sub_code: str | None = data.get("subscription_code")
    if not sub_code:
        return

    sub = await repo.get_by_provider_subscription_code(sub_code)
    if sub is None:
        logger.warning("subscription.disable: no subscription found for sub_code=%s", sub_code)
        return

    sub.status = SubscriptionStatus.CANCELLED
    sub.cancel_at_period_end = True

    await repo.save(sub)
    logger.info("subscription.disable: subscription CANCELLED sub_code=%s", sub_code)


async def _handle_invoice_payment_failed(data: dict, repo: SubscriptionRepository) -> None:
    """Recurring renewal failed.  Mark PAST_DUE; Paystack will retry.  If Paystack
    exhausts retries it will send subscription.disable."""
    # Paystack wraps the subscription inside the invoice object
    sub_code = (data.get("subscription") or {}).get("subscription_code") or data.get("subscription_code")

    if not sub_code:
        logger.warning("invoice.payment_failed: no subscription_code in payload")
        return

    sub = await repo.get_by_provider_subscription_code(sub_code)
    if sub is None:
        logger.warning("invoice.payment_failed: no subscription found for sub_code=%s", sub_code)
        return

    sub.status = SubscriptionStatus.PAST_DUE

    await repo.save(sub)
    logger.info("invoice.payment_failed: subscription PAST_DUE sub_code=%s", sub_code)
