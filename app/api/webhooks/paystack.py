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


def _plan_code_map() -> dict[str, str]:
    """Resolve Paystack plan codes → internal tier names from config (no DB hit)."""
    mapping: dict[str, str] = {}
    if settings.go_paystack_plan_code:
        mapping[settings.go_paystack_plan_code.strip()] = "go"
    if settings.premium_paystack_plan_code:
        mapping[settings.premium_paystack_plan_code.strip()] = "premium"
    return mapping


def _catalog_plans():
    from app.billing.service import BillingService

    return BillingService(repository=None, provider=None)._get_plans()  # type: ignore[arg-type]


def _extract_plan_code(data: dict) -> str:
    plan = data.get("plan")
    if isinstance(plan, dict):
        return str(plan.get("plan_code") or plan.get("code") or "").strip()
    if isinstance(plan, str):
        return plan.strip()
    return ""


def _extract_metadata(data: dict) -> dict:
    meta: dict = {}
    raw = data.get("metadata")
    if isinstance(raw, dict):
        meta.update(raw)
    customer = data.get("customer") or {}
    if isinstance(customer, dict):
        customer_meta = customer.get("metadata")
        if isinstance(customer_meta, dict):
            meta.update(customer_meta)
    return meta


def _resolve_tier(data: dict, metadata: dict | None = None) -> tuple[str | None, str]:
    """Resolve (tier, plan_code). Prefer amount paid, then our checkout metadata, then Paystack plan code."""
    metadata = metadata if metadata is not None else _extract_metadata(data)
    plan_code = _extract_plan_code(data)
    plans = _catalog_plans()

    amount = data.get("amount")
    if isinstance(amount, int):
        by_amount = next((p for p in plans.values() if p.amount == amount), None)
        if by_amount:
            return by_amount.tier, plan_code or by_amount.provider_plan_code

    plan_id = metadata.get("plan_id")
    if plan_id:
        matched = next((p for p in plans.values() if p.id == plan_id), None)
        if matched:
            return matched.tier, plan_code or matched.provider_plan_code

    mapped = _plan_code_map().get(plan_code)
    if mapped:
        return mapped, plan_code
    return None, plan_code


async def _find_existing_sub(
    repo: SubscriptionRepository,
    user_id: str | None = None,
    customer_code: str | None = None,
    sub_code: str | None = None,
):
    if sub_code:
        found = await repo.get_by_provider_subscription_code(sub_code)
        if found:
            return found
    if user_id:
        found = await repo.get_by_user_id(user_id)
        if found:
            return found
    if customer_code:
        return await repo.get_by_provider_customer_code(customer_code)
    return None


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

    Paystack may send this before or after subscription.create.  We upsert
    ACTIVE state from the amount actually paid, not a mismatched plan_code.
    """
    metadata = _extract_metadata(data)
    user_id: str | None = metadata.get("user_id")
    customer = data.get("customer") or {}
    customer_code = customer.get("customer_code") or ""
    incoming_sub_code = data.get("subscription_code") or (data.get("subscription") or {}).get("subscription_code") or ""

    if not user_id:
        logger.warning("charge.success: no user_id in metadata – cannot map subscription")
        return

    tier, plan_code = _resolve_tier(data, metadata)
    if not tier:
        logger.error(
            "charge.success: cannot resolve tier for user=%s plan_id=%s plan_code=%s amount=%s",
            user_id,
            metadata.get("plan_id"),
            plan_code,
            data.get("amount"),
        )
        return

    from datetime import datetime

    from dateutil.relativedelta import relativedelta

    now = datetime.now(timezone.utc)
    period_start = now
    paid_at_str = data.get("paid_at") or data.get("created_at")
    if paid_at_str:
        parsed = _parse_dt(paid_at_str)
        if parsed:
            period_start = parsed

    period_end = period_start + relativedelta(months=1)

    sub = await _find_existing_sub(repo, user_id=user_id, customer_code=customer_code, sub_code=incoming_sub_code)

    # Handle stopping the old subscription during an upgrade/plan change
    if sub is not None:
        old_sub_code = sub.provider_subscription_code
        if old_sub_code and (not incoming_sub_code or old_sub_code != incoming_sub_code) and sub.tier != tier:
            logger.info(
                "charge.success: plan change detected (%s -> %s). Stopping renewal of old sub %s",
                sub.tier,
                tier,
                old_sub_code,
            )
            try:
                from app.billing.providers.paystack import PaystackProvider

                provider = PaystackProvider()
                old_paystack_sub = await provider.get_subscription(old_sub_code)
                if old_paystack_sub:
                    email_token = old_paystack_sub.get("email_token")
                    if email_token:
                        await provider.cancel_subscription(old_sub_code, email_token)
                        logger.info("Successfully disabled old subscription %s on Paystack", old_sub_code)
            except Exception as e:
                logger.error("Failed to cancel old subscription %s during upgrade: %s", old_sub_code, e)

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
            provider_customer_code=customer_code,
            provider_subscription_code=incoming_sub_code,
            provider_plan_code=plan_code,
            current_period_start=period_start,
            current_period_end=period_end,
            cancel_at_period_end=False,
        )
    else:
        sub.user_id = user_id or sub.user_id
        sub.status = SubscriptionStatus.ACTIVE
        sub.tier = tier
        sub.provider_plan_code = plan_code or sub.provider_plan_code
        sub.provider_customer_code = customer_code or sub.provider_customer_code
        if incoming_sub_code:
            sub.provider_subscription_code = incoming_sub_code
        sub.current_period_start = period_start
        sub.current_period_end = period_end
        sub.cancel_at_period_end = False
        sub.amount = data.get("amount") or sub.amount

        # If this charge matches a pending scheduled change, clear it.
        if sub.scheduled_change and sub.scheduled_change.status == "pending":
            if sub.scheduled_change.target_tier == tier:
                logger.info("charge.success: completed scheduled change to %s", tier)
                sub.scheduled_change.status = "completed"
            elif tier:
                # If they somehow got charged for a different tier, also complete/clear it
                sub.scheduled_change.status = "cancelled"

    await repo.save(sub)
    logger.info("charge.success: subscription ACTIVE for user=%s tier=%s", user_id, tier)


async def _handle_subscription_create(data: dict, repo: SubscriptionRepository) -> None:
    """Paystack subscription object created.  Attach the subscription_code and
    accurate next payment date.  May arrive before charge.success."""
    customer = data.get("customer") or {}
    metadata = _extract_metadata(data)
    user_id: str | None = metadata.get("user_id")
    customer_code = customer.get("customer_code") or ""
    sub_code: str | None = data.get("subscription_code")

    if not sub_code:
        logger.warning("subscription.create: no subscription_code in payload")
        return

    sub = await _find_existing_sub(repo, user_id=user_id, customer_code=customer_code, sub_code=sub_code)
    tier, plan_code = _resolve_tier(data, metadata)

    if sub is None:
        if not user_id and not customer_code:
            logger.warning("subscription.create: cannot find subscription for user=%s", user_id)
            return
        from datetime import datetime

        from dateutil.relativedelta import relativedelta

        now = datetime.now(timezone.utc)
        sub = SubscriptionDB(
            id="",
            user_id=user_id or "",
            product="cognito",
            tier=tier or "go",
            status=SubscriptionStatus.ACTIVE,
            interval="monthly",
            amount=data.get("amount") or 0,
            currency=data.get("currency") or "NGN",
            provider="paystack",
            provider_customer_code=customer_code,
            provider_subscription_code=sub_code,
            provider_plan_code=plan_code,
            current_period_start=now,
            current_period_end=now + relativedelta(months=1),
            cancel_at_period_end=False,
        )
    else:
        if user_id:
            sub.user_id = user_id

        # If this is a future-dated subscription (e.g., from a scheduled downgrade)
        # we do not want to immediately overwrite the current active tier.
        import datetime as dt

        now = dt.datetime.now(timezone.utc)

        start_ts = data.get("start")
        is_future = False
        if isinstance(start_ts, (int, float)):
            start_dt = dt.datetime.fromtimestamp(start_ts, tz=timezone.utc)
            if start_dt > now:
                is_future = True

        is_scheduled = sub.scheduled_change and sub.scheduled_change.status == "pending"

        if is_future or is_scheduled:
            logger.info(
                "subscription.create: detected future/scheduled subscription %s. Preserving current tier %s",
                sub_code,
                sub.tier,
            )
            if sub.scheduled_change:
                sub.scheduled_change.provider_subscription_code = sub_code
        else:
            sub.provider_subscription_code = sub_code
            sub.provider_customer_code = customer_code or sub.provider_customer_code
            if tier:
                sub.tier = tier
            if plan_code:
                sub.provider_plan_code = plan_code

            next_payment = _parse_dt(data.get("next_payment_date"))
            if next_payment:
                sub.current_period_end = next_payment
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
