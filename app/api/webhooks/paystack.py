import hashlib
import hmac
import json
import logging
from datetime import datetime, timezone

import dateutil.parser
from fastapi import APIRouter, Depends, HTTPException, Request
from google.cloud.firestore_v1.async_client import AsyncClient

from app.api.dependencies import get_db
from app.billing.models import SubscriptionDB, SubscriptionStatus
from app.billing.repository import SubscriptionRepository
from app.core.config import settings

router = APIRouter(prefix="/webhooks/paystack", tags=["Webhooks"])
logger = logging.getLogger(__name__)


@router.post("")
async def paystack_webhook(request: Request, db: AsyncClient = Depends(get_db)):
    body = await request.body()
    signature = request.headers.get("x-paystack-signature")

    if not signature:
        raise HTTPException(status_code=400, detail="Missing signature")

    secret = settings.paystack_webhook_secret or settings.paystack_secret_key
    expected_signature = hmac.new(secret.encode("utf-8"), body, hashlib.sha512).hexdigest()

    if signature != expected_signature:
        logger.error("Invalid paystack webhook signature")
        raise HTTPException(status_code=400, detail="Invalid signature")

    payload = json.loads(body)
    event = payload.get("event")
    data = payload.get("data", {})

    repo = SubscriptionRepository(db)

    # Idempotency
    # Sometimes data doesn't have an ID, we could use the payload's own ID or reference if available
    event_id_val = data.get("id") or data.get("reference")
    if event_id_val:
        event_id = f"paystack_{event_id_val}_{event}"
        if not await repo.mark_event_processed(event_id):
            logger.info(f"Event {event_id} already processed")
            return {"status": "success"}

    logger.info(f"Processing paystack webhook event: {event}")

    try:
        if event == "charge.success":
            metadata = data.get("metadata", {})
            user_id = metadata.get("user_id")
            plan_id = metadata.get("plan_id")

            if user_id and plan_id:
                # We can create a pending subscription or update an existing one
                from app.billing.service import PLANS

                plan = PLANS.get(plan_id.replace("_monthly", ""))  # handle key if needed
                if not plan:
                    plan = next((p for p in PLANS.values() if p.id == plan_id), None)

                if plan:
                    customer = data.get("customer", {})
                    # For charge.success, there is no subscription_code yet until subscription.create
                    sub = await repo.get_by_user_id(user_id)
                    if not sub:
                        sub = SubscriptionDB(
                            user_id=user_id,
                            tier=plan.tier,
                            status=SubscriptionStatus.ACTIVE,
                            interval=plan.interval,
                            amount=plan.amount,
                            currency=plan.currency,
                            provider_customer_code=customer.get("customer_code", ""),
                            provider_subscription_code="",
                            provider_plan_code=plan.provider_plan_code,
                        )
                    else:
                        sub.status = SubscriptionStatus.ACTIVE
                        sub.tier = plan.tier
                        sub.amount = plan.amount

                    # Estimate period end (1 month)
                    from dateutil.relativedelta import relativedelta

                    sub.current_period_start = datetime.now(timezone.utc)
                    sub.current_period_end = sub.current_period_start + relativedelta(months=1)
                    await repo.save(sub)

        elif event == "subscription.create":
            # Link the subscription_code to the user
            customer = data.get("customer", {})
            metadata = customer.get("metadata", {})
            user_id = metadata.get("user_id")

            sub_code = data.get("subscription_code")

            # Find the user's subscription or create
            if user_id:
                sub = await repo.get_by_user_id(user_id)
            else:
                # Fallback, we don't have user_id, can't map
                sub = None

            if sub and sub_code:
                sub.provider_subscription_code = sub_code
                if data.get("next_payment_date"):
                    sub.current_period_end = dateutil.parser.isoparse(data["next_payment_date"])
                await repo.save(sub)

        elif event == "subscription.disable":
            sub_code = data.get("subscription_code")
            if sub_code:
                sub = await repo.get_by_provider_subscription_code(sub_code)
                if sub:
                    sub.status = SubscriptionStatus.CANCELLED
                    sub.cancel_at_period_end = True
                    await repo.save(sub)

        elif event == "invoice.payment_failed":
            sub_code = data.get("subscription", {}).get("subscription_code")
            if not sub_code:
                sub_code = data.get("subscription_code")

            if sub_code:
                sub = await repo.get_by_provider_subscription_code(sub_code)
                if sub:
                    sub.status = SubscriptionStatus.PAST_DUE
                    await repo.save(sub)

    except Exception as e:
        logger.error(f"Error processing webhook: {e}")
        # Return success so Paystack doesn't keep retrying if it's a code error,
        # or we could return 500. Let's return 200 but log error.

    return {"status": "success"}
