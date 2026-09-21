from fastapi import APIRouter, Depends
from google.cloud.firestore_v1.async_client import AsyncClient

from app.api.dependencies import get_current_user, get_db, get_persisted_user
from app.billing.models import SubscriptionStatus
from app.billing.providers.paystack import PaystackProvider
from app.billing.repository import SubscriptionRepository
from app.billing.schemas import (
    CheckoutRequest,
    CheckoutResponse,
    DowngradeRequest,
    DowngradeResponse,
    PlansResponse,
    SubscriptionSchema,
)
from app.billing.service import BillingService
from app.models.users import UserDB

router = APIRouter(prefix="/billing", tags=["Billing"])


def get_billing_service(db: AsyncClient = Depends(get_db)) -> BillingService:
    repo = SubscriptionRepository(db)
    provider = PaystackProvider()
    return BillingService(repo, provider)


@router.get("/plans", response_model=PlansResponse)
async def get_plans(service: BillingService = Depends(get_billing_service)):
    return PlansResponse(plans=service.get_plans())


@router.get("", response_model=SubscriptionSchema)
async def get_billing_status(
    current_user: UserDB = Depends(get_current_user), service: BillingService = Depends(get_billing_service)
):
    sub = await service.get_user_subscription(str(current_user.id))
    if not sub:
        return SubscriptionSchema(
            tier="free",
            status=SubscriptionStatus.EXPIRED,
            interval="monthly",
            amount=0,
            currency="NGN",
            current_period_end=None,
            cancel_at_period_end=False,
        )
    return sub


@router.get("/verify", response_model=SubscriptionSchema)
async def verify_transaction(
    reference: str,
    current_user: UserDB = Depends(get_current_user),
    service: BillingService = Depends(get_billing_service),
):
    return await service.verify_and_activate_payment(reference, str(current_user.id))


@router.post("/checkout", response_model=CheckoutResponse)
async def create_checkout(
    request: CheckoutRequest,
    current_user: UserDB = Depends(get_persisted_user),
    service: BillingService = Depends(get_billing_service),
):
    return await service.create_checkout(
        request.plan, str(current_user.id), current_user.email, callback_url=request.callback_url
    )


@router.post("/subscription/cancel")
async def cancel_subscription(
    current_user: UserDB = Depends(get_current_user), service: BillingService = Depends(get_billing_service)
):
    success = await service.cancel_subscription(str(current_user.id), current_user.email)
    return {"success": success}


@router.post("/subscription/downgrade", response_model=DowngradeResponse)
async def downgrade_subscription(
    request: DowngradeRequest,
    current_user: UserDB = Depends(get_current_user),
    service: BillingService = Depends(get_billing_service),
):
    from fastapi import HTTPException

    try:
        return await service.schedule_downgrade(str(current_user.id), request.plan)
    except HTTPException:
        raise
    except Exception:
        import logging

        logging.getLogger(__name__).exception("Failed to schedule downgrade")
        raise HTTPException(status_code=500, detail="Internal server error scheduling downgrade")


@router.post("/subscription/downgrade/cancel")
async def cancel_downgrade(
    current_user: UserDB = Depends(get_current_user), service: BillingService = Depends(get_billing_service)
):
    from fastapi import HTTPException

    try:
        success = await service.cancel_downgrade(str(current_user.id))
        return {"success": success}
    except HTTPException:
        raise
    except Exception:
        import logging

        logging.getLogger(__name__).exception("Failed to cancel scheduled downgrade")
        raise HTTPException(status_code=500, detail="Internal server error cancelling downgrade")
