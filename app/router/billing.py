from fastapi import APIRouter, Depends
from google.cloud.firestore_v1.async_client import AsyncClient

from app.api.dependencies import get_current_user, get_db
from app.billing.models import SubscriptionStatus
from app.billing.providers.paystack import PaystackProvider
from app.billing.repository import SubscriptionRepository
from app.billing.schemas import CheckoutRequest, CheckoutResponse, PlansResponse, SubscriptionSchema
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


@router.post("/checkout", response_model=CheckoutResponse)
async def create_checkout(
    request: CheckoutRequest,
    current_user: UserDB = Depends(get_current_user),
    service: BillingService = Depends(get_billing_service),
):
    return await service.create_checkout(request.plan, str(current_user.id), current_user.email)


@router.post("/subscription/cancel")
async def cancel_subscription(
    current_user: UserDB = Depends(get_current_user), service: BillingService = Depends(get_billing_service)
):
    success = await service.cancel_subscription(str(current_user.id), current_user.email)
    return {"success": success}
