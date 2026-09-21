# Paystack Setup Instructions

This document outlines the manual steps required to fully configure the Paystack billing integration for Cognito.

## Environments
You will need to perform these steps for both your **Test** environment and **Live/Production** environment. Do not mix test and live plan codes.

## Checklist

### 1. Create Paystack Plans
You must create two recurring plans in the Paystack Dashboard.
1. Go to Paystack Dashboard -> Plans.
2. Click **New Plan**.

#### Cognito Go
- **Name:** Cognito Go
- **Amount:** ₦5,999
- **Interval:** Monthly
- **Currency:** NGN

#### Cognito Premium
- **Name:** Cognito Premium
- **Amount:** ₦9,999
- **Interval:** Monthly
- **Currency:** NGN

### 2. Copy Plan Codes
After creating each plan, copy its exact Plan Code (it looks like `PLN_xxxxx`).
- [ ] Copy Go plan code (`GO_PAYSTACK_PLAN_CODE`) - PLN_sr1h7zo9qogclvb
- [ ] Copy Premium plan code (`PREMIUM_PAYSTACK_PLAN_CODE`) - PLN_3v65wje2l7kin47

### 3. Configure Environment Variables
Add the following variables to your environment configuration (e.g. `.env` or Secret Manager):
- [ ] `PAYSTACK_SECRET_KEY`: Your Paystack Secret Key (Test or Live). - sk_test_dc8973dd7d21fd6658833238637ebd7bb48f4bdb
- [ ] `PAYSTACK_PUBLIC_KEY`: Your Paystack Public Key. - pk_test_0e0c6e73367841737812f357a6012baaf5421313
- [ ] `PAYSTACK_WEBHOOK_SECRET`: If configured, or use the Secret Key if not explicitly set.
- [ ] `GO_PAYSTACK_PLAN_CODE`: The code for the Go plan.
- [ ] `PREMIUM_PAYSTACK_PLAN_CODE`: The code for the Premium plan.

### 4. Configure Webhooks
Paystack needs to know where to send billing events (successful charges, failed renewals, etc).
1. Go to Paystack Dashboard -> Settings -> API Keys & Webhooks.
2. Under **Webhook URL**, enter your backend's webhook endpoint:
   `https://<your-api-domain>/webhooks/paystack`
3. [ ] Configure webhook endpoint in Paystack.

### 5. Test the Integration
In test mode, use the Paystack test cards to verify:
- [ ] Test successful checkout and subscription activation.
- [ ] Test recurring payment webhook triggers.
- [ ] Test subscription cancellation.

### 6. Production Credentials
When going live, repeat the process with your Live Paystack credentials and create Live plans.
- [ ] Configure production credentials.
