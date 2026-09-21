"""001_set_tier_quotas — Billing Tier Token Quota Migration

Date: 2026-09-10
Idempotent: Yes — overwrites only when the stored value differs.

Writes the canonical per-tier token limits into configs/app_config and
forces free-tier defaults so stale 60k/300k values cannot leak to unpaid users.

  Free (default)     →  6h: 10,000   weekly: 100,000
  Go  (₦5,999/mo)    →  6h: 40,000   weekly: 400,000
  Premium (₦9,999)   →  6h: 60,000   weekly: 600,000

Runtime enforcement uses app/services/quota.py TIER_LIMITS (same numbers).
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from app.database import create_db_client, init_db

TIER_QUOTAS = {
    "free": {
        "token_limit_6h": 10_000,
        "token_limit_weekly": 100_000,
    },
    "go": {
        "token_limit_6h": 40_000,
        "token_limit_weekly": 400_000,
    },
    "premium": {
        "token_limit_6h": 60_000,
        "token_limit_weekly": 600_000,
    },
}

CONFIG_KEY_PREFIX = "tier_quota_"


async def migrate():
    print("  [001_set_tier_quotas] Initializing Firestore connection...")
    init_db()
    db = create_db_client()

    config_ref = db.collection("configs").document("app_config")
    config_doc = await config_ref.get()
    existing = config_doc.to_dict() or {} if config_doc.exists else {}

    updates = {}

    free = TIER_QUOTAS["free"]
    desired_defaults = {
        "default_token_limit_6h": free["token_limit_6h"],
        "default_token_limit_weekly": free["token_limit_weekly"],
    }
    for key, value in desired_defaults.items():
        current = existing.get(key)
        if current != value:
            updates[key] = value
            print(f"  [001_set_tier_quotas]   {key}: {current} → {value:,}")
        else:
            print(f"  [001_set_tier_quotas]   ✓ {key} already {value:,}")

    for tier, limits in TIER_QUOTAS.items():
        for field, value in limits.items():
            key = f"{CONFIG_KEY_PREFIX}{tier}_{field}"
            current = existing.get(key)
            if current != value:
                updates[key] = value
                print(f"  [001_set_tier_quotas]   {key}: {current} → {value:,}")
            else:
                print(f"  [001_set_tier_quotas]   ✓ {key} already {value:,}")

    if updates:
        if config_doc.exists:
            await config_ref.update(updates)
        else:
            await config_ref.set(updates)
        print(f"  [001_set_tier_quotas] ✓ Wrote {len(updates)} quota fields to app_config.")
    else:
        print("  [001_set_tier_quotas] ✓ All tier quota fields already canonical — no changes needed.")


if __name__ == "__main__":
    asyncio.run(migrate())
