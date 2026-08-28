"""
Synthetic data generator for Abuse-Ring Sentinel.

Generates:
- accounts.csv: account_id, signup_date, device_id, ip_address, card_fingerprint, address_id
- transactions.csv: txn_id, account_id, timestamp, amount, merchant_id
- ground_truth.csv: account_id, ring_id (null if not part of a ring), is_ring_member

Design principles (why this isn't a toy):
1. Rings share SOME attributes, not ALL - real colluders vary their footprint
   (e.g. same device, different declared address) to look less obvious.
2. Innocent noise clusters exist: families/roommates/offices that legitimately
   share a device or IP but are NOT rings. This is what makes false-positive
   cost meaningful instead of trivially zero.
3. Ring size varies (3-8 accounts) and sharing pattern varies (device-heavy ring
   vs card-heavy ring vs address-heavy ring) so the model can't learn one
   trivial rule.
4. Timing matters: rings tend to act in bursts (created/transact close together);
   innocent shared-device clusters are spread over time.
"""

import random
import uuid
from datetime import datetime, timedelta
import pandas as pd
from faker import Faker

fake = Faker()
Faker.seed(42)
random.seed(42)

N_INDEPENDENT_ACCOUNTS = 3000
N_RINGS = 40                # planted collusion rings
RING_SIZE_RANGE = (3, 8)
N_INNOCENT_CLUSTERS = 60    # innocent shared-attribute groups (the hard negatives)
INNOCENT_CLUSTER_SIZE_RANGE = (2, 5)

MERCHANT_ID = "merchant_demo_001"
SIM_START = datetime(2025, 1, 1)
SIM_END = datetime(2026, 8, 1)


def random_timestamp(start=SIM_START, end=SIM_END):
    delta = end - start
    return start + timedelta(seconds=random.randint(0, int(delta.total_seconds())))


def make_account(account_id, device_id=None, ip=None, card=None, address_id=None,
                  signup_time=None):
    return {
        "account_id": account_id,
        "signup_date": (signup_time or random_timestamp()).isoformat(),
        "device_id": device_id or f"dev_{uuid.uuid4().hex[:10]}",
        "ip_address": ip or fake.ipv4(),
        "card_fingerprint": card or f"card_{uuid.uuid4().hex[:12]}",
        "address_id": address_id or f"addr_{uuid.uuid4().hex[:10]}",
    }


def generate_independent_accounts(n):
    return [make_account(f"acct_{uuid.uuid4().hex[:10]}") for _ in range(n)]


def generate_ring(ring_id, size):
    """
    A collusion ring: accounts created in a short burst, sharing 1-2 attributes
    (not all - realistic colluders vary declared info) with tight timing.
    Pattern varies per ring so no single rule solves it.
    """
    pattern = random.choice(["device_heavy", "card_heavy", "address_heavy", "mixed"])
    burst_start = random_timestamp()
    burst_window = timedelta(hours=random.randint(1, 72))  # created close together

    shared_device = f"dev_{uuid.uuid4().hex[:10]}"
    shared_card = f"card_{uuid.uuid4().hex[:12]}"
    shared_address = f"addr_{uuid.uuid4().hex[:10]}"

    accounts = []
    for i in range(size):
        signup_time = burst_start + timedelta(
            seconds=random.randint(0, int(burst_window.total_seconds()))
        )
        acct_id = f"acct_{uuid.uuid4().hex[:10]}"

        if pattern == "device_heavy":
            device = shared_device
            card = shared_card if random.random() < 0.3 else None
            address = None
        elif pattern == "card_heavy":
            card = shared_card
            device = shared_device if random.random() < 0.3 else None
            address = None
        elif pattern == "address_heavy":
            address = shared_address
            device = shared_device if random.random() < 0.2 else None
            card = None
        else:  # mixed - shares different combos, hardest to detect
            device = shared_device if random.random() < 0.5 else None
            card = shared_card if random.random() < 0.5 else None
            address = shared_address if random.random() < 0.4 else None

        acct = make_account(acct_id, device_id=device, card=card,
                             address_id=address, signup_time=signup_time)
        acct["ring_id"] = ring_id
        acct["is_ring_member"] = 1
        accounts.append(acct)

    return accounts


def generate_innocent_cluster(cluster_id, size):
    """
    Legit shared attribute (e.g. household sharing wifi/device, or office
    sharing an IP) but accounts are NOT colluding: signups spread over time,
    no coordinated transaction bursts, and they share exactly ONE attribute
    (a ring usually correlates on timing AND attributes together).
    """
    shared_attr_type = random.choice(["device", "ip", "address"])
    shared_value = None
    if shared_attr_type == "device":
        shared_value = f"dev_{uuid.uuid4().hex[:10]}"
    elif shared_attr_type == "address":
        shared_value = f"addr_{uuid.uuid4().hex[:10]}"

    accounts = []
    for i in range(size):
        # spread signups over months, NOT a tight burst
        signup_time = random_timestamp()
        kwargs = {}
        if shared_attr_type == "device":
            kwargs["device_id"] = shared_value
        elif shared_attr_type == "address":
            kwargs["address_id"] = shared_value

        acct_id = f"acct_{uuid.uuid4().hex[:10]}"
        acct = make_account(acct_id, signup_time=signup_time, **kwargs)
        acct["ring_id"] = None
        acct["is_ring_member"] = 0
        accounts.append(acct)

        if shared_attr_type == "ip":
            # shared IP handled via transactions, not account field - skip here
            pass

    return accounts


def generate_transactions(accounts, ring_lookup):
    """
    Ring members transact in coordinated bursts around their signup time.
    Independent/innocent accounts transact at normal, spread-out intervals.
    """
    txns = []
    for acct in accounts:
        signup = datetime.fromisoformat(acct["signup_date"])
        is_ring = acct.get("is_ring_member", 0) == 1
        n_txns = random.randint(1, 5) if not is_ring else random.randint(1, 3)

        for _ in range(n_txns):
            if is_ring:
                # tight burst near signup - typical abuse pattern (promo grab, etc.)
                offset = timedelta(minutes=random.randint(0, 180))
            else:
                offset = timedelta(days=random.randint(0, 400))

            ts = signup + offset
            amount = round(random.uniform(50, 500) if is_ring
                            else random.uniform(50, 5000), 2)
            txns.append({
                "txn_id": f"txn_{uuid.uuid4().hex[:12]}",
                "account_id": acct["account_id"],
                "timestamp": ts.isoformat(),
                "amount": amount,
                "merchant_id": MERCHANT_ID,
            })
    return txns


def main():
    print("Generating independent accounts...")
    independent = generate_independent_accounts(N_INDEPENDENT_ACCOUNTS)
    for a in independent:
        a["ring_id"] = None
        a["is_ring_member"] = 0

    print(f"Generating {N_RINGS} collusion rings...")
    ring_accounts = []
    for i in range(N_RINGS):
        size = random.randint(*RING_SIZE_RANGE)
        ring_accounts.extend(generate_ring(f"ring_{i:03d}", size))

    print(f"Generating {N_INNOCENT_CLUSTERS} innocent shared-attribute clusters...")
    innocent_accounts = []
    for i in range(N_INNOCENT_CLUSTERS):
        size = random.randint(*INNOCENT_CLUSTER_SIZE_RANGE)
        innocent_accounts.extend(generate_innocent_cluster(f"cluster_{i:03d}", size))

    all_accounts = independent + ring_accounts + innocent_accounts
    random.shuffle(all_accounts)

    ring_lookup = {a["account_id"]: a.get("ring_id") for a in all_accounts}

    print("Generating transactions...")
    transactions = generate_transactions(all_accounts, ring_lookup)

    accounts_df = pd.DataFrame(all_accounts)
    txns_df = pd.DataFrame(transactions)

    accounts_df.to_csv("data/accounts.csv", index=False)
    txns_df.to_csv("data/transactions.csv", index=False)

    print(f"\nDone.")
    print(f"Total accounts: {len(accounts_df)}")
    print(f"  Independent: {len(independent)}")
    print(f"  Ring members: {len(ring_accounts)} across {N_RINGS} rings")
    print(f"  Innocent cluster members: {len(innocent_accounts)} across {N_INNOCENT_CLUSTERS} clusters")
    print(f"Total transactions: {len(txns_df)}")
    print(f"\nSaved to data/accounts.csv and data/transactions.csv")


if __name__ == "__main__":
    main()
