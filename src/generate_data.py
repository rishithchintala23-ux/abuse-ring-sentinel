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
N_RINGS = 40                # planted collusion rings (obvious timing pattern)
N_EVASIVE_RINGS = 15         # rings that spread transactions to evade timing detection
RING_SIZE_RANGE = (3, 8)
N_INNOCENT_CLUSTERS = 40    # easy innocent clusters (loose timing)
N_HARD_INNOCENT_CLUSTERS = 30  # hard innocent clusters (tight timing, like rings)
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


def generate_independent_accounts(n, shared_ip_pool_fraction=0.15, n_shared_ips=25):
    """
    Most accounts get a unique IP. But a fraction of independent, totally
    unrelated accounts are assigned from a small pool of "shared" IPs -
    simulating real-world IP reuse from mobile carrier NAT, public WiFi,
    or corporate networks. These accounts have NOTHING else in common
    (different device, card, address, signup time) - they only coincide
    on IP, purely by the nature of how networks work in reality.

    This is what makes IP-based false positives a genuine, tested risk
    in our evaluation instead of an unmodeled gap.
    """
    shared_ip_pool = [fake.ipv4() for _ in range(n_shared_ips)]
    accounts = []
    for _ in range(n):
        if random.random() < shared_ip_pool_fraction:
            ip = random.choice(shared_ip_pool)
        else:
            ip = None  # will get a unique IP from make_account's default
        accounts.append(make_account(f"acct_{uuid.uuid4().hex[:10]}", ip=ip))
    return accounts


def generate_ring(ring_id, size, evasive=False):
    """
    A collusion ring: accounts created in a short burst, sharing 1-2 attributes
    (not all - realistic colluders vary declared info) with tight timing.
    Pattern varies per ring so no single rule solves it.

    evasive=True: a more sophisticated ring that spreads its transactions
    out over days/weeks instead of bursting near signup, deliberately
    evading the txn_spread_hours signal. This is our deliberate hard case.
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
        acct["is_hard_negative"] = False
        acct["is_evasive_ring"] = evasive
        accounts.append(acct)

    return accounts


def generate_innocent_cluster(cluster_id, size, hard_mode=False):
    """
    Legit shared attribute (e.g. household sharing wifi/device, or office
    sharing an IP) but accounts are NOT colluding.

    hard_mode=False: signups spread over months (easy negative).
    hard_mode=True: signups clustered in a SHORT window - e.g. a family
      signing up together during a promo weekend, or an office onboarding
      several employees the same day. This is the genuinely hard case:
      tight timing like a ring, but no coordinated abuse behavior
      (transactions still spread out normally, not bursted near signup).
      This is what makes a 0-false-positive claim credible instead of
      an artifact of an easy dataset.
    """
    shared_attr_type = random.choice(["device", "ip", "address"])
    shared_value = None
    if shared_attr_type == "device":
        shared_value = f"dev_{uuid.uuid4().hex[:10]}"
    elif shared_attr_type == "address":
        shared_value = f"addr_{uuid.uuid4().hex[:10]}"

    if hard_mode:
        burst_start = random_timestamp()
        burst_window = timedelta(hours=random.randint(1, 72))  # same as rings

    accounts = []
    for i in range(size):
        if hard_mode:
            signup_time = burst_start + timedelta(
                seconds=random.randint(0, int(burst_window.total_seconds()))
            )
        else:
            signup_time = random_timestamp()  # spread over months

        kwargs = {}
        if shared_attr_type == "device":
            kwargs["device_id"] = shared_value
        elif shared_attr_type == "address":
            kwargs["address_id"] = shared_value

        acct_id = f"acct_{uuid.uuid4().hex[:10]}"
        acct = make_account(acct_id, signup_time=signup_time, **kwargs)
        acct["ring_id"] = None
        acct["is_ring_member"] = 0
        acct["is_hard_negative"] = hard_mode
        acct["is_evasive_ring"] = False
        accounts.append(acct)

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
        is_evasive = acct.get("is_evasive_ring", False)
        n_txns = random.randint(1, 5) if not is_ring else random.randint(1, 3)

        for _ in range(n_txns):
            if is_ring and not is_evasive:
                # tight burst near signup - typical abuse pattern (promo grab, etc.)
                offset = timedelta(minutes=random.randint(0, 180))
            elif is_ring and is_evasive:
                # evasive ring deliberately spreads transactions like a
                # normal account would, to avoid the timing signal
                offset = timedelta(days=random.randint(1, 60))
            else:
                offset = timedelta(days=random.randint(0, 400))

            ts = signup + offset
            # Amount drawn from the SAME distribution regardless of ring
            # membership - transaction amount must not leak the label.
            # The detector should rely on graph structure and timing, not
            # a spending-amount shortcut.
            amount = round(random.uniform(50, 5000), 2)
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
        a["is_hard_negative"] = False
        a["is_evasive_ring"] = False

    print(f"Generating {N_RINGS} collusion rings...")
    ring_accounts = []
    for i in range(N_RINGS):
        size = random.randint(*RING_SIZE_RANGE)
        ring_accounts.extend(generate_ring(f"ring_{i:03d}", size))

    print(f"Generating {N_EVASIVE_RINGS} EVASIVE collusion rings (spread transactions, hard case)...")
    evasive_ring_accounts = []
    for i in range(N_EVASIVE_RINGS):
        size = random.randint(*RING_SIZE_RANGE)
        evasive_ring_accounts.extend(generate_ring(f"evasive_ring_{i:03d}", size, evasive=True))

    print(f"Generating {N_INNOCENT_CLUSTERS} easy innocent clusters (loose timing)...")
    innocent_accounts = []
    for i in range(N_INNOCENT_CLUSTERS):
        size = random.randint(*INNOCENT_CLUSTER_SIZE_RANGE)
        innocent_accounts.extend(generate_innocent_cluster(f"cluster_{i:03d}", size, hard_mode=False))

    print(f"Generating {N_HARD_INNOCENT_CLUSTERS} HARD innocent clusters (tight timing, like rings)...")
    hard_innocent_accounts = []
    for i in range(N_HARD_INNOCENT_CLUSTERS):
        size = random.randint(*INNOCENT_CLUSTER_SIZE_RANGE)
        hard_innocent_accounts.extend(
            generate_innocent_cluster(f"hard_cluster_{i:03d}", size, hard_mode=True)
        )

    all_accounts = (independent + ring_accounts + evasive_ring_accounts
                     + innocent_accounts + hard_innocent_accounts)
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
    print(f"  Ring members (obvious): {len(ring_accounts)} across {N_RINGS} rings")
    print(f"  Ring members (evasive): {len(evasive_ring_accounts)} across {N_EVASIVE_RINGS} rings")
    print(f"  Easy innocent cluster members: {len(innocent_accounts)} across {N_INNOCENT_CLUSTERS} clusters")
    print(f"  HARD innocent cluster members: {len(hard_innocent_accounts)} across {N_HARD_INNOCENT_CLUSTERS} clusters")
    print(f"Total transactions: {len(txns_df)}")
    print(f"\nSaved to data/accounts.csv and data/transactions.csv")


if __name__ == "__main__":
    main()
