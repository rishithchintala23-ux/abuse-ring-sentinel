"""
Explainability + audit trail layer.

For every account the model flags, produces:
1. A structured audit record (JSON) - which attributes were shared, with whom,
   graph stats, confidence, model version, timestamp of decision.
2. A plain-English explanation a non-technical risk analyst can read.

Design principle: this layer NEVER takes action on an account (no blocking,
no freezing, no auto-rejection). It only produces a recommendation + evidence
for a human reviewer. This is what "strictly defense-only" and "explainable,
bounded, gated" mean in practice - the system stops at "here's what I found
and why," never "and here's what I did about it."
"""

import pandas as pd
import networkx as nx
import json
from datetime import datetime, timezone

MODEL_VERSION = "abuse-ring-sentinel-v1"


def build_graph_for_lookup(accounts_df):
    """Rebuild the graph so we can look up exact shared attributes per flagged account."""
    G = nx.Graph()
    for _, row in accounts_df.iterrows():
        G.add_node(row["account_id"])

    for attr in ["device_id", "card_fingerprint", "address_id"]:
        groups = accounts_df.dropna(subset=[attr]).groupby(attr)["account_id"].apply(list)
        for accts in groups:
            if len(accts) > 1:
                for i in range(len(accts)):
                    for j in range(i + 1, len(accts)):
                        if G.has_edge(accts[i], accts[j]):
                            G[accts[i]][accts[j]]["shared_attrs"].append(attr)
                        else:
                            G.add_edge(accts[i], accts[j], shared_attrs=[attr])

    ip_groups = accounts_df.dropna(subset=["ip_address"]).groupby("ip_address")["account_id"].apply(list)
    for accts in ip_groups:
        if len(accts) > 1:
            for i in range(len(accts)):
                for j in range(i + 1, len(accts)):
                    if G.has_edge(accts[i], accts[j]):
                        G[accts[i]][accts[j]]["shared_attrs"].append("ip_address")
                    else:
                        G.add_edge(accts[i], accts[j], shared_attrs=["ip_address"])
    return G


ATTR_LABELS = {
    "device_id": "device",
    "card_fingerprint": "payment card",
    "address_id": "delivery address",
    "ip_address": "IP address",
}


def explain_account(account_id, G, accounts_df, probability, feature_row):
    """Builds one structured audit record + plain-English explanation for a flagged account."""
    neighbors = list(G.neighbors(account_id)) if account_id in G else []

    # collect what's actually shared with whom, for evidence
    shared_evidence = []
    for nbr in neighbors:
        edge_data = G.get_edge_data(account_id, nbr)
        attrs = edge_data["shared_attrs"]
        shared_evidence.append({
            "connected_account": nbr,
            "shared_attributes": [ATTR_LABELS.get(a, a) for a in attrs],
        })

    attr_types_involved = sorted(set(
        label for ev in shared_evidence for label in ev["shared_attributes"]
    ))

    component_size = int(feature_row["component_size"])
    signup_spread_hours = round(float(feature_row["signup_spread_hours"]), 1)
    txn_spread_hours = round(float(feature_row["txn_spread_hours"]), 1)

    # Plain-English explanation - the part a human reviewer actually reads
    if len(neighbors) == 0:
        explanation = "No shared attributes found with other accounts."
    else:
        attr_str = ", ".join(attr_types_involved)
        explanation = (
            f"This account is connected to {len(neighbors)} other account(s) "
            f"(cluster of {component_size} total) via shared {attr_str}. "
        )
        if signup_spread_hours < 72:
            explanation += (
                f"All accounts in this cluster signed up within "
                f"{signup_spread_hours:.1f} hours of each other. "
            )
        if txn_spread_hours < 200:
            explanation += (
                f"Transactions across the cluster occurred within a "
                f"{txn_spread_hours:.1f}-hour window, suggesting coordinated activity "
                f"rather than independent, organic usage."
            )
        else:
            explanation += (
                f"Transactions were spread over {txn_spread_hours:.1f} hours - "
                f"less immediately suspicious on timing alone, but the shared-attribute "
                f"structure and cluster size still warrant review."
            )

    audit_record = {
        "account_id": account_id,
        "model_version": MODEL_VERSION,
        "decision_timestamp": datetime.now(timezone.utc).isoformat(),
        "action_taken": "FLAGGED_FOR_REVIEW",  # never anything stronger - defense-only
        "ring_probability": round(float(probability), 4),
        "cluster_size": component_size,
        "n_connected_accounts": len(neighbors),
        "shared_attribute_types": attr_types_involved,
        "signup_spread_hours": signup_spread_hours,
        "txn_spread_hours": txn_spread_hours,
        "connected_accounts_evidence": shared_evidence,
        "explanation": explanation,
        "note": "This is a recommendation for human review only. No automated "
                "account action (blocking, freezing, rejection) has been taken.",
    }
    return audit_record


def main():
    accounts_df = pd.read_csv("data/accounts.csv")
    predictions_df = pd.read_csv("data/test_predictions.csv")
    features_df = pd.read_csv("data/features.csv")

    print("Rebuilding graph for evidence lookup...")
    G = build_graph_for_lookup(accounts_df)

    flagged = predictions_df[predictions_df["predicted_ring_member"] == 1]
    print(f"Generating audit records for {len(flagged)} flagged accounts...")

    audit_log = []
    for _, row in flagged.iterrows():
        acct_id = row["account_id"]
        feature_row = features_df[features_df["account_id"] == acct_id].iloc[0]
        record = explain_account(
            acct_id, G, accounts_df, row["ring_probability"], feature_row
        )
        audit_log.append(record)

    with open("data/audit_log.json", "w") as f:
        json.dump(audit_log, f, indent=2)

    print(f"Saved {len(audit_log)} audit records to data/audit_log.json")

    # print one example for a quick look
    if audit_log:
        print("\n--- Example audit record ---")
        print(json.dumps(audit_log[0], indent=2))


if __name__ == "__main__":
    main()
