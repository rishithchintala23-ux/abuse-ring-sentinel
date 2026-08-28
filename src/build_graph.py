"""
Builds an account graph where edges connect accounts sharing an attribute
(device, card, address, or IP-via-transactions), then extracts per-account
and per-connected-component features that feed the classifier.
 
Key idea: a ring isn't just "shares stuff" - innocent clusters share stuff too.
The separating signal is COMBINING structure (component size, edge density,
how many distinct attribute types are shared) WITH timing (do signups/txns
cluster tightly, or spread out).
"""
 
import pandas as pd
import networkx as nx
from datetime import datetime
import numpy as np
 
 
def build_graph(accounts_df, txns_df):
    G = nx.Graph()
    for _, row in accounts_df.iterrows():
        G.add_node(row["account_id"])
 
    # Connect accounts sharing an attribute
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
 
    # Connect accounts sharing an IP address (via transactions - IP is on
    # the account record here for simplicity, but modeled as a distinct pass
    # to mirror how IP-sharing would typically be joined from login/txn logs)
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
 
 
def extract_features(G, accounts_df, txns_df):
    accounts_df = accounts_df.copy()
    accounts_df["signup_ts"] = pd.to_datetime(accounts_df["signup_date"])
 
    # Component-level stats, computed once per component then broadcast to members
    components = list(nx.connected_components(G))
    comp_id_of = {}
    comp_stats = {}
 
    for idx, comp in enumerate(components):
        for node in comp:
            comp_id_of[node] = idx
 
        comp_size = len(comp)
        subG = G.subgraph(comp)
        n_edges = subG.number_of_edges()
        max_possible_edges = comp_size * (comp_size - 1) / 2
        density = n_edges / max_possible_edges if max_possible_edges > 0 else 0
 
        # distinct attribute types shared across the component
        attr_types = set()
        for _, _, data in subG.edges(data=True):
            attr_types.update(data["shared_attrs"])
        n_attr_types = len(attr_types)
 
        # timing tightness: std dev of signup times within the component (hours)
        comp_signups = accounts_df[accounts_df["account_id"].isin(comp)]["signup_ts"]
        if len(comp_signups) > 1:
            signup_spread_hours = (comp_signups.max() - comp_signups.min()).total_seconds() / 3600
        else:
            signup_spread_hours = 0
 
        # transaction timing tightness within the component
        comp_txns = txns_df[txns_df["account_id"].isin(comp)]
        if len(comp_txns) > 1:
            txn_times = pd.to_datetime(comp_txns["timestamp"])
            txn_spread_hours = (txn_times.max() - txn_times.min()).total_seconds() / 3600
            avg_txn_amount = comp_txns["amount"].mean()
        else:
            txn_spread_hours = 0
            avg_txn_amount = comp_txns["amount"].mean() if len(comp_txns) else 0
 
        comp_stats[idx] = {
            "component_size": comp_size,
            "edge_density": density,
            "n_shared_attr_types": n_attr_types,
            "signup_spread_hours": signup_spread_hours,
            "txn_spread_hours": txn_spread_hours,
            "avg_txn_amount": avg_txn_amount,
        }
 
    # Per-account features
    rows = []
    for _, acct in accounts_df.iterrows():
        acct_id = acct["account_id"]
        comp_idx = comp_id_of.get(acct_id)
 
        if comp_idx is None:
            # isolated node - no shared attributes with anyone
            feat = {
                "account_id": acct_id,
                "degree": 0,
                "component_size": 1,
                "edge_density": 0,
                "n_shared_attr_types": 0,
                "signup_spread_hours": 0,
                "txn_spread_hours": 0,
                "avg_txn_amount": txns_df[txns_df["account_id"] == acct_id]["amount"].mean()
                                  if len(txns_df[txns_df["account_id"] == acct_id]) else 0,
            }
        else:
            feat = {
                "account_id": acct_id,
                "degree": G.degree(acct_id),
                **comp_stats[comp_idx],
            }
        rows.append(feat)
 
    features_df = pd.DataFrame(rows)
    return features_df
 
 
def main():
    accounts_df = pd.read_csv("data/accounts.csv")
    txns_df = pd.read_csv("data/transactions.csv")
 
    print("Building graph...")
    G = build_graph(accounts_df, txns_df)
    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
 
    n_components = nx.number_connected_components(G)
    print(f"Connected components: {n_components}")
 
    print("Extracting features...")
    features_df = extract_features(G, accounts_df, txns_df)
 
    # merge in ground truth for later training/eval
    merged = features_df.merge(
        accounts_df[["account_id", "ring_id", "is_ring_member", "is_hard_negative", "is_evasive_ring"]],
        on="account_id"
    )
    merged.to_csv("data/features.csv", index=False)
    print(f"Saved {len(merged)} rows to data/features.csv")
 
    # quick sanity check: do ring members have higher component_size/density on average?
    print("\nSanity check (mean feature value by class):")
    print(merged.groupby("is_ring_member")[
        ["component_size", "edge_density", "n_shared_attr_types",
         "signup_spread_hours", "txn_spread_hours"]
    ].mean())
 
 
if __name__ == "__main__":
    main()
 