"""
Abuse-Ring Sentinel - Streamlit Demo

A risk analyst's view: browse flagged clusters, see the network graph of
shared attributes, and read the evidence/explanation for each flag.

Run with: streamlit run src/app.py
"""

import streamlit as st
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import json

st.set_page_config(page_title="Abuse-Ring Sentinel", layout="wide")


@st.cache_data
def load_data():
    accounts_df = pd.read_csv("data/accounts.csv")
    predictions_df = pd.read_csv("data/test_predictions.csv")
    with open("data/audit_log.json") as f:
        audit_log = json.load(f)
    with open("data/model_results.json") as f:
        model_results = json.load(f)
    return accounts_df, predictions_df, audit_log, model_results


def build_graph(accounts_df, account_ids):
    """Build a small graph just for the accounts we want to visualize."""
    G = nx.Graph()
    subset = accounts_df[accounts_df["account_id"].isin(account_ids)]
    for _, row in subset.iterrows():
        G.add_node(row["account_id"])

    for attr, label in [("device_id", "device"), ("card_fingerprint", "card"),
                         ("address_id", "address"), ("ip_address", "ip")]:
        groups = subset.dropna(subset=[attr]).groupby(attr)["account_id"].apply(list)
        for accts in groups:
            if len(accts) > 1:
                for i in range(len(accts)):
                    for j in range(i + 1, len(accts)):
                        G.add_edge(accts[i], accts[j], label=label)
    return G


def draw_graph(G, flagged_ids):
    fig, ax = plt.subplots(figsize=(6, 5))
    pos = nx.spring_layout(G, seed=42, k=0.8)
    colors = ["#e74c3c" if n in flagged_ids else "#3498db" for n in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=400, ax=ax)
    nx.draw_networkx_edges(G, pos, alpha=0.5, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=6, ax=ax,
                             labels={n: n.replace("acct_", "") for n in G.nodes()})
    ax.axis("off")
    return fig


def main():
    accounts_df, predictions_df, audit_log, model_results = load_data()

    st.title("🔗 Abuse-Ring Sentinel")
    st.caption("AI Risk Manager — detecting coordinated multi-account abuse via shared-attribute graphs")

    # --- Top metrics ---
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Precision", f"{model_results['precision']:.1%}")
    col2.metric("Recall", f"{model_results['recall']:.1%}")
    col3.metric("F1 Score", f"{model_results['f1']:.1%}")
    cm = model_results["confusion_matrix"]
    col4.metric("False Positives", cm["fp"], help="On held-out, entirely unseen rings")

    with st.expander("📊 Full metrics + cost estimate"):
        st.json(model_results)

    st.divider()

    # --- Flagged accounts browser ---
    st.subheader("Flagged Accounts (held-out test set)")

    flagged_ids = [rec["account_id"] for rec in audit_log]
    sorted_log = sorted(audit_log, key=lambda r: -r["ring_probability"])

    selected_idx = st.selectbox(
        "Select a flagged account to inspect:",
        range(len(sorted_log)),
        format_func=lambda i: f"{sorted_log[i]['account_id']} "
                               f"(confidence: {sorted_log[i]['ring_probability']:.1%}, "
                               f"cluster size: {sorted_log[i]['cluster_size']})"
    )
    record = sorted_log[selected_idx]

    left, right = st.columns([1, 1])

    with left:
        st.markdown("### Evidence")
        st.markdown(f"**Confidence:** {record['ring_probability']:.1%}")
        st.markdown(f"**Cluster size:** {record['cluster_size']}")
        st.markdown(f"**Shared attribute types:** {', '.join(record['shared_attribute_types']) or 'none'}")
        st.markdown(f"**Signup spread:** {record['signup_spread_hours']:.1f} hours")
        st.markdown(f"**Transaction spread:** {record['txn_spread_hours']:.1f} hours")

        st.markdown("### Explanation")
        st.info(record["explanation"])

        st.markdown("### Action taken")
        st.warning(f"**{record['action_taken']}** — {record['note']}")

        with st.expander("Raw audit record (JSON)"):
            st.json(record)

    with right:
        st.markdown("### Cluster network graph")
        cluster_ids = [record["account_id"]] + [
            ev["connected_account"] for ev in record["connected_accounts_evidence"]
        ]
        if len(cluster_ids) > 1:
            G = build_graph(accounts_df, cluster_ids)
            fig = draw_graph(G, flagged_ids=[record["account_id"]])
            st.pyplot(fig)
            st.caption("Red = selected flagged account. Blue = connected accounts. "
                       "Edges = shared device/card/address/IP.")
        else:
            st.write("No connected accounts to visualize.")

    st.divider()
    st.caption(
        "Defense-only by design: this system flags clusters for human review. "
        "It never blocks, freezes, or takes automated action on any account."
    )


if __name__ == "__main__":
    main()
