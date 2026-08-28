"""
Trains a classifier to flag ring membership using graph + transaction features.
 
Critical design choice: split train/test by CONNECTED COMPONENT, not by row.
If we split by row randomly, accounts from the same ring could appear in both
train and test, letting the model "cheat" by recognizing a component it already
saw part of. Splitting by component simulates the real deployment scenario:
the model has never seen this ring before.
"""
 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score, recall_score, f1_score, confusion_matrix,
    classification_report, roc_auc_score
)
import xgboost as xgb
import json
 
FEATURE_COLS = [
    "degree", "component_size", "edge_density", "n_shared_attr_types",
    "signup_spread_hours", "txn_spread_hours",
]
# avg_txn_amount deliberately excluded: it's not a genuine structural signal
# of ring membership, and including it risks the model latching onto a
# spurious shortcut instead of the graph/timing signal this project is
# actually about.
 
# Business assumptions for false-positive / false-negative cost, made explicit
# and easy to change - this is a stated assumption, not a hidden number.
COST_PER_FALSE_POSITIVE_INVESTIGATION = 150   # analyst time to review a wrongly flagged cluster
AVG_LOSS_PER_MISSED_RING = 8000                # estimated abuse payout per undetected ring
 
 
def component_level_split(features_df, test_size=0.3, random_state=42):
    """
    Group accounts by their component (proxy: same component_size + edge_density +
    n_shared_attr_types + signup_spread signature isn't reliable; instead we
    recover component membership via ring_id for ring members, and treat every
    isolated / non-ring account as its own singleton component for splitting.
    """
    # For ring members, split whole rings together (never split a ring across sets)
    ring_ids = features_df.loc[features_df["is_ring_member"] == 1, "ring_id"].dropna().unique()
    ring_ids = np.array(list(ring_ids))  # force plain numpy array - avoids
    # pyarrow-backed indexing errors from train_test_split on newer pandas
    train_rings, test_rings = train_test_split(
        ring_ids, test_size=test_size, random_state=random_state
    )
 
    ring_rows = features_df[features_df["is_ring_member"] == 1]
    train_ring_rows = ring_rows[ring_rows["ring_id"].isin(train_rings)]
    test_ring_rows = ring_rows[ring_rows["ring_id"].isin(test_rings)]
 
    # Non-ring accounts (independent + innocent clusters) split randomly by row
    # since they aren't part of a labeled group we need to keep intact for THIS
    # label (innocent clusters are already grouped by shared attribute, but
    # since none of them are positives, row-level split is fine for the negative class)
    non_ring_rows = features_df[features_df["is_ring_member"] == 0].reset_index(drop=True)
    train_non_ring, test_non_ring = train_test_split(
        non_ring_rows, test_size=test_size, random_state=random_state
    )
 
    train_df = pd.concat([train_ring_rows, train_non_ring]).sample(frac=1, random_state=random_state)
    test_df = pd.concat([test_ring_rows, test_non_ring]).sample(frac=1, random_state=random_state)
 
    return train_df, test_df
 
 
def train_and_evaluate():
    df = pd.read_csv("data/features.csv")
 
    train_df, test_df = component_level_split(df)
    print(f"Train: {len(train_df)} rows ({train_df['is_ring_member'].sum()} ring members)")
    print(f"Test:  {len(test_df)} rows ({test_df['is_ring_member'].sum()} ring members)")
    print(f"Test rings are entirely unseen during training (split by ring_id, not row).\n")
 
    X_train, y_train = train_df[FEATURE_COLS], train_df["is_ring_member"]
    X_test, y_test = test_df[FEATURE_COLS], test_df["is_ring_member"]
 
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        scale_pos_weight=(y_train == 0).sum() / max((y_train == 1).sum(), 1),
        random_state=42,
        eval_metric="logloss",
    )
    model.fit(X_train, y_train)
 
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
 
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
 
    fp_cost = fp * COST_PER_FALSE_POSITIVE_INVESTIGATION
    fn_cost = fn * AVG_LOSS_PER_MISSED_RING  # note: per-account, conservative proxy for per-ring loss
    total_cost = fp_cost + fn_cost
 
    print("=" * 60)
    print("HELD-OUT TEST SET METRICS (unseen rings)")
    print("=" * 60)
    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    print(f"F1:        {f1:.3f}")
    print(f"ROC-AUC:   {auc:.3f}")
    print(f"\nConfusion matrix:")
    print(f"  True Negatives:  {tn}")
    print(f"  False Positives: {fp}  (innocent accounts wrongly flagged)")
    print(f"  False Negatives: {fn}  (ring members missed)")
    print(f"  True Positives:  {tp}")
    print(f"\nEstimated cost:")
    print(f"  False positive investigation cost: ${fp_cost:,} ({fp} x ${COST_PER_FALSE_POSITIVE_INVESTIGATION})")
    print(f"  False negative missed-fraud cost:  ${fn_cost:,} ({fn} x ${AVG_LOSS_PER_MISSED_RING})")
    print(f"  Total estimated cost:              ${total_cost:,}")
    print(f"\n(Cost assumptions are illustrative and stated explicitly above -")
    print(f" adjust COST_PER_FALSE_POSITIVE_INVESTIGATION / AVG_LOSS_PER_MISSED_RING")
    print(f" to match real merchant figures.)")
 
    # Feature importance - which signals actually drive the decision
    importance = dict(zip(FEATURE_COLS, model.feature_importances_.round(4).tolist()))
    print(f"\nFeature importance:")
    for feat, imp in sorted(importance.items(), key=lambda x: -x[1]):
        print(f"  {feat}: {imp}")
 
    # Critical credibility check: how does the model do specifically on the
    # HARD negatives (innocent clusters with tight timing, deliberately
    # designed to resemble rings)? A 0-FP claim only means something if it
    # holds here too, not just on easy negatives.
    if "is_hard_negative" in test_df.columns:
        hard_mask = test_df["is_hard_negative"] == True
        n_hard = hard_mask.sum()
        if n_hard > 0:
            hard_fp = ((y_pred == 1) & hard_mask.values & (y_test.values == 0)).sum()
            print(f"\nHard-negative stress test:")
            print(f"  {n_hard} hard innocent accounts in test set (tight-timing clusters)")
            print(f"  False positives among them: {hard_fp} ({hard_fp/n_hard*100:.1f}%)")
 
    # Same for evasive rings - our deliberately hard POSITIVE case
    if "is_evasive_ring" in test_df.columns:
        evasive_mask = (test_df["is_evasive_ring"] == True).values
        n_evasive = evasive_mask.sum()
        if n_evasive > 0:
            evasive_caught = ((y_pred == 1) & evasive_mask & (y_test.values == 1)).sum()
            evasive_missed = ((y_pred == 0) & evasive_mask & (y_test.values == 1)).sum()
            print(f"\nEvasive-ring stress test (rings that spread transactions to evade timing detection):")
            print(f"  {n_evasive} evasive ring members in test set")
            print(f"  Caught: {evasive_caught}  Missed: {evasive_missed}  "
                  f"Recall on evasive rings: {evasive_caught/n_evasive*100:.1f}%")
            print(f"  (This is our honest documented weak spot: the model leans heavily on")
            print(f"   txn_spread_hours; a ring sophisticated enough to spread its transactions")
            print(f"   like a normal account can partially evade detection.)")
 
    # Save everything needed for the explainability layer + report
    results = {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "roc_auc": round(auc, 4),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "cost_estimate": {
            "fp_cost": int(fp_cost), "fn_cost": int(fn_cost), "total_cost": int(total_cost),
            "assumptions": {
                "cost_per_fp_investigation": COST_PER_FALSE_POSITIVE_INVESTIGATION,
                "avg_loss_per_missed_ring_member": AVG_LOSS_PER_MISSED_RING,
            }
        },
        "feature_importance": importance,
        "n_train": len(train_df),
        "n_test": len(test_df),
        "n_test_ring_members": int(test_df["is_ring_member"].sum()),
    }
    with open("data/model_results.json", "w") as f:
        json.dump(results, f, indent=2)
 
    # Save test predictions for the explainability/audit layer to consume next
    test_df = test_df.copy()
    test_df["predicted_ring_member"] = y_pred
    test_df["ring_probability"] = y_proba
    test_df.to_csv("data/test_predictions.csv", index=False)
 
    model.save_model("data/model.json")
    print(f"\nSaved: data/model_results.json, data/test_predictions.csv, data/model.json")
 
    return model, results
 
 
if __name__ == "__main__":
    train_and_evaluate()
 