"""
FinSight AI - Model Training Script
------------------------------------
Trains the fraud detection model with a proper train/test split,
computes precision/recall/F1/confusion matrix on a HELD-OUT test set,
and saves everything needed for the dashboard to display honest metrics.

Run this once (or whenever the dataset changes):
    python train_model.py
"""

import json
import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, classification_report
)

RANDOM_STATE = 42

# ---------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------
df = pd.read_csv("data/transactions_sample.csv")
df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])
df["hour"] = df["trans_date_trans_time"].dt.hour

# ---------------------------------------------------------------------
# 2. Encode categoricals
# ---------------------------------------------------------------------
le_category = LabelEncoder()
le_gender = LabelEncoder()
df["category_encoded"] = le_category.fit_transform(df["category"])
df["gender_encoded"] = le_gender.fit_transform(df["gender"])

FEATURES = [
    "amt", "category_encoded", "gender_encoded", "hour",
    "city_pop", "lat", "long", "merch_lat", "merch_long"
]
X = df[FEATURES]
y = df["is_fraud"]

# ---------------------------------------------------------------------
# 3. Proper train/test split (stratified, so fraud ratio is preserved
#    in both halves) — the test set is NEVER shown to the model during
#    training, so metrics on it are honest, not memorized.
# ---------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

# ---------------------------------------------------------------------
# 4. Train
# ---------------------------------------------------------------------
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    class_weight="balanced",
    random_state=RANDOM_STATE,
    n_jobs=-1,
)
model.fit(X_train, y_train)

# ---------------------------------------------------------------------
# 5. Evaluate on the HELD-OUT test set only
# ---------------------------------------------------------------------
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)

# ---------------------------------------------------------------------
# 6. False-positive cost framing (buildathon explicitly asks for this)
#    - A false positive = genuine transaction wrongly blocked/flagged
#      -> customer friction, potential lost sale.
#    - A false negative = fraud that slipped through -> direct monetary loss.
#    We report both counts + a simple cost estimate so the trade-off is
#    visible, not hidden behind a single accuracy number.
# ---------------------------------------------------------------------
avg_genuine_amt = df.loc[df.is_fraud == 0, "amt"].mean()
avg_fraud_amt = df.loc[df.is_fraud == 1, "amt"].mean()

estimated_fp_friction_cost = fp * avg_genuine_amt * 0.02  # assume 2% of a flagged genuine txn's value is lost to friction/support cost - clearly document this assumption
estimated_fn_fraud_loss = fn * avg_fraud_amt  # full value lost per missed fraud

# ---------------------------------------------------------------------
# 6b. Slice checks — a lightweight stand-in for drift/stress testing.
#     Instead of trusting one aggregate number, we re-check precision/
#     recall on two harder slices of the test set: high-amount
#     transactions, and night-hour transactions. If the model's
#     performance falls apart on a slice, that's a real signal a
#     single aggregate F1 score would hide.
# ---------------------------------------------------------------------
def slice_metrics(mask, y_test_full, y_pred_full):
    mask = np.asarray(mask)
    n = int(mask.sum())
    if n == 0:
        return {"n": 0, "precision": None, "recall": None, "f1": None}
    yt = y_test_full[mask]
    yp = y_pred_full[mask]
    return {
        "n": n,
        "precision": round(float(precision_score(yt, yp, zero_division=0)), 4),
        "recall": round(float(recall_score(yt, yp, zero_division=0)), 4),
        "f1": round(float(f1_score(yt, yp, zero_division=0)), 4),
    }

y_test_arr = y_test.values
high_amt_threshold = X_test["amt"].quantile(0.75)
high_amt_mask = (X_test["amt"] > high_amt_threshold).values
night_mask = X_test["hour"].isin([0, 1, 2, 3, 4, 5]).values

slice_checks = {
    "high_amount_txns": {
        "description": f"Transactions above the 75th percentile amount (> ${high_amt_threshold:.2f})",
        **slice_metrics(high_amt_mask, y_test_arr, y_pred),
    },
    "night_hour_txns": {
        "description": "Transactions between 12am-6am",
        **slice_metrics(night_mask, y_test_arr, y_pred),
    },
}

metrics = {
    "test_set_size": int(len(y_test)),
    "train_set_size": int(len(y_train)),
    "precision": round(float(precision), 4),
    "recall": round(float(recall), 4),
    "f1_score": round(float(f1), 4),
    "roc_auc": round(float(auc), 4),
    "confusion_matrix": {
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    },
    "false_positive_cost_estimate_usd": round(float(estimated_fp_friction_cost), 2),
    "false_negative_cost_estimate_usd": round(float(estimated_fn_fraud_loss), 2),
    "cost_assumptions": (
        "FP cost assumes 2% of a wrongly-flagged genuine transaction's value "
        "is lost to customer friction / support overhead. FN cost assumes the "
        "full transaction amount is lost when fraud is missed. These are "
        "illustrative assumptions, not measured business figures."
    ),
    "feature_importances": dict(
        sorted(
            zip(FEATURES, model.feature_importances_.round(4).tolist()),
            key=lambda x: -x[1],
        )
    ),
    "slice_checks": slice_checks,
}

print(json.dumps(metrics, indent=2))
print("\nFull classification report:\n", classification_report(y_test, y_pred, target_names=["Genuine", "Fraud"]))

# ---------------------------------------------------------------------
# 7. Save model, encoders, and metrics
# ---------------------------------------------------------------------
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/fraud_model.pkl")
joblib.dump(le_category, "models/le_category.pkl")
joblib.dump(le_gender, "models/le_gender.pkl")

with open("models/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

# ---------------------------------------------------------------------
# 8. Save the raw test-set predictions too, so the dashboard can let
#    the user drag a threshold slider and see precision/recall/cost
#    recompute LIVE, without needing to retrain or reload big objects.
# ---------------------------------------------------------------------
test_predictions = pd.DataFrame({
    "y_true": y_test.values,
    "y_proba": y_proba,
    "amt": X_test["amt"].values,
})
test_predictions.to_csv("models/test_predictions.csv", index=False)

print("\nSaved to models/: fraud_model.pkl, le_category.pkl, le_gender.pkl, metrics.json, test_predictions.csv")
