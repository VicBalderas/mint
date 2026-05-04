# MINT — Saved Models

This folder contains trained model and scaler pairs saved as `.pkl` files.
Each model has a paired scaler — both must be loaded together for predictions.

---

## Active Models

| File                    | Status   | Notes                                |
|-------------------------|----------|--------------------------------------|
| `QQQ_logreg_model.pkl`  | ✅ ACTIVE | QQQ production model                 |
| `QQQ_logreg_scaler.pkl` | ✅ ACTIVE | Paired scaler — load alongside model |

---

## Retired Models (kept for reference)

| File                          | Status     | Notes                              |
|-------------------------------|------------|------------------------------------|
| `QQQ_randomforest_model.pkl`  | 🔴 RETIRED | Overfits — 78% train vs 50% val    |
| `QQQ_randomforest_scaler.pkl` | 🔴 RETIRED | Paired scaler for retired RF model |

---

## Loading a Model

```python
import joblib

model  = joblib.load("models/QQQ_logreg_model.pkl")
scaler = joblib.load("models/QQQ_logreg_scaler.pkl")

# Scale features before predicting — always use the paired scaler
X_scaled = scaler.transform(X)
predictions = model.predict(X_scaled)
probabilities = model.predict_proba(X_scaled)
```

---

## Naming Convention

```
<TICKER>_<model>_model.pkl
<TICKER>_<model>_scaler.pkl
```

Examples: `QQQ_logreg_model.pkl`, `NVDA_randomforest_model.pkl`