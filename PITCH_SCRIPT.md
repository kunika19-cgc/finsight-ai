# 5-Minute Pitch Video — Script & Shot List

The Buildathon asks for a public repo, a 5-minute pitch video, and the
architecture — this covers the video. Read it once, then talk over the
dashboard in your own words; don't read it verbatim on camera.

Record with the Streamlit app open (locally or the Render demo) and your
face optional (screen-only is fine). Aim for 4:30–5:00.

---

## 0:00–0:30 — The problem (state it, don't sell it)

> "Card-not-present fraud costs merchants money two ways: missed fraud,
> and genuine customers wrongly blocked. Most demo fraud detectors only
> report accuracy, which hides both of those costs. I built FinSight AI
> for Razorpay's AI Risk Manager track to show both, honestly, on a
> held-out test set — not just a good-looking number."

**Show:** the Overview tab.

## 0:30–1:30 — What it does & the metrics (this is the core of the pitch)

> "It's a Random Forest classifier trained on transaction amount,
> category, gender, hour, and location. On 9,502 held-out transactions
> it never saw during training: 93.4% precision, 94.7% recall, F1 of
> 94.1%. Here's the confusion matrix — 100 false positives, 79 false
> negatives, out of 9,502."

**Show:** the Performance tab — metrics table + confusion matrix.

> "I don't stop at accuracy. I estimate the cost of getting it wrong in
> each direction: blocking a genuine transaction costs friction and
> support overhead; missing real fraud costs the full transaction
> amount. Those are illustrative assumptions, not measured business
> figures — I say that explicitly in the dashboard."

**Show:** the false-positive / false-negative cost breakdown.

> "And I don't trust one aggregate number. I re-check precision and
> recall on two harder slices — high-amount transactions and night-hour
> transactions — because a model can look great on average and still
> fail on the cases that matter most."

**Show:** the slice-check table.

## 1:30–2:30 — Live prediction + explainability

> "Here's a live transaction. The model outputs a fraud probability —
> but I also run SHAP on every single prediction, not just a global
> feature-importance chart, so you see exactly which features pushed
> *this* transaction toward fraud or genuine."

**Show:** enter a transaction, show the SHAP waterfall/bar for that one
prediction.

> "There's an optional LLM layer — it takes those SHAP values and
> narrates them in plain English for an analyst. It's a communication
> layer only: it explains the SHAP output, it never influences the
> model's actual decision."

**Show:** click "Explain in plain English," read 1 line of the output.

## 2:30–3:15 — Batch scoring + audit trail

> "For a finance team, one transaction at a time isn't realistic. Batch
> scoring takes a CSV, scores every row, and flags rows with a category
> or gender the model wasn't trained on instead of silently
> mishandling them."

**Show:** upload the sample batch, show the risk-distribution chart.

> "Every prediction — live or batch — is logged to an audit trail with
> a timestamp, the inputs, the probability, and the decision, so every
> flag is traceable after the fact."

**Show:** the audit trail (or just say it's logged — no need to open the
raw CSV on camera).

## 3:15–4:00 — Honesty about limits (judges specifically reward this)

> "Three things I want to be upfront about. One: this dataset's fraud
> rate is 15.8%, far higher than real card fraud, which is under 1%
> — it's balanced for a clean pipeline demo, not a real-world fraud
> rate claim. Two: I don't have transaction-velocity features — 'N
> transactions in the last hour' — which is one of the strongest real
> fraud signals, and this dataset doesn't have it. Three: this is
> strictly a detector. It scores and flags for review. It never
> autonomously blocks, reverses, or retaliates against a transaction —
> that's a deliberate scope boundary, not a missing feature."

**Show:** the "Known limitations" section of the README, or just talk —
no need to screen-record text.

## 4:00–4:45 — Why this fits the Risk Manager track + what's next

> "This maps directly to the AI Risk Manager track: a working detector,
> honest metrics including false-positive cost, and it's strictly
> defense-only. If I had more time, the next step is wiring in real
> velocity features and testing on a naturally imbalanced sample
> instead of a balanced one, to see how precision holds up when fraud
> is rare."

## 4:45–5:00 — Close

> "Full code, the metrics.json the dashboard reads from, and the
> architecture diagram are in the repo. Thanks for watching."

---

## Recording checklist

- [ ] Screen recording at 1080p minimum (OBS Studio or QuickTime are free)
- [ ] Test your mic levels before the full take — re-record if there's
      background noise
- [ ] Have the sample batch CSV ready to upload *before* you hit record
      so you're not searching for a file on camera
- [ ] Do one full dry run first — SHAP plots and the LLM call both take
      a couple of seconds to load; know where those pauses are so you
      can talk through them instead of going silent
- [ ] Trim dead air at the start/end in editing (even a free tool like
      the OS-native video trimmer is enough — no need for anything fancy)
- [ ] Upload unlisted to YouTube or use Loom; put the link in your
      Buildathon application, not just the README
