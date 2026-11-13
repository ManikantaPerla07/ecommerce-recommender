# 3-Minute Video Demonstration Script (Modernized UI Walkthrough)
**ALGIONS AI LABS — E-Commerce Product Recommendation Engine**
**Assessment:** AI & Machine Learning Track | Advanced Individual Project

---

### Total Duration: ~3 Minutes (~380 words)

#### 1. Introduction & Mission (0:00 – 0:30)
**Screen:** Show VS Code with `README.md` or Title Slide.
> *"Hello, I am **Manikanta Perla**, registration number **[Your Registration Number]**. Today I am presenting my solution for the Algions AI Labs assessment: an **Enterprise E-Commerce Product Recommendation Engine**.*
> 
> *Our mission was to build a top-N personalized discovery engine that operates as a transparent **decision-support tool** for shoppers and merchandisers, handling cold-start scenarios, catalog coverage, and feature explainability."*

---

#### 2. Data Engineering & Zero-Leakage Chronological Split (0:30 – 0:55)
**Screen:** Show terminal with `reports/eda_summary.json` or `src.eda` output.
> *"Our dataset contains **19,930 chronological interactions** across **1,200 users** and **250 catalog products** with a realistic 96.7% matrix sparsity and full purchase funnel progression.*
> 
> *To guarantee defensible evaluation, we implemented a strict **chronological split**: the earliest 70% of interactions for training, 15% for validation tuning, and the final 15% as a held-out test set. Our automated test suite mathematically verifies zero future interaction leakage."*

---

#### 3. Live UI: Discovery Feed, Cold-Start & Explainability (0:55 – 2:05)
**Screen:** Switch to browser showing the Streamlit App (`http://localhost:8501`).

> *"Here in our enterprise dashboard, all recommendations are computed dynamically from our trained `.joblib` model artifacts.*
> 
> *In **Tab 1 — Discovery Feed**, for an active **Power Shopper**, the Hybrid engine serves tailored recommendations with match percentage badges, pricing, and category tags.*
> 
> *Now, notice what happens when we switch to a **Brand-New Cold-Start User**: pure Collaborative Filtering collapses. However, our Hybrid engine automatically activates a cold-start fallback badge, seamlessly balancing Content-Based semantic matching with trending category popularity.*
> 
> *In **Tab 2 — Explainability Inspector**, we can inspect any individual product to see the exact multi-signal score breakdown: Collaborative Filtering latent score, Content similarity keyword chips like 'wireless' and 'noise-cancelling', and the Popularity prior—along with a decision-support governance notice for merchandisers."*

---

#### 4. Benchmarking & Empirical Results (2:05 – 2:40)
**Screen:** Click **Tab 4 — Offline Evaluation Metrics**.
> *"In **Tab 4**, we examine our held-out test set evaluation at $K=10$:*
> - *The **Popularity Baseline** achieved an NDCG@10 of 0.1149, but only surfaced **8.0% of the catalog**, heavily penalizing new inventory.*
> - *Our **Hybrid Recommender** achieved an **NDCG@10 of 0.1545**—a **+34.5% lift over baseline**—with **Recall@10 of 0.2223** and **98.0% catalog coverage**, ensuring long-tail discovery.*
> - *In the cold-start test slice, the Hybrid model maintained an NDCG@10 of **0.2489**, while pure Collaborative Filtering dropped to **0.0163**."*

---

#### 5. Verification & Conclusion (2:40 – 3:00)
**Screen:** Switch to terminal and run `python -m pytest tests -v`.
> *"Finally, running `pytest tests -v` verifies all **22 automated unit and integration tests** pass in ~5 seconds, confirming data integrity, ranking metrics accuracy, and FastAPI endpoint stability.*
> 
> *Thank you for your review, and I look forward to the technical walkthrough!"*
