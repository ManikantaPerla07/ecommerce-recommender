# Concise 3-Minute Video Demonstration Script
**ALGIONS AI LABS — E-Commerce Product Recommendation Engine**
**Assessment:** AI & Machine Learning Track

---

### Total Duration: ~3 Minutes (~380 words)

#### 1. Introduction (0:00 – 0:30)
**Screen:** Show VS Code with `README.md` or Title Slide.
> *"Hello, I am **Manikanta Perla**, registration number **[Your Reg Number]**. Today I am presenting my solution for the Algions AI Labs assessment: an **E-Commerce Product Recommendation Engine**.*
> 
> *The goal is to deliver personalized Top-10 product discovery as an advisory decision-support tool, addressing cold-start challenges, catalog coverage, and algorithmic explainability."*

---

#### 2. Data Engineering & Zero-Leakage (0:30 – 1:00)
**Screen:** Show terminal with `python -m src.eda` output or `reports/eda_summary.json`.
> *"Our dataset contains **19,930 chronological interactions** across **1,200 users** and **250 products**, spanning a 90-day window with 96.7% matrix sparsity.*
> 
> *To guarantee defensible evaluation, we enforced a strict **chronological split**: the earliest 70% of events for training, 15% for validation tuning, and the final 15% as a held-out test set. This mathematically prevents future interactions from leaking into model training."*

---

#### 3. Live UI: Recommendations, Cold-Start & Explainability (1:00 – 2:00)
**Screen:** Switch to browser showing Streamlit (`http://localhost:8501`).

> *"Here in our Streamlit interface, all recommendations are computed live from trained `.joblib` model artifacts—there are no hard-coded predictions.*
> 
> *For an active **Warm User**, the Hybrid engine serves tailored items based on their latent taste profile and interaction history.*
> 
> *Now, if we select a **Cold-Start User** with zero training history, pure Collaborative Filtering collapses. However, our Hybrid engine automatically activates a dynamic fallback, shifting weights to Content-Based semantic matching and trending category popularity.*
> 
> *Under Tab 2—**Explainability Inspector**—we can see the exact attribution breakdown: Collaborative signal, Content similarity keywords, Popularity score, and a clear Decision-Support Disclaimer reminding merchandisers that human review is required for high-stakes decisions."*

---

#### 4. Held-Out Benchmark & Results (2:00 – 2:40)
**Screen:** Click Tab 4 ("Offline Metrics & Benchmark").
> *"On the held-out chronological test set at $K=10$:*
> - *The **Popularity Baseline** achieved an NDCG@10 of 0.1149, but only surfaced **8% of the catalog**.*
> - *Our **Hybrid Recommender** achieved an **NDCG@10 of 0.1545**—a **+34.5% lift over baseline**—with **Recall@10 of 0.2223** and **98.0% catalog coverage**, ensuring discovery of long-tail items.*
> - *For cold-start users, the Hybrid model maintained an NDCG@10 of **0.2489**, while pure Collaborative Filtering dropped to **0.0163**."*

---

#### 5. Verification & Conclusion (2:40 – 3:00)
**Screen:** Switch to terminal and run `python -m pytest tests -v`.
> *"Finally, running `pytest tests -v` verifies all **22 automated unit and integration tests** pass, proving temporal leakage prevention, ranking metrics accuracy, and FastAPI endpoint stability.*
> 
> *Thank you for your review. I look forward to the technical walkthrough!"*
