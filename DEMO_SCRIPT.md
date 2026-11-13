# 3–5 MINUTE DEMONSTRATION VIDEO SCRIPT
**ALGIONS AI LABS — E-Commerce Product Recommendation Engine**
**Assessment:** AI & Machine Learning Track | Advanced Individual Project

---

## Video Recording Overview & Checklist
- **Target Duration:** 3 to 5 minutes (ideal: ~4 minutes).
- **Setup Before Recording:**
  1. Terminal open with virtual environment activated.
  2. Streamlit UI running on `http://localhost:8501`.
  3. FastAPI docs running on `http://127.0.0.1:8000/docs`.
  4. Screen recorder configured to record audio + desktop.

---

## Timed Walkthrough Script

### Scene 1: Introduction & Problem Framing (0:00 – 0:40)
**Visual on Screen:** Title slide or README.md in VS Code showing Candidate Name, Registration Number, and Project Title.

**Spoken Script:**
> "Hello everyone. My name is [Your Name], registration number [Your Reg Number]. Today, I am presenting my solution for the Algions AI Labs Advanced ML Assessment: an **E-Commerce Product Recommendation Engine**.
> 
> In this project, our mission was to build a top-N product recommendation system that personalizes product discovery while operating strictly as a human-centered **decision-support tool**.
> 
> The core business decision is: given a user's historical browsing and purchasing behavior, which top-10 items should we surface to maximize relevance and discovery, without trapping the user in a popularity bubble or failing on cold-start users?"

---

### Scene 2: Data Pipeline & Zero-Leakage Chronological Split (0:40 – 1:20)
**Visual on Screen:** Terminal showing `python -m src.eda` output or `reports/eda_summary.json`.

**Spoken Script:**
> "Let's begin with data integrity and leakage prevention.
> 
> We synthesized a realistic e-commerce dataset spanning 1,200 users, 250 catalog products across 6 major retail categories, and nearly 20,000 interactions over a 90-day period.
> 
> As you can see from our EDA profiling, the interaction matrix has a 96.7% sparsity rate and follows a realistic e-commerce conversion funnel: views account for 62%, add-to-carts 22%, purchases 11%, and ratings 4.5%.
> 
> Crucially, to prevent future target leakage, we implemented a strict **chronological split**: the earliest 70% of events form the training set, the next 15% are reserved for hyperparameter tuning, and the final 15% are held out as an untouched test set. Our automated tests mathematically verify that `train_max_timestamp < val_min_timestamp < test_min_timestamp`, ensuring zero future data leaked into model training."

---

### Scene 3: Live Interactive UI & Cold-Start Demonstration (1:20 – 2:40)
**Visual on Screen:** Switch to the browser showing the Streamlit App (`http://localhost:8501`).

**Spoken Script:**
> "Now let's move to our live inference interface.
> 
> Here in our Streamlit application, all models are loaded dynamically from our serialized `.joblib` artifacts—there are no hard-coded sample predictions.
> 
> First, let's select a **Warm User**, `USER_0001`, an active electronics and gadget enthusiast. When we query our **Hybrid Recommender**, it delivers a tailored list of top products, such as wireless earbuds, gaming monitors, and fast-charging docks, each with real confidence scores and category tags.
> 
> Next, let's test one of the most critical requirements: **Cold-Start Handling**.
> 
> If we switch the user profile to a **Brand-New Cold-Start User** with zero training history, pure Collaborative Filtering fails completely. However, our Hybrid engine automatically detects this condition and shifts weights dynamically toward our Content-Based TF-IDF model and trending category popularity. As you can see on the badge, it gracefully activates the '❄️ Cold Fallback' signal, guaranteeing the user is never left with an empty screen.
> 
> Now let's open Tab 2: the **Explainability & Feature Signals Inspector**.
> When we inspect any recommended product, the system transparently breaks down the hybrid score into its constituent drivers: Collaborative Filtering latent score, Content-Based semantic score, and Popularity prior. It displays the exact overlapping keywords—like 'wireless' and 'noise-cancelling'—and presents a prominent Decision-Support Disclaimer reminding platform managers that human review is required for high-stakes merchandising decisions."

---

### Scene 4: Model Comparison & Held-Out Benchmark Results (2:40 – 3:45)
**Visual on Screen:** Click Tab 4 in Streamlit ("Offline Metrics & Benchmark") or show terminal running `python -m src.evaluate`.

**Spoken Script:**
> "Now, let's examine our quantitative evaluation on the held-out test set.
> 
> We benchmarked four distinct approaches against ranking metrics at $K=10$:
> 
> 1. Our **Popularity Baseline** achieved an NDCG@10 of 0.1149 and Recall@10 of 0.1592, but suffered from extreme popularity bias, achieving only **8.0% catalog coverage**.
> 2. Our **Collaborative Filtering SVD** model captured latent affinities for warm users, but collapsed on cold-start users.
> 3. Our **Content-Based TF-IDF** recommender achieved outstanding catalog coverage of **99.6%** and the highest novelty score of 9.24 bits.
> 4. Finally, our **Hybrid Recommender** proved superior across all metrics: it achieved an **NDCG@10 of 0.1545**—a **+34.5% lift over baseline**—a **Recall@10 of 0.2223**, and maintained **98.0% catalog coverage**.
> 
> In the cold-start test slice, the Hybrid engine achieved an NDCG@10 of **0.2489**, proving that dynamic ensemble adaptation is essential for real-world platforms."

---

### Scene 5: API Endpoint Verification & Acceptance Tests (3:45 – 4:30)
**Visual on Screen:** Switch to FastAPI Swagger UI (`http://127.0.0.1:8000/docs`) or terminal executing `pytest`.

**Spoken Script:**
> "To demonstrate deployment readiness, we also exposed our models via a FastAPI REST API.
> 
> The API provides `/recommend`, `/explain`, `/health`, and `/metrics` endpoints. When we execute a POST request to `/recommend`, it loads the artifact, validates the schema with Pydantic, and returns live predictions in milliseconds.
> 
> In our terminal, running `pytest tests -v` executes our full suite of 22 unit and integration tests, validating chronological splitting, analytical ranking metrics, artifact persistence, and API response contracts."

---

### Scene 6: Responsible AI & Closing (4:30 – 5:00)
**Visual on Screen:** Tab 5 in Streamlit ("Responsible AI & Decision Support") or README summary.

**Spoken Script:**
> "In conclusion, this recommendation engine is built around three pillars:
> 1. **Engineering Correctness:** Zero data leakage, defensible metrics, and 100% test pass rate.
> 2. **Algorithmic Strength:** A robust hybrid architecture outperforming the popularity baseline by 34.5% while surfacing 98% of the catalog.
> 3. **Responsible Decision Support:** Transparent explainability, cold-start safety, and human-in-the-loop oversight.
> 
> Thank you for your time, and I look forward to the technical walkthrough!"

---

## Tips for Recording
- Keep your mouse movements deliberate and smooth.
- Zoom in slightly on code and terminal text (Ctrl + `+` in VS Code / browser) so reviewers on laptops can read comfortably.
- If you make a small verbal slip, pause for 2 seconds and re-say the sentence; you can trim it in editing or leave it if natural.
