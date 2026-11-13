# TECHNICAL REPORT: E-COMMERCE PRODUCT RECOMMENDATION ENGINE
**Algions AI Labs Private Limited — Advanced Individual Project Assessment**
**Track:** AI & Machine Learning  
**Project:** E-Commerce Product Recommendation Engine (Top-N Decision-Support System)

---

## 1. Executive Summary & Problem Framing

### 1.1 Business Objective & Decision Context
In an e-commerce platform offering hundreds or thousands of products across diverse categories, customers face choice overload and discovery friction. The objective of this project is to develop an **explainable, top-N personalized product recommendation engine** that acts as an advisory **decision-support tool**.

The system assists both the customer (by personalizing product discovery feeds on home and product detail pages) and the platform's merchandising managers (by surfacing inventory, boosting long-tail discovery, and highlighting high-affinity items while identifying prediction uncertainty).

### 1.2 Formal Prediction Unit, Target, Horizon & Point-in-Time
- **Business Decision:** Which $N$ products should be displayed to user $u$ at recommendation opportunity $t$ to maximize discovery relevance, session engagement, and conversion while preventing catalog fatigue?
- **Prediction Unit:** An ordered ranking list of product identifiers $R_N(u) = [i_1, i_2, \dots, i_N]$ for target user $u$.
- **Target Variable:** Subsequent user engagement in the evaluation window. In this implicit feedback system, ground-truth relevance reflects real actions (clicks/views, add-to-carts, purchases, and ratings).
- **Prediction Horizon:** Immediate discovery window (the user's next browsing session in the held-out test timeframe).
- **Point-in-Time Principle:** When generating recommendations at timestamp $t_{\text{pred}}$, the models are strictly restricted to information, interactions, and catalog states that occurred **at or prior to $t_{\text{pred}}$**. All post-$t_{\text{pred}}$ interactions are quarantined.

---

## 2. Data Sourcing, Profiling & Exploratory Analysis (EDA)

### 2.1 Dataset Provenance & Synthesis Methodology
To ensure 100% reproducibility without external network or licensing dependencies, a synthetic e-commerce interaction dataset was programmatically generated using a fixed random seed (`RANDOM_SEED = 42`). 

The generation process mirrors real-world e-commerce dynamics:
1. **Catalog ($\mathcal{I}$):** 250 products across 6 major categories (*Electronics, Fashion, Home & Kitchen, Beauty & Personal Care, Sports & Fitness, Books & Stationery*). Each product has realistic titles, brand associations, subcategories, price points ($12.45 to $1,037.83, median $43.67), customer rating distributions, and rich metadata tags.
2. **Users ($\mathcal{U}$):** 1,200 synthetic user accounts categorized into activity tiers: Power Users (15%), Regular Users (60%), and Casual / Cold-Start Users (25%). Each user is assigned primary and secondary category preferences reflecting real taste clustering.
3. **Interactions ($\mathcal{D}$):** 19,930 chronological interaction events spanning 90 days. Interactions follow an authentic e-commerce funnel with Pareto (Zipf) popularity:
   - **View (62.32%, 12,420 events):** Top-of-funnel browsing.
   - **Add to Cart (21.98%, 4,381 events):** High purchase intent.
   - **Purchase (11.20%, 2,233 events):** Successful transaction.
   - **Rating (4.50%, 896 events):** Explicit satisfaction feedback (scale 1 to 5).

### 2.2 Data Quality, Integrity & Sparsity Audit
- **Matrix Sparsity:**
  $$\text{Sparsity} = \left( 1 - \frac{|\text{Unique User-Item Pairs}|}{|\mathcal{U}| \times |\mathcal{I}|} \right) \times 100\% = 96.67\%$$
  A sparsity of 96.67% closely matches production e-commerce platforms, requiring factorization and content representations to overcome interaction density bottlenecks.
- **Power-Law Item Popularity:** The top 20% most active products drive **61.65%** of all interactions, demonstrating the classic "head vs long-tail" distribution.
- **Referential Integrity:** 0 orphaned user IDs, 0 orphaned product IDs, and 0 duplicate interaction events.
- **Chronological Verification:** The interaction timestamps are strictly monotonically non-decreasing, ensuring reliable temporal slicing.

---

## 3. Leakage Analysis & Chronological Split Strategy

### 3.1 Prohibited Future & Target Leakage
Standard randomized $K$-fold cross-validation or uniform train/test splits are **prohibited** in temporal recommendation systems. Shuffling interactions in time introduces severe **temporal target leakage**: the model uses a future purchase on day 80 to predict a view on day 20, artificially inflating evaluation scores and failing catastrophically in production.

### 3.2 Implemented Temporal Splitting Strategy
All interactions were ordered chronologically and divided into three non-overlapping temporal windows:
1. **Training Set (70% earliest events, 13,951 events):**
   - Temporal window: `2026-08-07 09:22:27` to `2026-11-04 15:35:50`
   - Purpose: Model parameter learning, item embeddings, and vocabulary extraction.
2. **Validation Set (15% middle events, 2,989 events):**
   - Temporal window: `2026-11-04 15:37:50` to `2026-11-10 05:25:25`
   - Purpose: Hyperparameter tuning (grid search for hybrid ensemble weights).
3. **Held-Out Test Set (15% final events, 2,990 events):**
   - Temporal window: `2026-11-10 05:26:32` to `2026-11-18 08:43:49`
   - Purpose: Final unbiased benchmarking. Zero tuning was performed on this slice.

$$\max(T_{\text{train}}) < \min(T_{\text{val}}) < \max(T_{\text{val}}) < \min(T_{\text{test}})$$

*Formal Assertion:* All unit tests rigorously assert this inequality (`tests/test_data_and_split.py`), proving zero temporal leakage.

---

## 4. Model Architectures & Feature Engineering

### 4.1 Feature Engineering & Implicit Feedback Confidence
In e-commerce, unobserved user-item pairs are not negative; they are simply unknown. We assign confidence weights $c_{u,i}$ to each event based on commitment depth:
$$w(\text{event}) = \begin{cases} 
1.0 & \text{if event = 'view'} \\
2.5 & \text{if event = 'cart'} \\
5.0 & \text{if event = 'purchase'} \\
1.0 + \frac{\text{rating}}{5.0} \times 4.0 & \text{if event = 'rating'} \in [1, 5]
\end{cases}$$

For multiple interactions by user $u$ with item $i$, we compute the total log-scaled interaction affinity:
$$R_{u, i} = \log\left(1 + \sum_{e \in E_{u,i}} w(e)\right)$$

### 4.2 Benchmark Baseline: Popularity Recommender
- **Formulation:** Computes global and category-level total interaction weights for each product:
  $$S_{\text{pop}}(i) = \frac{\sum_{u} R_{u,i}}{\max_{j} \sum_{u} R_{u,j}}$$
- **Role:** Non-personalized industry benchmark. Demonstrates what can be achieved purely by promoting head items.

### 4.3 Personalized Candidate 1: Collaborative Filtering (TruncatedSVD)
- **Formulation:** Implicit interaction matrix $\mathbf{R} \in \mathbb{R}^{|\mathcal{U}| \times |\mathcal{I}|}$ is decomposed into low-rank latent factor spaces:
  $$\mathbf{R} \approx \mathbf{U}_k \mathbf{\Sigma}_k \mathbf{V}_k^T$$
  - User representation: $\mathbf{P} = \mathbf{U}_k \mathbf{\Sigma}_k^{1/2} \in \mathbb{R}^{|\mathcal{U}| \times k}$ ($k=30$ latent factors).
  - Item representation: $\mathbf{Q} = \mathbf{V}_k \mathbf{\Sigma}_k^{1/2} \in \mathbb{R}^{|\mathcal{I}| \times k}$.
- **Scoring:** The preference of user $u$ for item $i$ is estimated by the latent factor inner product:
  $$\hat{R}_{u, i} = \sigma(\mathbf{p}_u \cdot \mathbf{q}_i^T) = \frac{1}{1 + e^{-\mathbf{p}_u \cdot \mathbf{q}_i^T}}$$
- **Strengths:** Captures subtle co-interaction patterns without relying on text metadata.
- **Weaknesses:** Complete cold-start failure mode when $u \notin \mathcal{U}_{\text{train}}$ or $i \notin \mathcal{I}_{\text{train}}$.

### 4.4 Personalized Candidate 2: Content-Based Recommender (TF-IDF Text Profiles)
- **Formulation:** Product metadata (title, category, subcategory, brand, description, tags) is vectorized using TF-IDF with uni-grams and bi-grams:
  $$\mathbf{x}_i = \text{TF-IDF}(\text{text}_i) \in \mathbb{R}^{M}, \quad \|\mathbf{x}_i\|_2 = 1$$
- **User Profiling:** An aggregate user taste vector is constructed by weighting past item vectors by interaction affinity:
  $$\mathbf{u}_p = \frac{\sum_{i \in \mathcal{H}_u} R_{u,i} \mathbf{x}_i}{\left\| \sum_{i \in \mathcal{H}_u} R_{u,i} \mathbf{x}_i \right\|_2}$$
- **Scoring:** Cosine similarity between user profile and candidate product:
  $$S_{\text{content}}(u, i) = \mathbf{u}_p \cdot \mathbf{x}_i^T$$
- **Strengths:** Solves item cold-start; highly interpretable via overlapping keyword extraction.

### 4.5 Candidate 3: Dynamic Hybrid Recommender
- **Ensemble Formulation:** Combines Collaborative Filtering, Content-Based semantic matching, and Popularity priors:
  $$S_{\text{hybrid}}(u, i) = \alpha \cdot \hat{R}_{u,i} + \beta \cdot S_{\text{content}}(u, i) + \gamma \cdot S_{\text{pop}}(i)$$
- **Optimal Hyperparameters (tuned on Validation Set):**
  - $\alpha = 0.40$ (Collaborative Filtering)
  - $\beta = 0.40$ (Content-Based TF-IDF)
  - $\gamma = 0.20$ (Popularity prior)
- **Dynamic Cold-Start Adaptation:**
  - When user $u$ is a **Cold-Start User** ($|\mathcal{H}_u| \le 2$ interactions in train):
    $$\alpha \leftarrow 0.0, \quad \beta \leftarrow 0.40 \text{ (using declared onboarding categories)}, \quad \gamma \leftarrow 0.60$$
  - When item $i$ is a **Cold-Start Item** (unseen in CF matrix):
    $\hat{R}_{u,i} = 0$, relying smoothly on content similarity $S_{\text{content}}(u, i)$.

---

## 5. Evaluation Methodology & Empirical Results

### 5.1 Metric Justification
Standard classification metrics (e.g. Accuracy) are fundamentally unsuited for recommendation engines due to extreme negative imbalance. We employ top-$K$ ranking metrics:
1. **Precision@K:** Fraction of top-$K$ recommendations that the user actually interacted with in the test window.
2. **Recall@K:** Fraction of the user's total test set interactions that were captured in the top-$K$ list.
3. **NDCG@K (Normalized Discounted Cumulative Gain):** Rewards placing relevant items higher up the ranked list using logarithmic rank discounting:
   $$\text{DCG@K} = \sum_{r=1}^K \frac{rel_r}{\log_2(r + 1)}, \quad \text{NDCG@K} = \frac{\text{DCG@K}}{\text{IDCG@K}}$$
4. **MAP@K (Mean Average Precision):** Evaluates rank-weighted precision across all test queries.
5. **Catalog Coverage (%):** Percentage of unique catalog items recommended at least once across all test users. High coverage prevents winner-take-all popularity monopolization.
6. **Novelty (Bits):** Average self-information of recommended items $-\log_2 P(i)$, quantifying catalog discovery beyond head items.

### 5.2 Held-Out Test Set Benchmark Results ($K = 10$)

```text
===============================================================================================
                          OVERALL TEST SET BENCHMARK SUMMARY (K = 10)                          
===============================================================================================
Model                          | Prec@10   | Recall@10  | NDCG@10   | MAP@10   | Coverage   | Novelty 
-----------------------------------------------------------------------------------------------
Popularity Baseline            | 0.0580    | 0.1592     | 0.1149    | 0.0623   | 8.0      % | 5.47    
Collaborative Filtering (SVD)  | 0.0341    | 0.1285     | 0.0829    | 0.0472   | 61.2     % | 7.42    
Content-Based (TF-IDF)         | 0.0456    | 0.1305     | 0.0848    | 0.0419   | 99.6     % | 9.24    
Hybrid Recommender             | 0.0776    | 0.2223     | 0.1545    | 0.0855   | 98.0     % | 7.22    
===============================================================================================
```

### 5.3 Cold-Start Test Users Slice ($K = 10$, Users with $\le 2$ Training Events)

```text
===============================================================================================
                          COLD-START USERS BENCHMARK SUMMARY (K = 10)                          
===============================================================================================
Model                          | Prec@10   | Recall@10  | NDCG@10   | MAP@10   | Coverage   | Novelty 
-----------------------------------------------------------------------------------------------
Popularity Baseline            | 0.1078    | 0.2371     | 0.1925    | 0.1033   | 4.4      % | 5.38    
Collaborative Filtering (SVD)  | 0.0065    | 0.0214     | 0.0163    | 0.0100   | 27.2     % | 7.43    
Content-Based (TF-IDF)         | 0.0725    | 0.1433     | 0.1057    | 0.0497   | 42.8     % | 9.28    
Hybrid Recommender             | 0.1451    | 0.3238     | 0.2489    | 0.1388   | 24.4     % | 5.57    
===============================================================================================
```

### 5.4 Empirical Findings & Business Interpretation
1. **Accuracy Lift:** The **Hybrid Recommender** achieves the highest ranking quality across the entire benchmark, outperforming the Popularity Baseline by **+34.5% in NDCG@10** (0.1545 vs 0.1149) and **+39.6% in Recall@10** (0.2223 vs 0.1592).
2. **Catalog Health & Coverage:** The Popularity Baseline surfaces only **8.0%** of the catalog, effectively ignoring 92% of the inventory. In contrast, the Hybrid Recommender achieves **98.0% catalog coverage**, ensuring discovery of long-tail and high-margin products.
3. **Cold-Start Resilience:** For cold-start users, pure Collaborative Filtering fails catastrophically (NDCG@10 drops from 0.0829 down to 0.0163). The Hybrid Recommender’s dynamic weighting avoids this failure mode, achieving **NDCG@10 = 0.2489**.

---

## 6. Error Analysis & Failure Modes

### 6.1 False Positive Inspection
- **Example Case:** User `USER_0045` viewed high-end electronics (`Ultra-Wide Curved Gaming Monitor 34-inch`). The collaborative model strongly recommended a `$899.99 Smart 4K OLED TV`. In the test period, the user actually purchased a `$49.99 Portable Power Bank`.
- **Root Cause:** Both the monitor and the TV share strong latent factor alignment in the "high-ticket electronics" subspace. However, the user’s real-time intent was constrained by budget and immediate portable charging utility.
- **Remediation:** Introduce real-time session price constraints and price-band filters into the inference ranking stage.

### 6.2 Filter Bubble & Over-Specialization Risk
Pure content-based filtering has a tendency to recommend identical items (e.g. recommending 5 different mechanical keyboards to a user who just bought one). 
- **Remediation:** The Hybrid engine penalizes intra-list category redundancy and incorporates the collaborative latent space to surface cross-category complementary products (e.g., ergonomic wrist rest or mouse with a keyboard).

---

## 7. Explainability, Uncertainty & Responsible AI

### 7.1 Decision-Support Transparency
Every recommendation served by the API and UI exposes a structured rationale:
1. **Multi-Score Attribution:** Displays exact percentage contributions from Collaborative Filtering ($\alpha$), Content-Based TF-IDF ($\beta$), and Popularity ($\gamma$).
2. **Feature Driver Extraction:** Highlights the top matched TF-IDF keywords (e.g. `['noise-cancelling', 'wireless', 'over-ear']`).
3. **Co-Interaction Linkage:** Identifies the user’s past interaction that drove the collaborative latent affinity.

### 7.2 Human Oversight & Safe Production Deployment
- **Not an Autonomous Agent:** The model functions strictly as an advisory discovery tool. E-commerce merchandisers retain ultimate authority to configure manual category boosts, holiday promotions, and out-of-stock suppressions.
- **Privacy Preservation:** The models train purely on anonymized behavioral sequences and catalog text. No PII, customer names, contact info, or payment credentials are ingested.
- **Fairness:** Category-level balancing ensures smaller vendors and newly launched items receive exposure through content matching rather than being eclipsed by incumbent bestsellers.

---

## 8. Acceptance Test Evidence Summary

| Acceptance Test | Requirement | Demonstrated Evidence |
| :--- | :--- | :--- |
| **Test 1** | Reproduce dataset split and training run from README | Verified via `python -m src.preprocess` & `python -m src.train`. Artifacts generated deterministically. |
| **Test 2** | Show baseline and candidate model metrics on held-out test set | Verified via `python -m src.evaluate`. Benchmark table saved to `artifacts/evaluation_metrics.json`. |
| **Test 3** | Submit real prediction through API/UI loading saved model | Verified via FastAPI `/recommend` endpoint and Streamlit UI. Real floating-point scores and titles returned. |
| **Test 4** | Explain one prediction/error, inputs, and limitations | Verified via FastAPI `/explain` endpoint and Streamlit Inspector panel with human review notice. |
| **Test 5** | Demonstrate no future leakage or sensitive data used | Verified via `tests/test_data_and_split.py` asserting non-overlapping timestamp bounds and zero PII. |
