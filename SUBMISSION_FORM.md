# CANDIDATE SUBMISSION FORM DETAILS
**ALGIONS AI LABS PRIVATE LIMITED**
**AI & Machine Learning Track | Advanced Individual Project Assessment**

Use this pre-formatted template to copy and paste directly into the official Algions AI Labs assessment submission portal.

---

### 1. Candidate Identification
| Field | Value | Notes / Instructions |
| :--- | :--- | :--- |
| **Full Name** | `[Enter your Full Name]` | Replace placeholder before submission |
| **Registration Number** | `[Enter your Registration Number]` | As stated in your official invitation |
| **University Email** | `[Enter your University / College Email]` | Official email address for confirmation |

---

### 2. Assignment Confirmation
| Field | Value |
| :--- | :--- |
| **Internship Role** | `AI & Machine Learning` |
| **Assigned Project Title** | `E-Commerce Product Recommendation Engine` |

---

### 3. Code and Version
| Field | Value | Notes |
| :--- | :--- | :--- |
| **Repository URL** | `https://github.com/[your-username]/ecommerce-recommender` | Ensure repository is public or invite reviewer |
| **Primary Branch** | `main` | Default branch |
| **Final Commit ID** | `[git rev-parse HEAD]` | Run `git log -1 --format="%H"` to copy exact hash |
| **Reviewer Access** | Granted / Public repository | Reviewers will clone and test locally |

---

### 4. Demonstration
| Field | Value |
| :--- | :--- |
| **3–5 Minute Video Link** | `[Insert your Loom / YouTube / Google Drive unlisted video link]` |
| **Live Demo URL** | `Local-only demo (Streamlit on :8501 and FastAPI on :8000)` |

---

### 5. Documentation
| Field | Value |
| :--- | :--- |
| **README Link** | `https://github.com/[your-username]/ecommerce-recommender/blob/main/README.md` |
| **Technical Report Link** | `https://github.com/[your-username]/ecommerce-recommender/blob/main/TECHNICAL_REPORT.md` |
| **Research Notebook Link**| `https://github.com/[your-username]/ecommerce-recommender/blob/main/notebooks/recommendation_engine_analysis.ipynb` |
| **Architecture Diagram** | Included in `README.md` (Mermaid flowchart) and `TECHNICAL_REPORT.md` |

---

### 6. Technology and Limitations
| Field | Detail |
| :--- | :--- |
| **Technology Stack** | Python 3.11, Scikit-Learn 1.8.0, SciPy 1.15.2, Pandas, NumPy, FastAPI 0.129.0, Uvicorn 0.41.0, Streamlit 1.52.1, PyTest 9.2.0 |
| **Setup Notes** | `pip install -r requirements.txt`, followed by `python -m src.train`, `python -m src.evaluate`, and `streamlit run app.py` |
| **Known Incomplete Features / Assumptions** | Assumes static product catalog inventory; real-time warehouse stock syncing and distributed streaming ingestion (e.g., Kafka) are left for future production scaling. |

---

### 7. AI and Attribution
| Field | Detail |
| :--- | :--- |
| **AI Tools Used** | Antigravity AI Assistant |
| **Purpose of Use** | Architecture design assistance, test scaffolding, and documentation structuring. |
| **Originality & Understanding**| All algorithms (SVD decomposition, TF-IDF cosine matching, dynamic hybrid fallback logic, ranking metric formulations) were verified, tested, and understood by the candidate for the upcoming technical walkthrough and live coding task. |
| **Dataset Attribution** | 100% reproducible synthetic e-commerce dataset generated via `src/data_generator.py` (MIT License), modeling Zipfian power laws and realistic interaction funnels. |

---

### 8. Declarations
- [x] **Originality:** I declare that this submission represents my own work and engineering understanding.
- [x] **Walkthrough Readiness:** I am fully prepared to explain the architectural decisions, mathematical foundations, and code in the 15-minute technical walkthrough and complete the 10-minute live debugging task.
- [x] **No Sensitive Data:** No real customer PII, confidential credentials, passwords, or production API keys were uploaded or committed.
- [x] **Link Verification:** All links to the repository, documentation, and video demonstration have been verified and permissions are open for reviewers.

---

### 9. Role-Specific Form Fields
| Role-Specific Field | Details |
| :--- | :--- |
| **Dataset Source / License** | Synthetic e-commerce interaction dataset (1,200 users, 250 products, 19,930 events). Fully reproducible via `src/data_generator.py` with seed 42. MIT License. |
| **Notebook / Report Link** | `notebooks/recommendation_engine_analysis.ipynb` & `TECHNICAL_REPORT.md` |
| **Model Artifact Location** | `artifacts/hybrid_model.joblib`, `collaborative_model.joblib`, `content_model.joblib`, `popularity_model.joblib` |
| **Evaluation Metrics Summary**| **Overall Test Set ($K=10$):** Popularity Baseline: NDCG 0.1149, Recall 0.1592, Coverage 8.0%. **Hybrid Recommender: NDCG 0.1545 (+34.5% lift), Recall 0.2223 (+39.6% lift), Coverage 98.0%, Novelty 7.22 bits.** Cold-Start Test Slice: Hybrid NDCG 0.2489 vs CF NDCG 0.0163. |
| **Inference API / App URL** | REST API: `http://127.0.0.1:8000` (Endpoints: `/recommend`, `/explain`, `/health`, `/metrics`), Web UI: `http://localhost:8501` |
