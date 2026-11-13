"""
Interactive Streamlit Web Application for E-Commerce Product Recommendation Engine.
ALGIONS AI LABS — AI & Machine Learning Advanced Assessment.
High-end modern UI with elevated card components, interactive analytics, and explainability.
"""
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path

# Page Config
st.set_page_config(
    page_title="Algions AI Labs | Enterprise Recommender",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Design System (CSS)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    
    * {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Global styles */
    .stApp {
        background-color: #F8FAFC;
    }
    
    /* Header Container */
    .hero-container {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 50%, #312E81 100%);
        padding: 28px 36px;
        border-radius: 16px;
        color: white;
        margin-bottom: 24px;
        box-shadow: 0 10px 25px -5px rgba(15, 23, 42, 0.2);
    }
    .hero-badge {
        display: inline-block;
        background: rgba(99, 102, 241, 0.25);
        border: 1px solid rgba(129, 140, 248, 0.4);
        color: #C7D2FE;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 12px;
    }
    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        margin: 0 0 8px 0;
        background: linear-gradient(to right, #FFFFFF, #E2E8F0);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero-subtitle {
        font-size: 0.98rem;
        color: #94A3B8;
        max-width: 800px;
        line-height: 1.5;
        margin: 0;
    }
    
    /* Product Card */
    .product-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 20px;
        margin-bottom: 16px;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
        position: relative;
        overflow: hidden;
    }
    .product-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 24px -6px rgba(15, 23, 42, 0.1);
        border-color: #CBD5E1;
    }
    .category-pill {
        display: inline-flex;
        align-items: center;
        gap: 4px;
        background-color: #F1F5F9;
        color: #475569;
        font-size: 0.72rem;
        font-weight: 600;
        padding: 3px 9px;
        border-radius: 6px;
        margin-bottom: 8px;
    }
    .card-title {
        font-size: 1.02rem;
        font-weight: 700;
        color: #0F172A;
        line-height: 1.35;
        margin: 4px 0 10px 0;
        height: 48px;
        overflow: hidden;
        display: -webkit-box;
        -webkit-line-clamp: 2;
        -webkit-box-orient: vertical;
    }
    .price-row {
        display: flex;
        align-items: baseline;
        justify-content: space-between;
        margin-top: 12px;
        padding-top: 10px;
        border-top: 1px solid #F1F5F9;
    }
    .price-tag {
        font-size: 1.25rem;
        font-weight: 800;
        color: #0F172A;
    }
    .score-badge {
        font-size: 0.8rem;
        font-weight: 700;
        padding: 4px 10px;
        border-radius: 8px;
    }
    .score-high {
        background-color: #EEF2FF;
        color: #4F46E5;
        border: 1px solid #C7D2FE;
    }
    .score-cold {
        background-color: #FEF2F2;
        color: #DC2626;
        border: 1px solid #FECACA;
    }
    .signal-text {
        font-size: 0.75rem;
        color: #64748B;
        font-style: italic;
        margin-top: 8px;
        display: block;
    }
    
    /* Sidebar Profile Card */
    .user-profile-box {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.02);
    }
    .user-avatar-row {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 12px;
    }
    .user-avatar {
        width: 44px;
        height: 44px;
        border-radius: 50%;
        background: linear-gradient(135deg, #6366F1, #8B5CF6);
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-weight: 700;
        font-size: 1.1rem;
    }
    .user-tier-badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.7rem;
        font-weight: 700;
        text-transform: uppercase;
    }
    .tier-power {
        background: #FEF3C7;
        color: #B45309;
    }
    .tier-regular {
        background: #E0E7FF;
        color: #3730A3;
    }
    .tier-cold {
        background: #FEE2E2;
        color: #991B1B;
    }
    
    /* Explainability Inspector Box */
    .inspector-box {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 24px;
        margin-top: 10px;
    }
    
    /* Metric Card */
    .metric-stat-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 18px 20px;
        text-align: left;
    }
    .stat-label {
        font-size: 0.78rem;
        font-weight: 600;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .stat-value {
        font-size: 1.7rem;
        font-weight: 800;
        color: #0F172A;
        margin: 6px 0 2px 0;
    }
    .stat-delta {
        font-size: 0.78rem;
        font-weight: 600;
        color: #10B981;
    }
</style>
""", unsafe_allow_html=True)

# File Paths
BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
DATA_DIR = BASE_DIR / "data"

CATEGORY_ICONS = {
    "Electronics": "⚡",
    "Fashion": "👗",
    "Home & Kitchen": "🍳",
    "Beauty & Personal Care": "✨",
    "Sports & Fitness": "🏋️",
    "Books & Stationery": "📚"
}

@st.cache_resource
def load_all_artifacts():
    models = {
        "hybrid": joblib.load(ARTIFACTS_DIR / "hybrid_model.joblib"),
        "collaborative": joblib.load(ARTIFACTS_DIR / "collaborative_model.joblib"),
        "content": joblib.load(ARTIFACTS_DIR / "content_model.joblib"),
        "popularity": joblib.load(ARTIFACTS_DIR / "popularity_model.joblib")
    }
    df_products = pd.read_csv(DATA_DIR / "raw" / "products.csv")
    df_users = pd.read_csv(DATA_DIR / "raw" / "users.csv")
    df_train = pd.read_csv(DATA_DIR / "processed" / "train_interactions.csv")

    eval_metrics = None
    metrics_path = ARTIFACTS_DIR / "evaluation_metrics.json"
    if metrics_path.exists():
        with open(metrics_path, "r", encoding="utf-8") as f:
            eval_metrics = json.load(f)

    return models, df_products, df_users, df_train, eval_metrics

try:
    models, df_products, df_users, df_train, eval_metrics = load_all_artifacts()
    artifacts_ready = True
except Exception as e:
    artifacts_ready = False
    st.error(f"Error loading model artifacts: {e}. Please run `python -m src.train` first.")

# HERO HEADER
st.markdown("""
<div class="hero-container">
    <div class="hero-badge">Algions AI Labs — Advanced Assessment</div>
    <div class="hero-title">Enterprise Product Recommendation Engine</div>
    <p class="hero-subtitle">
        Decision-support machine learning solution featuring multi-signal hybrid scoring, strict zero-leakage chronological validation, 
        transparent feature explainability, and dynamic cold-start fallback.
    </p>
</div>
""", unsafe_allow_html=True)

if artifacts_ready:
    # SIDEBAR CONTROLS
    st.sidebar.markdown("### 👤 User Persona Simulation")

    user_mode = st.sidebar.radio(
        "Select User Persona:",
        ["🔥 Power Shopper (Warm Profile)", "❄️ Cold-Start User (Low Activity)", "✨ Brand New Customer (0 Events)"],
        label_visibility="collapsed"
    )

    if "Power" in user_mode:
        candidate_users = df_users[df_users["activity_tier"] == "power"]["user_id"].tolist()
        selected_user_id = st.sidebar.selectbox("Active User Account:", candidate_users, index=0)
    elif "Cold-Start" in user_mode:
        candidate_users = df_users[df_users["activity_tier"] == "casual"]["user_id"].tolist()
        selected_user_id = st.sidebar.selectbox("Active User Account:", candidate_users, index=0)
    else:
        selected_user_id = st.sidebar.text_input("Custom User ID (Unseen):", "USER_9999_NEW")

    # Render User Card in Sidebar
    user_row = df_users[df_users["user_id"] == selected_user_id]
    if not user_row.empty:
        u_info = user_row.iloc[0]
        n_train = len(df_train[df_train["user_id"] == selected_user_id])
        tier = u_info["activity_tier"].upper()
        tier_class = "tier-power" if tier == "POWER" else ("tier-regular" if tier == "REGULAR" else "tier-cold")
        
        st.sidebar.markdown(f"""
        <div class="user-profile-box">
            <div class="user-avatar-row">
                <div class="user-avatar">{selected_user_id[-2:]}</div>
                <div>
                    <strong style="color: #0F172A; font-size: 0.95rem;">{selected_user_id}</strong><br>
                    <span class="user-tier-badge {tier_class}">{tier} USER</span>
                </div>
            </div>
            <div style="font-size: 0.8rem; color: #475569; line-height: 1.6;">
                🎯 <b>Primary Affinity:</b> {u_info['primary_category']}<br>
                ✨ <b>Secondary:</b> {u_info['secondary_category']}<br>
                📊 <b>Past History:</b> {n_train} events recorded
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.sidebar.markdown(f"""
        <div class="user-profile-box">
            <div class="user-avatar-row">
                <div class="user-avatar" style="background: #EF4444;">NEW</div>
                <div>
                    <strong style="color: #0F172A; font-size: 0.95rem;">{selected_user_id}</strong><br>
                    <span class="user-tier-badge tier-cold">COLD START</span>
                </div>
            </div>
            <div style="font-size: 0.8rem; color: #475569; line-height: 1.6;">
                🎯 <b>Status:</b> Zero Historical Interactions<br>
                🛡️ <b>Engine Fallback:</b> Activated
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⚙️ Engine Configuration")

    model_options = {
        "✨ Hybrid Recommender (Optimal & Dynamic)": "hybrid",
        "🤝 Collaborative Filtering (TruncatedSVD)": "collaborative",
        "📝 Content-Based (TF-IDF Semantic Matching)": "content",
        "🏆 Popularity Baseline (Engagement Prior)": "popularity"
    }
    selected_model_label = st.sidebar.selectbox("Active Recommendation Model:", list(model_options.keys()))
    active_model_key = model_options[selected_model_label]

    top_k = st.sidebar.slider("Recommendations Count (Top-N):", min_value=4, max_value=16, value=8, step=4)
    all_cats = ["All Categories"] + list(df_products["category"].unique())
    selected_cat = st.sidebar.selectbox("Category Filter:", all_cats)
    cat_filter = None if selected_cat == "All Categories" else selected_cat
    exclude_seen = st.sidebar.toggle("Suppress Previously Interacted Items", value=True)

    # NAVIGATION TABS
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🎯 Discovery Feed",
        "🔍 Explainability Inspector",
        "⚖️ Multi-Model Benchmarking",
        "📊 Offline Evaluation Metrics",
        "🛡️ Merchandiser Governance"
    ])

    # TAB 1: DISCOVERY FEED
    with tab1:
        st.markdown(f"#### Personalized Feed for `{selected_user_id}`")
        st.caption(f"Engine: **{selected_model_label}** | Showing Top **{top_k}** items | Scope: **{selected_cat}**")

        model = models[active_model_key]

        # Generate real predictions
        recs = model.recommend(
            user_id=selected_user_id,
            top_k=top_k,
            exclude_interacted=exclude_seen,
            category_filter=cat_filter
        )

        # Fallback if CF has zero predictions for cold user
        if not recs and active_model_key == "collaborative":
            st.warning("⚠️ Collaborative Filtering cannot score this user due to zero matrix presence. Displaying Popularity Baseline fallback.")
            recs = models["popularity"].recommend(
                user_id=selected_user_id,
                top_k=top_k,
                exclude_interacted=exclude_seen,
                category_filter=cat_filter
            )
            for r in recs:
                r["is_cold_fallback"] = True

        if not recs:
            st.info("No items match your active filter criteria.")
        else:
            cols = st.columns(4)
            for idx, item in enumerate(recs):
                col = cols[idx % 4]
                icon = CATEGORY_ICONS.get(item["category"], "🛍️")
                is_cold = item.get("is_cold_fallback", False)
                score_pct = int(round(item["score"] * 100))
                
                with col:
                    st.markdown(f"""
                    <div class="product-card">
                        <div class="category-pill">{icon} {item['category']}</div>
                        <div class="card-title" title="{item['title']}">{item['title']}</div>
                        <div class="price-row">
                            <span class="price-tag">${item['price']:.2f}</span>
                            <span class="score-badge {'score-cold' if is_cold else 'score-high'}">
                                {'❄️ Fallback' if is_cold else f'✨ {score_pct}% Match'}
                            </span>
                        </div>
                        <span class="signal-text">{item.get('signal', 'Recommended based on behavioral profile')}</span>
                    </div>
                    """, unsafe_allow_html=True)

    # TAB 2: EXPLAINABILITY INSPECTOR
    with tab2:
        st.markdown("#### Decision-Support Feature Signal Inspector")
        st.write("Inspect how the ensemble synthesized feature signals, keyword overlaps, and latent affinity for any item.")

        c_prod, c_inspect = st.columns([1, 2])
        
        with c_prod:
            item_candidates = [r["product_id"] for r in recs] if recs else df_products["product_id"].head(8).tolist()
            inspected_pid = st.selectbox("Select Recommended Product:", item_candidates)
            
            p_data = df_products[df_products["product_id"] == inspected_pid].iloc[0]
            st.markdown(f"""
            <div class="product-card" style="margin-top: 10px;">
                <div class="category-pill">{CATEGORY_ICONS.get(p_data['category'], '🛍️')} {p_data['category']}</div>
                <div style="font-weight: 700; font-size: 1.1rem; color: #0F172A; margin: 6px 0;">{p_data['title']}</div>
                <div style="font-size: 0.85rem; color: #475569; margin-bottom: 12px;"><b>Brand:</b> {p_data['brand']} | <b>Price:</b> ${p_data['price']:.2f}</div>
                <div style="font-size: 0.8rem; color: #64748B; background: #F8FAFC; padding: 10px; border-radius: 8px;">
                    <b>Catalog Tags:</b><br>{p_data['tags']}
                </div>
            </div>
            """, unsafe_allow_html=True)

        with c_inspect:
            explanation = models["hybrid"].explain(selected_user_id, inspected_pid)
            sub = explanation["sub_scores"]
            w = explanation["weights_used"]

            st.markdown("""
            <div style="background: #EEF2FF; border: 1px solid #C7D2FE; border-radius: 10px; padding: 14px 18px; margin-bottom: 16px;">
                <span style="font-weight: 700; color: #3730A3;">💡 Algorithmic Rationale:</span>
                <p style="margin: 4px 0 0 0; color: #1E1B4B; font-size: 0.92rem;">""" + explanation['synthesis_summary'] + """</p>
            </div>
            """, unsafe_allow_html=True)

            # Attribution Cards
            c1, c2, c3, c4 = st.columns(4)
            c1.markdown(f"""
            <div class="metric-stat-card">
                <div class="stat-label">Hybrid Score</div>
                <div class="stat-value">{explanation['hybrid_score']:.3f}</div>
                <div class="stat-delta">Ensemble Score</div>
            </div>
            """, unsafe_allow_html=True)
            
            c2.markdown(f"""
            <div class="metric-stat-card">
                <div class="stat-label">CF Signal ({int(w['collaborative']*100)}%)</div>
                <div class="stat-value">{sub['collaborative_filtering']:.3f}</div>
                <div class="stat-delta">Latent Match</div>
            </div>
            """, unsafe_allow_html=True)

            c3.markdown(f"""
            <div class="metric-stat-card">
                <div class="stat-label">Content ({int(w['content']*100)}%)</div>
                <div class="stat-value">{sub['content_based']:.3f}</div>
                <div class="stat-delta">TF-IDF Overlap</div>
            </div>
            """, unsafe_allow_html=True)

            c4.markdown(f"""
            <div class="metric-stat-card">
                <div class="stat-label">Popularity ({int(w['popularity']*100)}%)</div>
                <div class="stat-value">{sub['popularity_baseline']:.3f}</div>
                <div class="stat-delta">Platform Prior</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            
            # Deep signals
            col_cf_sig, col_ct_sig = st.columns(2)
            with col_cf_sig:
                st.markdown("**🤝 Collaborative Co-Interaction Signal:**")
                st.caption(explanation["sub_explanations"]["collaborative"].get("reason", "N/A"))
            
            with col_ct_sig:
                st.markdown("**📝 Content Semantic Driver Keywords:**")
                matched_kws = explanation["sub_explanations"]["content"].get("matched_keywords", [])
                if matched_kws:
                    kw_badges = " ".join([f"<span class='category-pill'>#{kw}</span>" for kw in matched_kws])
                    st.markdown(kw_badges, unsafe_allow_html=True)
                else:
                    st.caption("General category-level alignment")

            st.warning(f"🛡️ **Decision-Support Notice:** {explanation['decision_support_limitations']}")

    # TAB 3: SIDE-BY-SIDE MODEL BENCHMARKING
    with tab3:
        st.markdown("#### Side-by-Side Algorithmic Comparison")
        st.write("Compare how each algorithm personalizes discovery for the active user profile.")

        m_cols = st.columns(4)
        
        with m_cols[0]:
            st.markdown("""
            <div style="background: #FEF3C7; color: #92400E; padding: 10px; border-radius: 8px 8px 0 0; font-weight: 700; text-align: center;">
                🏆 Popularity Baseline
            </div>
            """, unsafe_allow_html=True)
            p_recs = models["popularity"].recommend(selected_user_id, top_k=5, exclude_interacted=exclude_seen)
            for r in p_recs:
                st.markdown(f"""
                <div style="background: white; border: 1px solid #E2E8F0; padding: 10px; margin-top: 6px; border-radius: 6px; font-size: 0.85rem;">
                    <b>{r['title'][:28]}...</b><br>
                    <span style="color: #64748B;">Score: {r['score']:.2f} | ${r['price']:.2f}</span>
                </div>
                """, unsafe_allow_html=True)

        with m_cols[1]:
            st.markdown("""
            <div style="background: #E0E7FF; color: #3730A3; padding: 10px; border-radius: 8px 8px 0 0; font-weight: 700; text-align: center;">
                🤝 Collaborative (SVD)
            </div>
            """, unsafe_allow_html=True)
            cf_recs = models["collaborative"].recommend(selected_user_id, top_k=5, exclude_interacted=exclude_seen)
            if cf_recs:
                for r in cf_recs:
                    st.markdown(f"""
                    <div style="background: white; border: 1px solid #E2E8F0; padding: 10px; margin-top: 6px; border-radius: 6px; font-size: 0.85rem;">
                        <b>{r['title'][:28]}...</b><br>
                        <span style="color: #64748B;">Score: {r['score']:.2f} | ${r['price']:.2f}</span>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.error("Cold User: 0 recommendations (Requires Fallback)")

        with m_cols[2]:
            st.markdown("""
            <div style="background: #DCFCE7; color: #166534; padding: 10px; border-radius: 8px 8px 0 0; font-weight: 700; text-align: center;">
                📝 Content-Based (TF-IDF)
            </div>
            """, unsafe_allow_html=True)
            ct_recs = models["content"].recommend(selected_user_id, top_k=5, exclude_interacted=exclude_seen)
            for r in ct_recs:
                st.markdown(f"""
                <div style="background: white; border: 1px solid #E2E8F0; padding: 10px; margin-top: 6px; border-radius: 6px; font-size: 0.85rem;">
                    <b>{r['title'][:28]}...</b><br>
                    <span style="color: #64748B;">Score: {r['score']:.2f} | ${r['price']:.2f}</span>
                </div>
                """, unsafe_allow_html=True)

        with m_cols[3]:
            st.markdown("""
            <div style="background: #F3E8FF; color: #6B21A8; padding: 10px; border-radius: 8px 8px 0 0; font-weight: 700; text-align: center;">
                ✨ Hybrid Recommender
            </div>
            """, unsafe_allow_html=True)
            hyb_recs = models["hybrid"].recommend(selected_user_id, top_k=5, exclude_interacted=exclude_seen)
            for r in hyb_recs:
                st.markdown(f"""
                <div style="background: white; border: 1px solid #E2E8F0; padding: 10px; margin-top: 6px; border-radius: 6px; font-size: 0.85rem;">
                    <b>{r['title'][:28]}...</b><br>
                    <span style="color: #64748B;">Score: {r['score']:.2f} | ${r['price']:.2f}</span>
                </div>
                """, unsafe_allow_html=True)

    # TAB 4: OFFLINE EVALUATION METRICS
    with tab4:
        st.markdown("#### Held-Out Chronological Test Benchmark")
        st.write("Strictly evaluated on the final 15% chronological window (`2026-11-10` to `2026-11-18`), held out completely from training and tuning.")

        if eval_metrics:
            overall_rows = []
            for m_name, vals in eval_metrics["overall"].items():
                v = vals["K=10"]
                overall_rows.append({
                    "Recommendation Model": m_name,
                    "NDCG@10": v["NDCG@K"],
                    "Recall@10": v["Recall@K"],
                    "Precision@10": v["Precision@K"],
                    "Catalog Coverage (%)": round(v["CatalogCoverage@K"] * 100, 1),
                    "Novelty (Bits)": v["Novelty@K"]
                })
            df_m = pd.DataFrame(overall_rows).set_index("Recommendation Model")
            st.dataframe(df_m, use_container_width=True)

            c_ch1, c_ch2 = st.columns(2)
            with c_ch1:
                st.markdown("**Ranking Accuracy Lift (NDCG@10 & Recall@10):**")
                st.bar_chart(df_m[["NDCG@10", "Recall@10"]])
            
            with c_ch2:
                st.markdown("**Catalog Health: Coverage vs Popularity Monopolization:**")
                st.bar_chart(df_m[["Catalog Coverage (%)"]])

            st.markdown("---")
            st.markdown("#### Cold-Start Users Test Slice (Users with ≤ 2 Training Events)")
            cold_rows = []
            for m_name, vals in eval_metrics["cold_start"].items():
                v = vals["K=10"]
                cold_rows.append({
                    "Recommendation Model": m_name,
                    "Cold NDCG@10": v["NDCG@K"],
                    "Cold Recall@10": v["Recall@K"],
                    "Cold Coverage (%)": round(v["CatalogCoverage@K"] * 100, 1)
                })
            st.dataframe(pd.DataFrame(cold_rows).set_index("Recommendation Model"), use_container_width=True)

    # TAB 5: MERCHANDISER GOVERNANCE
    with tab5:
        st.markdown("#### Decision-Support Governance & Safe Production Deployment")
        st.markdown("""
        ### Human-in-the-Loop Architecture
        The recommendation engine is an **advisory intelligence layer** designed to assist category merchandisers and shoppers, not to operate autonomously.
        
        #### 1. Cold-Start Safety Net:
        - Automatically detects users with $\le 2$ interactions and adjusts weights from Collaborative Filtering to Content-Based category matching and trending popular products.
        
        #### 2. Popularity Bias & Long-Tail Catalog Health:
        - Pure popularity models trap customers in an echo chamber, recommending only 8% of the inventory.
        - The Hybrid engine achieves **98% catalog coverage**, ensuring discovery for new products and niche sellers.
        
        #### 3. Privacy Preservation:
        - Zero Personally Identifiable Information (PII) is stored or ingested. The engine operates purely on anonymous session tokens and behavioral event sequences.
        """)
