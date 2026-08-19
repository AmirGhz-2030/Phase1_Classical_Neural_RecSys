"""
Streamlit Interactive Dashboard for Phase 1 Recommender Systems
===============================================================
Features:
1. Executive Overview & Dataset KPI Metrics (MovieLens 20M).
2. Interactive Benchmark Leaderboard & Plotly Comparison Charts (All-Item vs Sampled-100).
3. Live Recommendation Playground (Select User -> View History -> Compare Model Recommendations).
4. Academic Reference & Methodology Guide for Phase 1.

Developer: AmirGhz-2030 (https://github.com/AmirGhz-2030)
Project: Phase 1 - Classical & Neural Recommender Systems
"""

import os
import sys
import pickle
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Ensure project root is in sys.path for robust module imports
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

# 1. Page Configuration
st.set_page_config(
    page_title="Phase 1: RecSys Benchmarking Studio",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)


# 2. Cached Data Loaders
@st.cache_data
def load_metadata():
    meta_path = os.path.join(PROCESSED_DIR, "meta.pkl")
    if not os.path.exists(meta_path):
        return None
    with open(meta_path, "rb") as f:
        return pickle.load(f)


@st.cache_data
def load_benchmark_results():
    all_item_path = os.path.join(PROCESSED_DIR, "classical_benchmark_results.csv")
    sampled_path = os.path.join(PROCESSED_DIR, "sampled100_benchmark_results.csv")
    full_path = os.path.join(PROCESSED_DIR, "full_phase1_benchmark_results.csv")

    df_all = pd.read_csv(full_path if os.path.exists(full_path) else all_item_path, index_col=0) if os.path.exists(all_item_path) else None
    df_sampled = pd.read_csv(sampled_path, index_col=0) if os.path.exists(sampled_path) else None
    return df_all, df_sampled


@st.cache_data
def load_movies_metadata():
    movies_path = os.path.join(PROCESSED_DIR, "movies_clean.csv")
    if os.path.exists(movies_path):
        return pd.read_csv(movies_path)
    return None


@st.cache_data
def load_sample_user_histories():
    train_path = os.path.join(PROCESSED_DIR, "train.csv")
    test_path = os.path.join(PROCESSED_DIR, "test.csv")
    if not (os.path.exists(train_path) and os.path.exists(test_path)):
        return None, None

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df


@st.cache_resource
def load_trained_models(n_users: int, n_items: int):
    """Initializes and fits light classical models for live playground interactions."""
    from src.models.classical import PopularityRecommender, MatrixFactorizationRecommender, ItemKNNRecommender
    train_path = os.path.join(PROCESSED_DIR, "train.csv")
    train_df = pd.read_csv(train_path)

    pop_model = PopularityRecommender()
    pop_model.fit(train_df, n_users=n_users, n_items=n_items)

    svd_model = MatrixFactorizationRecommender(n_factors=64)
    svd_model.fit(train_df, n_users=n_users, n_items=n_items)

    knn_model = ItemKNNRecommender()
    knn_model.fit(train_df, n_users=n_users, n_items=n_items)

    return {"Matrix Factorization (SVD)": svd_model, "Item-KNN (CF)": knn_model, "Popularity": pop_model}


# 3. Sidebar Navigation
st.sidebar.title("🎬 RecSys Studio")
st.sidebar.markdown("**Phase 1 Benchmarking**")
st.sidebar.markdown("*Classical & Neural RecSys*")
st.sidebar.markdown("---")

nav_choice = st.sidebar.radio(
    "Navigation Menu",
    [
        "📊 Benchmark & Leaderboard",
        "🎮 Live Recommendation Playground",
        "📈 Dataset & Sparsity Analytics",
        "📖 Academic & Methodology Guide"
    ]
)

st.sidebar.markdown("---")
st.sidebar.info(
    "**Project:** Phase 1 RecSys\n\n"
    "**GitHub:** [@AmirGhz-2030](https://github.com/AmirGhz-2030)\n\n"
    "**Benchmark:** MovieLens 20M"
)

# 4. App Pages

meta = load_metadata()
df_all_item, df_sampled = load_benchmark_results()
movies_df = load_movies_metadata()

# ==========================================
# PAGE 1: BENCHMARK & LEADERBOARD
# ==========================================
if nav_choice == "📊 Benchmark & Leaderboard":
    st.title("🏆 Phase 1: Recommender Systems Benchmark Leaderboard")
    st.markdown(
        "Evaluation of Classical & Neural recommendation models on **MovieLens 20M** "
        "using standard academic ranking protocols."
    )

    # KPI Top Cards
    if meta:
        stats = meta["stats"]
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Active Users", f"{stats['n_users']:,}")
        col2.metric("Catalog Movies", f"{stats['n_items']:,}")
        col3.metric("Total Ratings", f"{stats['n_interactions']:,}")
        col4.metric("Matrix Sparsity", f"{stats['sparsity_pct']:.2f}%")

    st.markdown("---")

    tab1, tab2 = st.tabs(["🎯 Sampled 100-Item Protocol (NCF 2017)", "🌐 All-Item Full Ranking Protocol"])

    # Tab 1: Sampled 100
    with tab1:
        st.subheader("1. Sampled 100 Candidates Evaluation Protocol")
        st.caption(
            "For each user: 1 ground-truth target item + 99 unobserved negative items (Standard protocol of He et al., WWW 2017)."
        )

        if df_sampled is not None:
            col_tbl, col_chart = st.columns([1, 1.2])

            with col_tbl:
                st.markdown("##### 📋 Leaderboard Table")
                display_df = df_sampled.copy()
                display_df["HR@5 (%)"] = (display_df["HR@5"] * 100).round(2)
                display_df["HR@10 (%)"] = (display_df["HR@10"] * 100).round(2)
                display_df["HR@20 (%)"] = (display_df["HR@20"] * 100).round(2)
                display_df["NDCG@10"] = display_df["NDCG@10"].round(4)
                display_df["MRR@10"] = display_df["MRR@10"].round(4)

                st.dataframe(
                    display_df[["HR@5 (%)", "HR@10 (%)", "HR@20 (%)", "NDCG@10", "MRR@10"]],
                    use_container_width=True
                )

                best_model = df_sampled["HR@10"].idxmax()
                st.success(f"🌟 **Top Performer:** `{best_model}` achieved **{df_sampled.loc[best_model, 'HR@10']*100:.2f}% HR@10**!")

            with col_chart:
                st.markdown("##### 📊 Metric Comparison (HR@10 vs NDCG@10)")
                plot_df = df_sampled.reset_index().rename(columns={"index": "Model"})
                fig = px.bar(
                    plot_df,
                    x="Model",
                    y=["HR@10", "NDCG@10", "MRR@10"],
                    barmode="group",
                    text_auto=".3f",
                    title="Sampled 100 Ranking Performance Across Models",
                    color_discrete_sequence=["#1f77b4", "#ff7f0e", "#2ca02c"]
                )
                fig.update_layout(yaxis_title="Score (0 to 1)", legend_title_text="Metric")
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Sampled benchmark results CSV not found.")

    # Tab 2: All Item Ranking
    with tab2:
        st.subheader("2. All-Item Full Space Ranking Protocol (13,130 Candidates)")
        st.caption(
            "Rigorous unconstrained evaluation across the entire catalog (KDD 2020 recommended standard)."
        )

        if df_all_item is not None:
            col_tbl2, col_chart2 = st.columns([1, 1.2])

            with col_tbl2:
                st.markdown("##### 📋 Full-Ranking Leaderboard Table")
                disp_all = df_all_item.copy()
                disp_all["HR@5 (%)"] = (disp_all["HR@5"] * 100).round(2)
                disp_all["HR@10 (%)"] = (disp_all["HR@10"] * 100).round(2)
                disp_all["HR@20 (%)"] = (disp_all["HR@20"] * 100).round(2)
                disp_all["NDCG@10"] = disp_all["NDCG@10"].round(4)
                disp_all["Fit Time (s)"] = disp_all["Fit_Time_s"]

                st.dataframe(
                    disp_all[["HR@5 (%)", "HR@10 (%)", "HR@20 (%)", "NDCG@10", "Fit Time (s)"]],
                    use_container_width=True
                )

                st.info("💡 **Note:** Random baseline in full space is 0.076%. SVD achieves 9.06% (119x over random).")

            with col_chart2:
                st.markdown("##### 📊 Top-K Hit Rate Progression")
                fig2 = go.Figure()
                for model_name in df_all_item.index:
                    fig2.add_trace(go.Scatter(
                        x=[5, 10, 20],
                        y=[df_all_item.loc[model_name, "HR@5"] * 100,
                           df_all_item.loc[model_name, "HR@10"] * 100,
                           df_all_item.loc[model_name, "HR@20"] * 100],
                        mode="lines+markers",
                        name=model_name,
                        line=dict(width=3)
                    ))
                fig2.update_layout(
                    title="Hit Rate @ K Growth (K = 5, 10, 20)",
                    xaxis_title="K (Cut-off Rank)",
                    yaxis_title="Hit Rate (%)",
                    hovermode="x unified"
                )
                st.plotly_chart(fig2, use_container_width=True)


# ==========================================
# PAGE 2: LIVE RECOMMENDATION PLAYGROUND
# ==========================================
elif nav_choice == "🎮 Live Recommendation Playground":
    st.title("🎮 Live Recommendation Playground")
    st.markdown("Select an actual user from the MovieLens test set to inspect their watched history and compare real-time model recommendations.")

    if meta and movies_df is not None:
        train_df, test_df = load_sample_user_histories()
        models = load_trained_models(meta["stats"]["n_users"], meta["stats"]["n_items"])

        # Movie metadata dictionary
        movie_dict = movies_df.set_index("item_idx").to_dict(orient="index")

        col_ctrl1, col_ctrl2 = st.columns([1, 2])
        with col_ctrl1:
            selected_user = st.number_input(
                "Select User Index (0 to 138,407):",
                min_value=0,
                max_value=meta["stats"]["n_users"] - 1,
                value=42,
                step=1
            )
            top_k_rec = st.slider("Number of Recommendations (Top-K):", min_value=3, max_value=20, value=5)

        # Retrieve User History
        user_history_items = train_df[train_df["user_idx"] == selected_user]["item_idx"].tolist()
        test_target_item = test_df[test_df["user_idx"] == selected_user]["item_idx"].values

        with col_ctrl2:
            st.markdown(f"#### 👤 User Profile: `User #{selected_user}`")
            st.markdown(f"- **Total Movies Rated in Training:** `{len(user_history_items)} movies`")
            if len(test_target_item) > 0:
                target_idx = test_target_item[0]
                target_info = movie_dict.get(target_idx, {"title": "Unknown", "genres": "N/A"})
                st.markdown(f"- **🎯 Ground-Truth Test Target (Held-out):** **{target_info['title']}** `({target_info['genres']})`")

        st.markdown("---")

        # Display History & Live Model Comparison
        col_hist, col_recs = st.columns([1, 2])

        with col_hist:
            st.markdown("##### 📜 Recent Watch History")
            history_display = []
            for it in user_history_items[-8:]:  # last 8 items
                info = movie_dict.get(it, {"title": f"Movie #{it}", "genres": "N/A"})
                history_display.append({"Title": info["title"], "Genres": info["genres"]})
            st.dataframe(pd.DataFrame(history_display), use_container_width=True, height=350)

        with col_recs:
            st.markdown("##### 🔮 Real-Time Model Recommendations")
            tab_svd, tab_knn, tab_pop = st.tabs(["Matrix Factorization (SVD)", "Item-KNN (CF)", "Popularity"])

            for tab, (m_name, m_obj) in zip([tab_svd, tab_knn, tab_pop], models.items()):
                with tab:
                    recs = m_obj.recommend(selected_user, top_k=top_k_rec, filter_history=True)
                    rec_rows = []
                    for rank, it in enumerate(recs, start=1):
                        info = movie_dict.get(it, {"title": f"Movie #{it}", "genres": "N/A"})
                        is_hit = "🎯 HIT!" if (len(test_target_item) > 0 and it == test_target_item[0]) else ""
                        rec_rows.append({
                            "Rank": f"#{rank}",
                            "Movie Title": info["title"],
                            "Genres": info["genres"],
                            "Status": is_hit
                        })
                    st.dataframe(pd.DataFrame(rec_rows), use_container_width=True, height=280)


# ==========================================
# PAGE 3: DATASET & SPARSITY ANALYTICS
# ==========================================
elif nav_choice == "📈 Dataset & Sparsity Analytics":
    st.title("📈 MovieLens 20M Dataset Analytics")
    st.markdown("Deep dive into user interaction distribution, movie popularity power-laws, and temporal dynamics.")

    if meta and movies_df is not None:
        stats = meta["stats"]

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("##### 🎭 Genre Distribution in Filtered Catalog")
            all_genres = []
            for g_str in movies_df["genres"].dropna():
                all_genres.extend(g_str.split("|"))
            genre_counts = pd.Series(all_genres).value_counts().reset_index()
            genre_counts.columns = ["Genre", "Count"]

            fig_genre = px.pie(
                genre_counts.head(10),
                names="Genre",
                values="Count",
                title="Top 10 Movie Genres in Catalog",
                hole=0.4
            )
            st.plotly_chart(fig_genre, use_container_width=True)

        with col2:
            st.markdown("##### ⚡ Interaction Matrix Properties")
            matrix_data = pd.DataFrame({
                "Property": ["Total Possible Cells", "Observed Interactions", "Empty/Sparse Cells", "Matrix Sparsity Rate"],
                "Value": [
                    f"{stats['n_users'] * stats['n_items']:,}",
                    f"{stats['n_interactions']:,}",
                    f"{(stats['n_users'] * stats['n_items']) - stats['n_interactions']:,}",
                    f"{stats['sparsity_pct']:.4f}%"
                ]
            })
            st.table(matrix_data)
            st.info("K-Core threshold = 20 guarantees every user and movie has at least 20 ratings, removing severe cold-start noise.")


# ==========================================
# PAGE 4: ACADEMIC GUIDE
# ==========================================
elif nav_choice == "📖 Academic & Methodology Guide":
    st.title("📖 Academic Reference & Methodology Guide")
    st.markdown("### Recommender Systems — Phase 1 Technical Overview")

    st.markdown("""
    #### 1. Theoretical Foundations
    - **Collaborative Filtering (CF):** Assumes users who agreed in the past will agree in the future.
    - **Matrix Factorization (SVD / Funk-SVD):** Projects both users and items into a joint latent factor space of dimensionality $d=64$:
      $$\hat{y}_{ui} = \mathbf{u}_u^T \mathbf{v}_i$$
    - **Neural Collaborative Filtering (NCF / NeuMF):** Combines linear GMF (element-wise product) with non-linear MLP layers (ReLU activation) to capture complex higher-order interactions.

    #### 2. Evaluation Metrics Formulas
    - **Hit Rate ($HR@K$):**
      $$HR@K = \\frac{1}{|U|} \sum_{u \in U} \mathbb{I}(\text{target}_u \in \\text{Top-}K_u)$$
    - **Normalized Discounted Cumulative Gain ($NDCG@K$):**
      $$NDCG@K = \\frac{DCG@K}{IDCG@K}, \quad DCG@K = \sum_{i=1}^K \\frac{2^{rel_i} - 1}{\log_2(i + 1)}$$
    - **Mean Reciprocal Rank ($MRR@K$):**
      $$MRR = \\frac{1}{|U|} \sum_{u \in U} \\frac{1}{\\text{rank}_u}$$
    """)

st.markdown("---")
st.caption("Phase 1 Recommender Systems Studio | Maintained by [@AmirGhz-2030](https://github.com/AmirGhz-2030)")
