import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from streamlit_option_menu import option_menu
from evaluation import MODEL_INFO
from analyzer import (
    get_sentiment,
    get_textblob_sentiment,
    get_lr_sentiment,
    get_keywords,
    analyze_bulk,
    load_lr_model,
    explain_prediction
)



st.set_page_config(
    page_title="SentiScan AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
            
    /* Bigger Tabs */
    button[data-baseweb="tab"] {
        font-size: 17px !important;
        font-weight: 600 !important;
        padding: 14px 22px !important;
        border-radius: 10px 10px 0 0;
    }

    /* Active Tab */
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #60A5FA !important;
        border-bottom: 3px solid #3B82F6 !important;
    }

    /* Tab List */
    div[data-baseweb="tab-list"] {
        gap: 15px;
    }        
    .app-header {
        background:#0f1f3d; padding:18px 24px; border-radius:12px;
        margin-bottom:20px; display:flex; justify-content:space-between; align-items:center;
    }
    .app-title  { color:white; font-size:20px; font-weight:600; margin:0; }
    .app-badge  { background:#1a3a6e; color:#8aaad4; border:1px solid #2d5a9e; padding:3px 12px; border-radius:20px; font-size:12px; }
    .result-positive { background:#f0fdf4; border-left:4px solid #22c55e; padding:16px 20px; border-radius:10px; margin:10px 0; }
    .result-negative { background:#fef2f2; border-left:4px solid #ef4444; padding:16px 20px; border-radius:10px; margin:10px 0; }
    .result-neutral  { background:#fffbeb; border-left:4px solid #f59e0b; padding:16px 20px; border-radius:10px; margin:10px 0; }
    .result-title { font-size:20px; font-weight:600; margin:0 0 4px 0; }
    .result-sub   { font-size:13px; margin:0; }
    .model-card{
        background:#161B22;
        border:1px solid #30363D;
        border-radius:18px;
        padding:22px;
        min-height:220px;
        text-align:center;
        transition:0.3s;
    }
    .model-card:hover{
        transform:translateY(-4px);
        border:1px solid #3B82F6;
    }
    .model-name{
        font-size:15px;
        color:#94A3B8;
        margin-bottom:10px;
        font-weight:700;
        text-transform:uppercase;
        letter-spacing:1px;
    }
    .model-conf  { font-size:15px; color:#CBD5E1; margin-top:8px; }
    .model-type  { font-size:12px;  background:#2563EB; color:white; border-radius:25px; padding:8px 14px; display:inline-block; margin-top:18px; }
    .agree-badge { background:#f0fdf4; color:#166534; border:1px solid #86efac; padding:4px 12px; border-radius:20px; font-size:12px; display:inline-block; margin-top:8px; }
    .disagree-badge { background:#fef2f2; color:#991b1b; border:1px solid #fca5a5; padding:4px 12px; border-radius:20px; font-size:12px; display:inline-block; margin-top:8px; }
    .kw-pos { display:inline-block; background:#f0fdf4; color:#166534; border:1px solid #86efac; padding:3px 10px; border-radius:20px; font-size:12px; margin:3px; }
    .kw-neg { display:inline-block; background:#fef2f2; color:#991b1b; border:1px solid #fca5a5; padding:3px 10px; border-radius:20px; font-size:12px; margin:3px; }
    .kw-neu { display:inline-block; background:#fffbeb; color:#92400e; border:1px solid #fcd34d; padding:3px 10px; border-radius:20px; font-size:12px; margin:3px; }
    .hist-row { background:#f8fafc; border:1px solid #e2e8f0; border-radius:8px; padding:10px 14px; margin:6px 0; display:flex; align-items:center; gap:10px; font-size:13px; }
    #MainMenu {visibility:hidden;} footer {visibility:hidden;} .stDeployButton {display:none;}
            
    .kpi-card{
    background:#1E293B;
    border:1px solid #30363D;
    border-radius:18px;
    padding:24px;
    text-align:center;
    transition:0.3s;
    height:180px;
    box-shadow:0 4px 20px rgba(37,99,235,.15);
    }

    .kpi-card:hover{
    transform:translateY(-5px);
    box-shadow:0 8px 30px rgba(37,99,235,.25);
    }

    .kpi-icon{
    font-size:34px;
    margin-bottom:10px;
    }

    .kpi-value{
    font-size:34px;
    font-weight:700;
    color:white;
    }

    .kpi-title{
    font-size:15px;
    color:#9CA3AF;
    margin-top:8px;
    }
            

    /* Primary Button */
    div[data-testid="stButton"] > button {
        width: 100%;
        height: 52px;
        background: ##3B82F6
        color: white;
        border: none;
        border-radius: 12px;
        font-size: 16px;
        font-weight: 600;
        transition: all 0.25s ease;
    }

    /* Hover Effect */
    div[data-testid="stButton"] > button:hover {
        background: #2563EB
        transform: translateY(-2px);
        box-shadow: 0px 8px 20px rgba(37,99,235,0.35);
    }

    /* Click Effect */
    div[data-testid="stButton"] > button:focus {
        border: none;
        box-shadow: 0 0 0 2px rgba(37,99,235,0.4);
    }
</style>
            
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────
if "history"  not in st.session_state: st.session_state.history  = []
if "lr_model" not in st.session_state:

    model, vectorizer = load_lr_model()

    st.session_state.lr_model = model
    st.session_state.vectorizer = vectorizer
if "model_trained" not in st.session_state: st.session_state.model_trained = st.session_state.lr_model is not None
with st.sidebar:
    st.markdown("## 🤖 SentiScan AI")
    st.markdown("---")

    st.success("System Ready")

    st.write("**Models**")
    st.write("• Logistic Regression")
    st.write("• VADER")
    st.write("• TextBlob")

    st.markdown("---")
    st.caption("Built by Jatin Kumar")
# ── Header ────────────────────────────────────────────────────
st.markdown("""
<div style="
background:linear-gradient(135deg,#1E3A8A,#2563EB);
border:1px solid #334155;
padding:35px;
border-radius:20px;
margin-bottom:25px;
">

<h1 style="
color:white;
margin:0;
font-size:42px;
">
🤖 SentiScan AI
</h1>

<p style="
color:#CBD5E1;
font-size:20px;
margin-top:12px;">
Enterprise Sentiment Analysis Platform
</p>

<hr style="border:1px solid #334155;">

<p style="
color:#CBD5E1;
font-size:16px;
line-height:1.8;">
✔ Logistic Regression (IMDb 50K)<br>
✔ VADER Rule-Based Analysis<br>
✔ TextBlob Polarity Analysis
</p>

</div>
""", unsafe_allow_html=True)
c1, c2, c3, c4 = st.columns(4)

cards = [
    ("🎯", "89.42%", "Model Accuracy"),
    ("📚", "IMDb 50K", "Training Dataset"),
    ("🤖", "3", "AI Models"),
    ("⚡", "<1 sec", "Prediction Time")
]

for col, (icon, value, title) in zip([c1, c2, c3, c4], cards):
    with col:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-icon">{icon}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-title">{title}</div>
        </div>
        """, unsafe_allow_html=True)
st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
selected = option_menu(
    menu_title=None,
    options=[
        "Review",
        "Compare",
        "Bulk",
        "History",
        "Evaluation"
    ],
    icons=[
        "search",
        "cpu",
        "folder",
        "clock-history",
        "bar-chart"
    ],
    orientation="horizontal",
    default_index=0,

    styles={
        "container": {
            "padding": "0!important",
            "margin": "0!important",
            "background-color": "#111827",
            "width": "100%",
        },

        "nav": {
            "display": "flex",
            "justify-content": "space-between",
            "width": "100%",
            "padding": "0",
            "margin": "0",
        },

        "nav-item": {
            "flex": "1",
            "text-align": "center",
        },

        "nav-link": {
            "text-align": "center",
            "font-size": "16px",
            "font-weight": "600",
            "padding": "14px 0",
            "--hover-color": "#1E293B",
        },

        "nav-link-selected": {
            "background-color": "#2563EB",
            "color": "white",
        },
    }
)


# ══════════════════════════════════════════════════════
# TAB 1 — SINGLE REVIEW
# ══════════════════════════════════════════════════════
if selected == "Review":
    st.markdown("""
    ## 📝 Review Input

    <span style="color:#9CA3AF;">
    Paste a product review below to analyze its sentiment using AI.
    </span>
    """, unsafe_allow_html=True)


    user_input = st.text_area("Review", height=220,
        placeholder='"Example:\n\nThe product quality exceeded my expectations. Delivery was fast and the packaging was excellent."',
        label_visibility="collapsed")
    
    st.markdown("<br>", unsafe_allow_html=True)
    analyze_clicked = st.button(
        "🚀 Analyze Review",
        use_container_width=True
    )
    
    ## if clear_clicked: st.rerun()

    if analyze_clicked:
        if not user_input.strip():
            st.warning("Please enter a review first!")
        else:
            with st.spinner("🤖 AI is analyzing..."):
                result = get_sentiment(user_input)
                keywords = get_keywords(user_input)
                st.session_state.history.insert(
                    0,
                    {
                        "text": user_input,
                        "label": result["label"],
                        "score": result["compound"]
                    }
                )

            st.markdown(f"""
            <div style="
            background:#161B22;
            border:1px solid #30363D;
            border-left:6px solid {result['color']};
            padding:25px;
            border-radius:18px;
            margin-top:10px;
            ">

            <h2 style="
            margin:0;
            color:{result['color']};
            font-size:34px;">
            {result['emoji']} {result['label']}
            </h2>

            <p style="
            margin-top:8px;
            color:#CBD5E1;
            font-size:18px;">
            AI Prediction Completed Successfully
            </p>

            <hr style="border:1px solid #30363D;">

            <h3 style="margin-bottom:5px;">
            Confidence Score
            </h3>

            <h1 style="
            margin-top:0;
            color:white;">
            {result['confidence']}%
            </h1>

            <p style="color:#94A3B8;">
            Compound Score : {result['compound']}
            </p>

            </div>
            """, unsafe_allow_html=True)

           ## st.markdown(f"**Sentiment strength: {result['confidence']}%**")
            st.progress(result["confidence"] / 100)
            st.divider()

            c1, c2, c3 = st.columns(3)
            c1.metric("😊 Positive", result["positive"])

            c2.metric("😠 Negative", result["negative"])

            c3.metric("😐 Neutral", result["neutral"])
            st.divider()

            st.subheader("📊 Sentiment Score Breakdown")
            fig, ax = plt.subplots(figsize=(3,2.5))
            bars = ax.bar(["Positive","Neutral","Negative"],
                          [result["positive"],result["neutral"],result["negative"]],
                          color=["#22c55e","#f59e0b","#ef4444"], edgecolor="white", width=0.5)
            for b, v in zip(bars, [result["positive"],result["neutral"],result["negative"]]):
                ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.01,
                        f"{v:.2f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
            ax.set_ylim(0,1.1); ax.set_ylabel("Score")
            ax.spines[["top","right"]].set_visible(False)
            ax.set_facecolor("#fafafa"); fig.patch.set_facecolor("#fafafa")
            st.pyplot(fig); plt.close()

            if keywords:
                st.divider()
                st.markdown("**🏷 Important Keywords*")
                st.markdown("".join(f'<span class="kw-{t}">{w}</span>' for w,t in keywords),
                            unsafe_allow_html=True)

            try:
                from wordcloud import WordCloud
                st.divider()
                st.markdown("**☁️ Word Cloud Visualization**")
                wc = WordCloud(width=600, height=100, background_color="white",
                               colormap="RdYlGn", max_words=30).generate(user_input)
                fig_wc, ax_wc = plt.subplots(figsize=(6,2))
                ax_wc.imshow(wc, interpolation="bilinear"); ax_wc.axis("off")
                st.pyplot(fig_wc); plt.close()
            except: pass


# ══════════════════════════════════════════════════════
# TAB 2 — MODEL COMPARISON
# ══════════════════════════════════════════════════════
elif selected == "Compare":
    st.markdown("""
    # ⚖️ AI Model Comparison

    <span style="color:#9CA3AF;">
    Compare predictions from three different sentiment analysis techniques.
    </span>
    """, unsafe_allow_html=True)
    st.caption("VADER (rule-based) vs TextBlob (statistical) vs Logistic Regression (ML)")

    comp_input = st.text_area("Paste a review to compare:", height=180,
        placeholder='"Amazing product! Highly recommended."',
        label_visibility="collapsed")

    # ── LR model status ───────────────────────────────
    if st.session_state.model_trained:
        st.success("✅ IMDb-trained Logistic Regression model loaded successfully", icon="🤖")
    else:
        st.warning("⚠️ LR model not trained yet. Upload a dataset in Bulk Upload tab to train it.", icon="🔧")

    if st.button("⚖️ Compare All Models", use_container_width=True, type="primary"):
        if not comp_input.strip():
            st.warning("Please enter a review first!")
        else:
            with st.spinner("Comparing AI models..."):
                vader_res = get_sentiment(comp_input)
                tb_res = get_textblob_sentiment(comp_input)
                lr_res = get_lr_sentiment(
                    comp_input,
                    st.session_state.lr_model,
                    st.session_state.vectorizer
                )

            st.divider()
            st.markdown("## 🤖 Model Predictions")

            c1, c2, c3 = st.columns(3)

            with c1:
                st.markdown(f"""
                <div class="model-card">
                    <div class="model-name">VADER</div>
                    <div class="model-label" style="color:{vader_res['color']}">{vader_res['label']}</div>
                    <div class="model-conf">Score: {vader_res['compound']}</div>
                    <div class="model-conf">Confidence: {vader_res['confidence']}%</div>
                    <div class="model-type">Rule-based NLP</div>
                </div>""", unsafe_allow_html=True)

            with c2:
                st.markdown(f"""
                <div class="model-card">
                    <div class="model-name">TextBlob</div>
                    <div class="model-label" style="color:{tb_res['color']}">{tb_res['label']}</div>
                    <div class="model-conf">Score: {tb_res['score']}</div>
                    <div class="model-conf">Confidence: {tb_res['confidence']}%</div>
                    <div class="model-type">Statistical NLP</div>
                </div>""", unsafe_allow_html=True)

            with c3:
                if lr_res:
                    st.markdown(f"""
                    <div class="model-card">
                        <div class="model-name">Logistic Regression</div>
                        <div class="model-label" style="color:{lr_res['color']}">{lr_res['label']}</div>
                        <div class="model-conf">Prediction Probability: {lr_res['confidence']}%</div>
                        <div class="model-type">Trained ML Model</div>
                    </div>""", unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="model-card">
                        <div class="model-name">Logistic Regression</div>
                        <div class="model-label" style="color:#94a3b8">Not trained</div>
                        <div class="model-conf">Upload a dataset to train</div>
                        <div class="model-type">Trained ML Model</div>
                    </div>""", unsafe_allow_html=True)

            st.divider()

            # ── Agreement check ───────────────────────
            available = [vader_res["label"], tb_res["label"]]
            if lr_res: available.append(lr_res["label"])
            all_agree = len(set(available)) == 1

            if all_agree:
                st.markdown(f'<div class="agree-badge">✅ All models agree — <b>{vader_res["label"]}</b></div>',
                            unsafe_allow_html=True)
            else:
                st.markdown('<div class="disagree-badge">⚠️ Models disagree — review may be ambiguous or sarcastic</div>',
                            unsafe_allow_html=True)

            st.divider()

            # ── Confidence bar chart ──────────────────
            st.markdown("**## 📊 Confidence Comparison**")
            model_names = ["VADER", "TextBlob"]
            conf_vals   = [vader_res["confidence"], tb_res["confidence"]]
            bar_colors  = [
                {"Positive":"#22c55e","Negative":"#ef4444","Neutral":"#f59e0b"}.get(vader_res["label"],"#888"),
                {"Positive":"#22c55e","Negative":"#ef4444","Neutral":"#f59e0b"}.get(tb_res["label"],"#888"),
            ]
            if lr_res:
                model_names.append("Logistic Reg.")
                conf_vals.append(lr_res["confidence"])
                bar_colors.append({"Positive":"#22c55e","Negative":"#ef4444","Neutral":"#f59e0b"}.get(lr_res["label"],"#888"))

            fig_c, ax_c = plt.subplots(figsize=(5,2.5))
            bars_c = ax_c.bar(model_names, conf_vals, color=bar_colors, edgecolor="white", width=0.4)
            for b, v in zip(bars_c, conf_vals):
                ax_c.text(b.get_x()+b.get_width()/2, b.get_height()+0.5,
                          f"{v}%", ha="center", va="bottom", fontsize=10, fontweight="bold")
            ax_c.set_ylim(0,115); ax_c.set_ylabel("Confidence %")
            ax_c.spines[["top","right"]].set_visible(False)
            ax_c.set_facecolor("#fafafa"); fig_c.patch.set_facecolor("#fafafa")
            st.pyplot(fig_c); plt.close()
    st.divider()

    st.subheader("🧠 Most Important Words Learned by the Model")
    
    positive_words, negative_words = explain_prediction(
        st.session_state.lr_model,
        st.session_state.vectorizer
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### ✅ Positive Words")
        for word, score in positive_words:
            st.success(f"{word} ({score})")

    with col2:
        st.markdown("### ❌ Negative Words")
        for word, score in negative_words:
            st.error(f"{word} ({score})")

            # ── Model explanation ─────────────────────
            with st.expander("📖 What's the difference between these models?"):
                st.markdown("""
| Model | How it works | Best for |
|---|---|---|
| **VADER** | Pre-built dictionary of words with sentiment scores | Short reviews, social media, emojis |
| **TextBlob** | Pattern-based + simple statistical rules | General text, quick baseline |
| **Logistic Regression** | Trained on your actual data using TF-IDF features | Domain-specific reviews, higher accuracy |

**When they disagree** — the review is likely sarcastic, context-dependent, or uses slang.
                """)
            
            
# ══════════════════════════════════════════════════════
# TAB 3 — BULK UPLOAD
# ══════════════════════════════════════════════════════
elif selected == "Bulk":

    def bulk_with_progress(reviews_list):
        CHUNK, all_parts = 500, []
        bar = st.progress(0, text="Starting…")
        for i in range(0, len(reviews_list), CHUNK):
            chunk = reviews_list[i:i+CHUNK]
            all_parts.append(analyze_bulk(chunk))
            done = min(i+CHUNK, len(reviews_list))
            bar.progress(int(done/len(reviews_list)*100),
                         text=f"Analyzed {done} / {len(reviews_list)} reviews…")
        bar.empty()
        return pd.concat(all_parts, ignore_index=True)

    def show_chart(df_res):
        counts = df_res["Sentiment"].value_counts()
        fig2, ax2 = plt.subplots(figsize=(5,2.5))
        cmap = {"Positive":"#22c55e","Negative":"#ef4444","Neutral":"#f59e0b"}
        ax2.bar(counts.index, counts.values,
                color=[cmap.get(l,"#888") for l in counts.index],
                edgecolor="white", width=0.5)
        ax2.set_title("Sentiment Distribution", fontsize=12); ax2.set_ylabel("Count")
        ax2.spines[["top","right"]].set_visible(False)
        ax2.set_facecolor("#fafafa"); fig2.patch.set_facecolor("#fafafa")
        st.pyplot(fig2); plt.close()

    # ── CSV Upload ────────────────────────────────────
    st.markdown("""
    # 📂 Bulk Sentiment Analysis

    <span style="color:#9CA3AF;">
    Analyze thousands of reviews in seconds.
    </span>
    """, unsafe_allow_html=True)
    
    st.caption("You will pick which column to analyze — works with any dataset!")
    uploaded = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed")

    if uploaded:
        df_upload = pd.read_csv(uploaded)
        st.success(f"File loaded — {len(df_upload)} rows, {len(df_upload.columns)} columns")

        st.markdown("**Preview:**")
        st.dataframe(df_upload.head(3), use_container_width=True)

        # ── Column selector ───────────────────────────
        all_cols   = df_upload.columns.tolist()
        text_cols  = [c for c in all_cols if df_upload[c].dtype == object]

        # fallback — if no object columns found, show all columns
        if not text_cols:
            text_cols = all_cols

        st.markdown("**Select the review/text column:**")
        selected_col = st.selectbox("Text column", text_cols, label_visibility="collapsed")

        if selected_col is None:
            st.error("No columns found in your CSV. Please check the file.")
            st.stop()

        # Optional label column for LR training
        st.markdown("**Select label column (optional — for training LR model):**")
        label_options = ["None"] + all_cols
        selected_label = st.selectbox("Label column", label_options, label_visibility="collapsed")

        col_a, col_b = st.columns(2)

        with col_a:
            if st.button("📊 Analyze Reviews", use_container_width=True, type="primary"):
                reviews_list = df_upload[selected_col].dropna().astype(str).tolist()
                if len(reviews_list) > 5000:
                    st.warning(f"Large file! Analyzing first 5,000 of {len(reviews_list)} rows.")
                    reviews_list = reviews_list[:5000]
                st.info(f"Analyzing **{len(reviews_list)}** reviews…")
                with st.spinner("Analyzing reviews..."):
                    df_results = bulk_with_progress(reviews_list)
                st.success(f"Done! Analyzed {len(df_results)} reviews.")

                positive = (df_results["Sentiment"] == "Positive").sum()
                negative = (df_results["Sentiment"] == "Negative").sum()
                neutral = (df_results["Sentiment"] == "Neutral").sum()
                total = len(df_results)

                col1, col2, col3, col4 = st.columns(4)

                col1.metric("Total Reviews", total)
                col2.metric("😊 Positive", positive)
                col3.metric("😐 Neutral", neutral)
                col4.metric("😠 Negative", negative)

                st.progress(positive / total)
                st.caption(f"{round((positive / total) * 100, 1)}% Positive Reviews")

                st.dataframe(df_results, use_container_width=True)
                show_chart(df_results)
                st.download_button("⬇️ Download Results",
                    data=df_results.to_csv(index=False).encode("utf-8"),
                    file_name="sentiscan_results.csv", mime="text/csv",
                    use_container_width=True)

        with col_b:
            st.success("✅ Pre-trained IMDb Logistic Regression model loaded.")

    st.divider()

    # ── Paste option ──────────────────────────────────
    st.markdown("#### Or paste multiple reviews")
    st.caption("One review per line")
    bulk_text = st.text_area("Paste reviews", height=160,
        placeholder="This product is amazing!\nTerrible quality.\nIt is okay, nothing special.",
        label_visibility="collapsed")

    if st.button("Analyze All", use_container_width=True, type="primary"):
        if not bulk_text.strip():
            st.warning("Please paste some reviews!")
        else:
            reviews_list = [r.strip() for r in bulk_text.strip().split("\n") if r.strip()]
            st.info(f"Analyzing **{len(reviews_list)}** reviews…")
            with st.spinner("Analyzing reviews..."):
                df_results = bulk_with_progress(reviews_list)
            st.success("✅ Analysis completed successfully.")
            positive = (df_results["Sentiment"] == "Positive").sum()
            negative = (df_results["Sentiment"] == "Negative").sum()
            neutral = (df_results["Sentiment"] == "Neutral").sum()

            total = len(df_results)

            col1, col2, col3, col4 = st.columns(4)

            col1.metric("Total Reviews", total)
            col2.metric("😊 Positive", positive)
            col3.metric("😐 Neutral", neutral)
            col4.metric("😠 Negative", negative)

            st.progress(positive / total)
            st.caption(f"{round((positive / total) * 100,1)}% Positive Reviews")
            st.subheader("📄 Analysis Results")
            st.dataframe(df_results, use_container_width=True)
            st.subheader("📊 Sentiment Distribution")
            show_chart(df_results)
            st.subheader("📥 Export Results")
            st.download_button("⬇️ Download Results",
                data=df_results.to_csv(index=False).encode("utf-8"),
                file_name="sentiscan_results.csv", mime="text/csv",
                use_container_width=True)


# ══════════════════════════════════════════════════════
# TAB 4 — HISTORY
# ══════════════════════════════════════════════════════
elif selected == "History":
    st.markdown("""
# 🕒 Prediction History

<span style="color:#9CA3AF;">
All reviews analyzed during this session.
</span>
""", unsafe_allow_html=True)
    if not st.session_state.history:
        st.info("No analyses yet. Go to Single Review and analyze something!")
    else:
        st.markdown(f"**{len(st.session_state.history)} reviews analyzed this session**")
        st.divider()
        cmap = {"Positive":"#22c55e","Negative":"#ef4444","Neutral":"#f59e0b"}
        bgmap= {"Positive":"#f0fdf4","Negative":"#fef2f2","Neutral":"#fffbeb"}
        for item in st.session_state.history:
            color = cmap.get(item["label"],"#888")
            bg    = bgmap.get(item["label"],"#f8fafc")
            short = item["text"][:70]+"..." if len(item["text"])>70 else item["text"]
            st.markdown(f"""
            <div class="hist-row" style="background:{bg}; border-left:3px solid {color}">
                <span style="font-weight:600;color:{color};min-width:70px">{item['label']}</span>
                <span style="flex:1;color:#334155">{short}</span>
                <span style="color:#94a3b8;font-size:12px">{item['score']}</span>
            </div>""", unsafe_allow_html=True)

        st.divider()
        counts_h = Counter(i["label"] for i in st.session_state.history)
        fig_h, ax_h = plt.subplots(figsize=(4,3))
        pie_c = {"Positive":"#22c55e","Negative":"#ef4444","Neutral":"#f59e0b"}
        ax_h.pie(counts_h.values(), labels=counts_h.keys(),
                 colors=[pie_c.get(l,"#888") for l in counts_h.keys()],
                 autopct="%1.0f%%", startangle=90, textprops={"fontsize":11})
        ax_h.set_title("Session Summary", fontsize=12)
        st.pyplot(fig_h); plt.close()

        if st.button("🗑️ Clear History", use_container_width=True):
            st.session_state.history = []; st.rerun()
            
elif selected == "Evaluation":

    st.markdown("""
    # 📊 Model Evaluation Dashboard

    <span style="color:#9CA3AF;">
    Performance of the trained Logistic Regression classifier.
    </span>
    """, unsafe_allow_html=True)

    st.info(f"""
    ### Model Information

    **Dataset:** {MODEL_INFO["Dataset"]}

    **Algorithm:** {MODEL_INFO["Algorithm"]}

    **Vectorizer:** {MODEL_INFO["Vectorizer"]}

    **Training Samples:** {MODEL_INFO["Training Samples"]}

    **Testing Samples:** {MODEL_INFO["Testing Samples"]}

    **Vocabulary Size:** {MODEL_INFO["Vocabulary Size"]}
    """)

    st.subheader("📈 Model Performance")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Accuracy", MODEL_INFO["Accuracy"])
    col2.metric("Precision", MODEL_INFO["Precision"])
    col3.metric("Recall", MODEL_INFO["Recall"])
    col4.metric("F1 Score", MODEL_INFO["F1 Score"])

    col1, col2 = st.columns(2)

    

    st.subheader("🎯 Confusion Matrix")

    cm = MODEL_INFO["confusion_matrix"]

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4,4))

    ax.imshow(cm)

    ax.set_xticks([0,1])
    ax.set_yticks([0,1])

    ax.set_xticklabels(["Negative","Positive"])
    ax.set_yticklabels(["Negative","Positive"])

    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i][j],
                    ha="center",
                    va="center",
                    fontsize=12,
                    fontweight="bold")

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    st.pyplot(fig)

st.divider()
##st.caption("Built with Python · VADER · TextBlob · Logistic Regression · Streamlit | Jatin Kumar · CSE AI/ML")
st.markdown("""
---
<div style="text-align:center;color:#94A3B8">

### 🤖 SentiScan AI

Built with Python • Streamlit • Scikit-learn • NLP

Developed by <b>Jatin Kumar</b>

</div>
""", unsafe_allow_html=True)