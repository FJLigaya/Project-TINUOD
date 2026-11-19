from pathlib import Path

import joblib
import re
import pandas as pd
import streamlit as st

# ==========================================
# PROJECT TINUOD
# AI-Assisted Misinformation Detection
# ==========================================

PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_DIR / "models" / "tinuod_model_v2.joblib"
CHARTS_DIR = PROJECT_DIR / "charts"
RESULTS_DIR = PROJECT_DIR / "results"
SUMMARY_PATH = RESULTS_DIR / "mpi_performance_summary.csv"

st.set_page_config(
    page_title="TINUOD | AI News Analyzer",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==========================================
# CUSTOM STYLING
# ==========================================

st.markdown("""
<style>
    .block-container {
        max-width: 1250px;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    h1, h2, h3 {
        letter-spacing: -0.5px;
    }

    div[data-testid="stMetric"] {
        background: rgba(128, 128, 128, 0.07);
        border: 1px solid rgba(128, 128, 128, 0.18);
        border-radius: 12px;
        padding: 16px;
    }

    div[data-testid="stMetricValue"] {
        font-size: 1.7rem;
    }

    .section-note {
        color: #9CA3AF;
        font-size: 0.9rem;
        margin-bottom: 1rem;
    }

    .app-footer {
        text-align: center;
        color: #9CA3AF;
        font-size: 0.8rem;
        padding: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


# ==========================================
# LOAD RESOURCES
# ==========================================

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_benchmark_summary():
    return pd.read_csv(SUMMARY_PATH)


try:
    model = load_model()
except Exception as error:
    st.error(f"Unable to load model: {error}")
    st.stop()


# ==========================================
# HELPER FUNCTIONS
# ==========================================

def show_chart(title, filename, description):
    """Display a compact chart inside its column."""

    st.markdown(f"**{title}**")

    chart_path = CHARTS_DIR / filename

    if chart_path.exists():
        st.image(
            str(chart_path),
            width="stretch"
        )
    else:
        st.warning(f"Chart unavailable: {filename}")

    st.caption(description)


def clean_source_metadata(text):
    """Strip leading source attribution; do not alter the article body."""
    text = str(text).strip()
    text = re.sub(
        r"^\s*[A-Za-z][A-Za-z\s,.'-]{0,100}\s*\(\s*Reuters\s*\)\s*[-–—:]\s*",
        "", text, flags=re.IGNORECASE,
    )
    text = re.sub(
        r"^\s*(?:Reuters|Associated Press|AP)\s*[-–—:]\s*",
        "", text, flags=re.IGNORECASE,
    )
    return re.sub(r"\s+", " ", text).strip()


def analyze_article(text):
    """Apply V2 input cleaning and return predicted class probabilities."""
    cleaned_text = clean_source_metadata(text)
    if not cleaned_text:
        raise ValueError("No article content remains after preprocessing.")
    prediction = int(model.predict([cleaned_text])[0])
    probabilities = model.predict_proba([cleaned_text])[0]

    probability_map = {
        int(label): float(probability)
        for label, probability in zip(
            model.classes_, probabilities
        )
    }

    # Original dataset mapping:
    # 0 = Fake, 1 = Real

    return (
        prediction,
        probability_map[0],
        probability_map[1]
    )


# ==========================================
# APPLICATION HEADER
# ==========================================

st.title("🛡️ Project TINUOD")

st.markdown("### AI-Assisted Misinformation Detection")

st.write(
    "An academic machine learning and parallel computing "
    "prototype for exploring news classification, "
    "model behavior, and MPI performance."
)

st.divider()


# ==========================================
# NAVIGATION
# ==========================================

tab1, tab2, tab3 = st.tabs([
    "📰 News Analyzer",
    "📊 MPI Performance",
    "📘 About TINUOD"
])


# ==========================================
# TAB 1 - NEWS ANALYZER
# ==========================================

with tab1:

    st.header("News Article Analyzer")

    st.write(
        "Paste a news article to examine the language "
        "patterns recognized by the trained model."
    )

    article = st.text_area(
        "Article Content",
        placeholder="Paste your news article here...",
        height=180,
        key="article_input"
    )

    col_button, col_space = st.columns([1, 4])

    with col_button:
        analyze_clicked = st.button(
            "🔎 Analyze Article",
            type="primary",
            use_container_width=True
        )

    if analyze_clicked:

        if not article.strip():
            st.warning("Please enter an article first.")

        else:

            try:
                with st.spinner("Analyzing article..."):
                    prediction, fake_prob, real_prob = analyze_article(article)
                st.session_state["analysis_result"] = {
                    "article": article,
                    "prediction": prediction,
                    "fake_probability": fake_prob,
                    "real_probability": real_prob
                }
            except Exception as error:
                st.error(f"Unable to analyze this article: {error}")
                st.session_state.pop("analysis_result", None)

    # Display the last completed analysis
    if "analysis_result" in st.session_state:

        result = st.session_state["analysis_result"]

        if result["article"] != article:
            st.info(
                "The article has changed. Click Analyze Article "
                "to generate an updated result."
            )

        else:

            prediction = result["prediction"]
            fake_prob = result["fake_probability"]
            real_prob = result["real_probability"]

            predicted_category = (
                "Fake" if prediction == 0 else "Real"
            )

            st.divider()
            st.subheader("Analysis Result")

            if prediction == 0:
                st.error("⚠️ LIKELY FAKE")
            else:
                st.success("✅ LIKELY REAL")

            st.caption(
                "This is a model classification, "
                "not an independently verified fact-check."
            )

            st.subheader("Model Confidence")

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "Fake Class Probability",
                    f"{fake_prob * 100:.2f}%"
                )

            with col2:
                st.metric(
                    "Real Class Probability",
                    f"{real_prob * 100:.2f}%"
                )

            st.markdown("**Probability Distribution**")

            st.write(f"Fake — {fake_prob * 100:.2f}%")
            st.progress(fake_prob)

            st.write(f"Real — {real_prob * 100:.2f}%")
            st.progress(real_prob)

            with st.expander(
                "💡 Understanding this result",
                expanded=False
            ):

                st.write(
                    "The model identified language patterns "
                    "that are more strongly associated with "
                    f"the **{predicted_category}** category "
                    "in its training dataset."
                )

                st.info(
                    "These percentages are estimated model "
                    "classification probabilities. They do not "
                    "measure the actual factual truthfulness "
                    "of the submitted article."
                )

                st.warning(
                    "TINUOD does not verify claims against "
                    "external sources. Important information "
                    "should always be checked using "
                    "reliable references."
                )


# ==========================================
# TAB 2 - MPI PERFORMANCE
# ==========================================

with tab2:

    st.header("MPI Performance Dashboard")

    st.write(
        "Explore the performance of Project TINUOD "
        "using different MPI process configurations."
    )

    st.caption(
        "Original MPI benchmark (not rerun with V2) | "
        "Workload: 7,731 articles | 1, 2, and 4 processes"
    )

    if SUMMARY_PATH.exists():

        summary = load_benchmark_summary()

        required_columns = {
            "processes",
            "median",
            "speedup",
            "efficiency"
        }

        if required_columns.issubset(summary.columns):

            summary = summary.sort_values("processes")

            fastest = summary.loc[
                summary["median"].idxmin()
            ]

            baseline = summary.loc[
                summary["processes"] == 1
            ]

            st.subheader("Performance Overview")

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric(
                    "Articles Tested",
                    "7,731"
                )

            with col2:
                st.metric(
                    "Fastest Configuration",
                    f"{int(fastest['processes'])} Processes"
                )

            with col3:
                st.metric(
                    "Best Median Time",
                    f"{fastest['median']:.4f} s"
                )

            with col4:
                st.metric(
                    "Measured Speedup",
                    f"{fastest['speedup']:.2f}x"
                )

            st.divider()

            # ==================================
            # BENCHMARK TABLE
            # ==================================

            st.subheader("Benchmark Summary")

            display_summary = summary.copy()

            display_summary = display_summary.rename(
                columns={
                    "processes": "Processes",
                    "mean": "Mean (s)",
                    "median": "Median (s)",
                    "minimum": "Min (s)",
                    "maximum": "Max (s)",
                    "std_dev": "Std. Dev.",
                    "speedup": "Speedup",
                    "efficiency": "Efficiency (%)"
                }
            )

            st.dataframe(
                display_summary,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Processes": st.column_config.NumberColumn(
                        format="%d"
                    ),
                    "Efficiency (%)":
                        st.column_config.NumberColumn(
                            format="%.2f%%"
                        ),
                    "Speedup":
                        st.column_config.NumberColumn(
                            format="%.2fx"
                        )
                }
            )

            # ==================================
            # COMPACT VISUALIZATIONS
            # ==================================

            st.divider()

            st.subheader("Performance Visualizations")

            st.caption(
                "Compare execution time, speedup, "
                "efficiency, and benchmark variability."
            )

            col1, col2 = st.columns(
                2, gap="medium"
            )

            with col1:

                with st.container(border=True):
                    show_chart(
                        "⏱️ Execution Time",
                        "01_execution_time.png",
                        "Median classification execution time "
                        "for each MPI configuration."
                    )

                with st.container(border=True):
                    show_chart(
                        "⚙️ Parallel Efficiency",
                        "03_parallel_efficiency.png",
                        "Efficiency of the additional "
                        "MPI processes."
                    )

            with col2:

                with st.container(border=True):
                    show_chart(
                        "🚀 MPI Speedup",
                        "02_speedup.png",
                        "Measured acceleration compared "
                        "with the single-process baseline."
                    )

                with st.container(border=True):
                    show_chart(
                        "📦 Benchmark Variability",
                        "04_benchmark_variability.png",
                        "Variation across repeated "
                        "benchmark trials."
                    )

            # ==================================
            # PERFORMANCE INTERPRETATION
            # ==================================

            st.divider()

            st.subheader("Performance Interpretation")

            if not baseline.empty:

                baseline_time = float(
                    baseline.iloc[0]["median"]
                )

                fastest_time = float(
                    fastest["median"]
                )

                reduction = (
                    (baseline_time - fastest_time)
                    / baseline_time * 100
                )

                st.success(
                    f"The {int(fastest['processes'])}-process "
                    "configuration achieved the lowest "
                    "median execution time, reducing it "
                    f"by approximately {reduction:.1f}% "
                    "compared with the single-process baseline."
                )

            st.info(
                "Speedup is not necessarily proportional "
                "to process count. Communication overhead, "
                "workload distribution, and available "
                "hardware can affect parallel efficiency."
            )

        else:
            st.warning(
                "The benchmark summary is missing "
                "one or more required columns."
            )

    else:

        st.warning(
            "Benchmark summary not found. "
            "Run visualize_results.py first."
        )


# ==========================================
# TAB 3 - ABOUT PROJECT
# ==========================================

with tab3:

    st.header("About Project TINUOD")

    st.write(
        "Project TINUOD is an academic prototype "
        "that combines artificial intelligence "
        "and parallel computing to study "
        "automated news classification."
    )

    st.divider()

    # ==================================
    # PROJECT METRICS
    # ==================================

    st.subheader("Project Overview")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "V2 Held-Out Test Accuracy",
            "98.41%"
        )

    with col2:
        st.metric(
            "Test Articles",
            "7,731"
        )

    with col3:
        st.metric(
            "MPI Configurations",
            "1, 2, 4"
        )

    st.caption(
        "V2 accuracy was measured on its held-out test dataset. "
        "The MPI figures are from the original benchmark, not a V2 rerun. "
        "Performance on unfamiliar articles may vary."
    )

    st.divider()

    # ==================================
    # TECHNOLOGIES
    # ==================================

    st.subheader("Technologies Used")

    tech1, tech2 = st.columns(2)

    with tech1:

        with st.container(border=True):

            st.markdown("**🤖 Machine Learning**")
            st.write("TF-IDF Vectorization")
            st.write("Logistic Regression")
            st.write("Scikit-learn")

    with tech2:

        with st.container(border=True):

            st.markdown("**⚡ Parallel Computing**")
            st.write("Message Passing Interface")
            st.write("mpi4py")
            st.write("Python")

    with st.container(border=True):

        st.markdown("**🖥️ Application Interface**")
        st.write(
            "Streamlit for interactive news analysis "
            "and MPI performance visualization."
        )

    st.divider()

    # ==================================
    # METHODOLOGY
    # ==================================

    st.subheader("Project Methodology")

    with st.expander(
        "View Development Process",
        expanded=True
    ):

        st.markdown("""
        1. Dataset inspection and preparation
        2. Data cleaning and validation
        3. TF-IDF feature extraction
        4. Logistic Regression model training
        5. Held-out model evaluation
        6. Parallel classification using MPI
        7. Performance benchmarking
        8. Result visualization and interpretation
        9. Streamlit application development
        """)

    st.divider()

    # ==================================
    # LIMITATIONS
    # ==================================

    st.subheader("Project Limitations")

    st.warning(
        "Project TINUOD is an experimental "
        "classification system, not an independent "
        "fact-checking service."
    )

    st.write(
        "Source-attribution cleaning is applied to analyzer inputs. "
        "It reduces sensitivity to certain tested prefixes, "
        "but does not eliminate all source bias. "
        "The model learns language patterns from "
        "its training dataset. It may incorrectly "
        "classify unfamiliar topics, writing styles, "
        "or newly published information."
    )

    st.write(
        "MPI benchmark results reflect the tested "
        "workload and computing environment. "
        "Different hardware or workloads may "
        "produce different performance results."
    )


# ==========================================
# FOOTER
# ==========================================

st.divider()

st.markdown(
    """
    <div class="app-footer">
        Project TINUOD · Parallel and Distributed Computing
        · Academic Research Prototype
    </div>
    """,
    unsafe_allow_html=True
)
