import streamlit as st
import json
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path
from pyvis.network import Network
import tempfile
import ollama

# -------------------------------------------------------------
# PATHS
# -------------------------------------------------------------
ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
ASSETS_DIR = ROOT / "assets"

RESULTS_FILE = DATA_DIR / "results.json"
TS_FILE = DATA_DIR / "brazilian_ecommerce_timeseries.csv"
REPORT_FILE = DATA_DIR / "analysis_report.txt"
DEFAULT_GRAPH_IMAGE = ASSETS_DIR / "causal_graph.png"


# -------------------------------------------------------------
# LOAD DATA
# -------------------------------------------------------------
with open(RESULTS_FILE, "r") as f:
    results = json.load(f)

kpi_dict = results.get("kpi_analyses", {})
KPI_NAMES = list(kpi_dict.keys())

analysis_report = REPORT_FILE.read_text() if REPORT_FILE.exists() else ""
ts_df = pd.read_csv(TS_FILE, parse_dates=["date"]) if TS_FILE.exists() else None


# -------------------------------------------------------------
# HELPER FUNCTIONS
# -------------------------------------------------------------
def extract_contributors(kpi: str, level_filter: str):
    """Filter contributors by L0/L1/L2 if required."""
    rows = []
    for item in kpi_dict.get(kpi, []):
        factor = item.get("factor", "")

        # LEVEL DETECTION HEURISTIC
        if "seller" in factor or "approval" in factor:
            level = "L1"
        elif "geographic" in factor or "market" in factor:
            level = "L2"
        else:
            level = "L0"

        if level_filter != "All" and level != level_filter:
            continue

        rows.append({
            "Factor": factor,
            "Level": level,
            "R² Score": round(item.get("r2", 0), 3),
            "Path": " → ".join(item.get("path", [])),
            "Importance": round(item.get("multivariate_importance", 0), 3)
        })

    return pd.DataFrame(rows)


def build_subgraph(kpi: str):
    """Build causal graph for PyVis."""
    G = nx.DiGraph()
    for item in kpi_dict.get(kpi, []):
        for i in range(len(item.get("path", [])) - 1):
            G.add_edge(item["path"][i], item["path"][i+1])
    return G


def display_pyvis_graph(G, kpi_name):
    """Generate and display an interactive PyVis causal graph."""
    net = Network(height="600px", width="100%", directed=True, bgcolor="#ffffff", font_color="#000")

    for node in G.nodes():
        net.add_node(node, label=node)

    for u, v in G.edges():
        net.add_edge(u, v)

    # Export to temporary HTML
    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp:
        net.save_graph(tmp.name)
        st.components.v1.html(open(tmp.name).read(), height=600, scrolling=True)


def generate_llm_narrative(kpi_name, contributors_df):
    """Ask LLM to summarize insights for the KPI."""
    prompt = f"""
You are an expert business analyst.

KPI: {kpi_name}

Here are the top contributors:
{contributors_df.to_string()}

Here is the general analysis report:
{analysis_report}

Write a concise but insightful explanation of what drives changes in this KPI.
Explain causal pathways and what actions a business should prioritize.
"""

    try:
        response = ollama.chat(model="llama3:latest", messages=[{"role": "user", "content": prompt}])
        return response["message"]["content"]
    except Exception as e:
        return f"(LLM unavailable) {e}"


def plot_time_series(kpi_name):
    if ts_df is None or "total_revenue" not in ts_df.columns:
        st.warning("No time-series data available.")
        return

    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(ts_df["date"], ts_df["total_revenue"], linewidth=2)
    ax.set_title(f"Time Series for {kpi_name}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Revenue")
    ax.grid(True)
    st.pyplot(fig)


# -------------------------------------------------------------
# STREAMLIT UI
# -------------------------------------------------------------
st.set_page_config(page_title="Causal KPI Dashboard", layout="wide")
st.title("📊 Advanced Causal KPI Dashboard")
st.caption("Interactive causal graphs • LLM narratives • Contributor filtering")

# KPI Selection
kpi_choice = st.selectbox("Select a KPI", KPI_NAMES)

# Contributor Filters
level_filter = st.radio("Filter Contributors", ["All", "L0", "L1", "L2"], horizontal=True)

# Extract contributors after filtering
contributors_df = extract_contributors(kpi_choice, level_filter)

# Layout
left, right = st.columns([1, 1.3])


# -------------------------------------------------------------
# LEFT PANEL — Graph + Contributors
# -------------------------------------------------------------
with left:
    st.subheader("🧠 Interactive Causal Graph")

    kpi_specific_image = ASSETS_DIR / f"{kpi_choice}_causal_graph.png"
    if kpi_specific_image.exists():
        st.image(kpi_specific_image, caption=f"{kpi_choice} Causal Graph", use_column_width=True)
    else:
        G = build_subgraph(kpi_choice)
        if len(G.nodes) > 0:
            display_pyvis_graph(G, kpi_choice)
        else:
            st.info("No causal graph available.")

    st.subheader("📄 Contributors")
    st.dataframe(contributors_df, use_container_width=True)


# -------------------------------------------------------------
# RIGHT PANEL — LLM Narrative + Time Series
# -------------------------------------------------------------
with right:
    st.subheader("📝 LLM KPI Narrative")

    if st.button("Generate Narrative with Llama3"):
        narrative = generate_llm_narrative(kpi_choice, contributors_df)
        st.write(narrative)
    else:
        st.info("Click the button to generate an AI-powered narrative.")

    st.subheader("📈 Time Series")
    plot_time_series(kpi_choice)
