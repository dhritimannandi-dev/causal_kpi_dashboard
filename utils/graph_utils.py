import networkx as nx
import matplotlib.pyplot as plt
import streamlit as st


def build_subgraph(kpi_record):
    G = nx.DiGraph()
    for level in ["L0_contributors", "L1_contributors", "L2_contributors"]:
        for contrib in kpi_record.get(level, []):
            G.add_edge(contrib["column"], kpi_record["kpi_name"])
    return G


def render_graph(G):
    plt.figure(figsize=(7, 5))
    nx.draw_networkx(G, with_labels=True, arrows=True,
                     node_color="#85C1E9", font_size=8)
    st.pyplot(plt)

