
import itertools
import time

import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from scipy.optimize import minimize
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator


# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------

st.set_page_config(
    page_title="Q-NET | Quantum Network Optimizer",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Q-NET")
st.subheader("Quantum Network Optimization Platform")
st.caption("QAOA • Cloud Infrastructure • Intelligent Load Balancing")
st.caption(
    "Quantum Approximate Optimization Algorithm | "
    "Fat-Tree Data Center | Quantum-Classical Benchmarking"
)


# ---------------------------------------------------------
# NETWORK TOPOLOGY
# ---------------------------------------------------------

def build_fat_tree():
    graph = nx.Graph()

    cores = [f"C{i}" for i in range(4)]
    aggs = [f"A{i}" for i in range(4)]
    edges = [f"E{i}" for i in range(4)]
    hosts = [f"H{i}" for i in range(4)]

    for node in cores:
        graph.add_node(
            node, layer="Core", capacity=100.0, load=0.25
        )

    for node in aggs:
        graph.add_node(
            node, layer="Aggregation", capacity=80.0, load=0.40
        )

    for node in edges:
        graph.add_node(
            node, layer="Edge", capacity=60.0, load=0.30
        )

    for node in hosts:
        graph.add_node(
            node, layer="Host", capacity=50.0, load=0.75
        )

    for core in cores:
        for agg in aggs:
            graph.add_edge(
                core, agg, latency=1.5, bandwidth=10.0, cost=2.0
            )

    for i in range(4):
        graph.add_edge(
            aggs[i], edges[i],
            latency=2.0, bandwidth=10.0, cost=1.5
        )
        graph.add_edge(
            aggs[i], edges[(i + 1) % 4],
            latency=2.5, bandwidth=10.0, cost=1.8
        )

    for i in range(4):
        graph.add_edge(
            edges[i], hosts[i],
            latency=0.5, bandwidth=20.0, cost=1.0
        )

    return graph


def get_candidate_paths(graph):
    paths = list(
        nx.all_simple_paths(
            graph, source="H0", target="H2", cutoff=6
        )
    )

    paths.sort(
        key=lambda path: sum(
            graph[path[i]][path[i + 1]]["latency"]
            for i in range(len(path) - 1)
        )
    )

    result = []

    for path in paths[:6]:
        result.append({
            "nodes": path,
            "latency": sum(
                graph[path[i]][path[i + 1]]["latency"]
                for i in range(len(path) - 1)
            ),
            "cost": sum(
                graph[path[i]][path[i + 1]]["cost"]
                for i in range(len(path) - 1)
            )
        })

    return result


# ---------------------------------------------------------
# QUBO FORMULATION
# ---------------------------------------------------------

def build_qubo(paths, penalty=50.0):
    route_costs = np.array([
        path["latency"] * 0.6 + path["cost"] * 0.4
        for path in paths
    ])

    n = len(paths)

    linear = route_costs - penalty
    quadratic = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            quadratic[i, j] = 2.0 * penalty

    def energy(bits):
        value = float(np.dot(linear, bits))

        for i in range(n):
            for j in range(i + 1, n):
                value += quadratic[i, j] * bits[i] * bits[j]

        return value + penalty

    return route_costs, linear, quadratic, energy


# ---------------------------------------------------------
# QAOA CIRCUIT
# ---------------------------------------------------------

def create_qaoa_circuit(linear, quadratic, depth):
    n = len(linear)

    gamma = ParameterVector("gamma", depth)
    beta = ParameterVector("beta", depth)

    circuit = QuantumCircuit(n)

    circuit.h(range(n))

    for layer in range(depth):
        # Cost Hamiltonian: linear terms
        for i in range(n):
            circuit.rz(-gamma[layer] * linear[i], i)

        # Cost Hamiltonian: quadratic terms
        for i in range(n):
            for j in range(i + 1, n):
                q = quadratic[i, j]

                if q != 0:
                    circuit.rzz(gamma[layer] * q / 2, i, j)
                    circuit.rz(-gamma[layer] * q / 2, i)
                    circuit.rz(-gamma[layer] * q / 2, j)

        # Mixer Hamiltonian
        for i in range(n):
            circuit.rx(2 * beta[layer], i)

    return circuit, gamma, beta


def run_qaoa(linear, quadratic, energy, depth, shots, seed):
    n = len(linear)

    circuit, gamma, beta = create_qaoa_circuit(
        linear, quadratic, depth
    )

    rng = np.random.default_rng(seed)

    initial = np.concatenate([
        rng.uniform(0, np.pi, depth),
        rng.uniform(0, np.pi / 2, depth)
    ])

    def expectation(parameters):
        values = {}

        for i in range(depth):
            values[gamma[i]] = parameters[i]
            values[beta[i]] = parameters[depth + i]

        state_circuit = circuit.assign_parameters(values)
        state = Statevector.from_instruction(state_circuit)

        probabilities = state.probabilities_dict()
        expected_energy = 0.0

        for bitstring, probability in probabilities.items():
            bits = np.array([
                int(bitstring[-1 - i]) for i in range(n)
            ])
            expected_energy += probability * energy(bits)

        return float(expected_energy)

    optimization = minimize(
        expectation,
        initial,
        method="COBYLA",
        options={"maxiter": 60, "rhobeg": 0.5}
    )

    parameters = optimization.x
    values = {}

    for i in range(depth):
        values[gamma[i]] = parameters[i]
        values[beta[i]] = parameters[depth + i]

    measured = circuit.assign_parameters(values)
    measured.measure_all()

    simulator = AerSimulator()
    compiled = transpile(
        measured, simulator, seed_transpiler=seed
    )

    result = simulator.run(
        compiled, shots=shots, seed_simulator=seed
    ).result()

    counts = result.get_counts()

    samples = []

    for bitstring, count in counts.items():
        bits = np.array([
            int(bitstring[-1 - i]) for i in range(n)
        ])

        samples.append({
            "bits": bits,
            "count": count,
            "energy": energy(bits),
            "valid": int(np.sum(bits)) == 1
        })

    feasible = [sample for sample in samples if sample["valid"]]

    best = min(
        feasible,
        key=lambda sample: sample["energy"]
    ) if feasible else None

    return {
        "counts": counts,
        "samples": samples,
        "best": best,
        "expected_energy": expectation(parameters),
        "optimizer_success": optimization.success,
        "iterations": optimization.nfev,
        "circuit": measured,
        "parameters": parameters
    }


# ---------------------------------------------------------
# CLASSICAL BENCHMARK
# ---------------------------------------------------------

def run_classical(paths, energy):
    start = time.perf_counter()

    candidates = []

    for bits in itertools.product([0, 1], repeat=len(paths)):
        bits = np.array(bits)
        if np.sum(bits) == 1:
            candidates.append((energy(bits), bits))

    best_energy, best_bits = min(candidates, key=lambda x: x[0])

    exact_time = time.perf_counter() - start

    shortest_index = min(
        range(len(paths)),
        key=lambda i: paths[i]["latency"]
    )

    return {
        "energy": best_energy,
        "bits": best_bits,
        "time": exact_time,
        "shortest_index": shortest_index,
        "evaluated": len(candidates)
    }


# ---------------------------------------------------------
# NETWORK TRAFFIC SIMULATION
# ---------------------------------------------------------

def simulate_traffic(graph, paths, qaoa_index, requests, demand):
    policies = {
        "QAOA-selected route": [],
        "Shortest route": [],
        "Adaptive load balancing": []
    }

    for policy in policies:
        edge_load = {tuple(sorted(e)): 0.0 for e in graph.edges()}
        latencies = []

        for _ in range(requests):
            if policy == "QAOA-selected route":
                index = qaoa_index

            elif policy == "Shortest route":
                index = min(
                    range(len(paths)),
                    key=lambda i: paths[i]["latency"]
                )

            else:
                scores = []

                for i, path in enumerate(paths):
                    congestion = 0.0

                    for j in range(len(path["nodes"]) - 1):
                        edge = tuple(sorted((
                            path["nodes"][j],
                            path["nodes"][j + 1]
                        )))

                        capacity = graph.edges[edge]["bandwidth"]
                        utilization = edge_load[edge] / capacity

                        congestion += utilization ** 2

                    score = paths[i]["latency"] + 5.0 * congestion
                    scores.append((score, i))

                index = min(scores)[1]

            path = paths[index]
            actual_latency = 0.0

            for j in range(len(path["nodes"]) - 1):
                edge = tuple(sorted((
                    path["nodes"][j],
                    path["nodes"][j + 1]
                )))

                capacity = graph.edges[edge]["bandwidth"]
                edge_load[edge] += demand

                utilization = min(
                    edge_load[edge] / capacity, 0.99
                )

                base_latency = graph.edges[edge]["latency"]

                actual_latency += (
                    base_latency * (1 + 0.25 * utilization ** 2)
                )

            latencies.append(actual_latency)

        utilizations = []

        for edge, load in edge_load.items():
            capacity = graph.edges[edge]["bandwidth"]
            utilizations.append(min(load / capacity, 1.0) * 100)

        policies[policy] = {
            "average_latency": float(np.mean(latencies)),
            "maximum_latency": float(np.max(latencies)),
            "average_utilization": float(np.mean(utilizations)),
            "maximum_utilization": float(np.max(utilizations)),
            "latencies": latencies,
            "edge_load": edge_load
        }

    return policies


# ---------------------------------------------------------
# USER INTERFACE
# ---------------------------------------------------------

graph = build_fat_tree()
paths = get_candidate_paths(graph)

if len(paths) < 2:
    st.error("Not enough candidate paths were found.")
    st.stop()

route_costs, linear, quadratic, energy = build_qubo(paths)

st.sidebar.header("Optimization Settings")

depth = st.sidebar.slider(
    "QAOA circuit depth (p)",
    min_value=1,
    max_value=3,
    value=1
)

shots = st.sidebar.select_slider(
    "Quantum measurement shots",
    options=[512, 1024, 2048, 4096],
    value=2048
)

requests = st.sidebar.slider(
    "Number of traffic requests",
    min_value=10,
    max_value=200,
    value=60,
    step=10
)

demand = st.sidebar.slider(
    "Traffic demand per request",
    min_value=0.1,
    max_value=2.0,
    value=0.5,
    step=0.1
)

seed = st.sidebar.number_input(
    "Random seed",
    min_value=1,
    max_value=99999,
    value=42
)

if st.button("🚀 Run Quantum Optimization", type="primary"):
    start = time.perf_counter()

    with st.spinner("Executing QAOA quantum simulation..."):
        qaoa = run_qaoa(
            linear,
            quadratic,
            energy,
            depth,
            shots,
            int(seed)
        )

    qaoa_time = time.perf_counter() - start

    classical = run_classical(paths, energy)

    st.session_state["qaoa"] = qaoa
    st.session_state["classical"] = classical
    st.session_state["qaoa_time"] = qaoa_time
    st.session_state["seed"] = int(seed)

if "qaoa" not in st.session_state:
    st.info(
        "Configure the settings in the sidebar and click "
        "'Run Quantum Optimization' to start."
    )

    st.subheader("Data Center Network")

    positions = nx.spring_layout(graph, seed=42)

    fig = go.Figure()

    for u, v in graph.edges():
        x = [positions[u][0], positions[v][0]]
        y = [positions[u][1], positions[v][1]]

        fig.add_trace(go.Scatter(
            x=x, y=y,
            mode="lines",
            line=dict(color="#9CA3AF", width=1),
            hoverinfo="none",
            showlegend=False
        ))

    for node in graph.nodes():
        fig.add_trace(go.Scatter(
            x=[positions[node][0]],
            y=[positions[node][1]],
            mode="markers+text",
            text=[node],
            textposition="top center",
            marker=dict(size=20),
            name=graph.nodes[node]["layer"]
        ))

    fig.update_layout(
        height=500,
        title="16-Node Fat-Tree Topology",
        showlegend=True
    )

    st.plotly_chart(fig, use_container_width=True)
    st.stop()


# ---------------------------------------------------------
# RESULTS
# ---------------------------------------------------------

qaoa = st.session_state["qaoa"]
classical = st.session_state["classical"]
qaoa_time = st.session_state["qaoa_time"]

st.success("Quantum optimization completed.")

st.subheader("Optimization Results")

if qaoa["best"] is None:
    st.error(
        "QAOA did not sample a feasible one-path solution. "
        "Try increasing the number of measurement shots or circuit depth."
    )
    st.stop()

qaoa_bits = qaoa["best"]["bits"]
qaoa_index = int(np.argmax(qaoa_bits))
classical_index = int(np.argmax(classical["bits"]))

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "QAOA objective",
    f"{qaoa['best']['energy']:.3f}"
)

col2.metric(
    "Exact classical objective",
    f"{classical['energy']:.3f}"
)

col3.metric(
    "QAOA execution time",
    f"{qaoa_time:.3f} s"
)

col4.metric(
    "QAOA circuit qubits",
    len(paths)
)

st.subheader("Selected Network Routes")

for label, index in [
    ("QAOA", qaoa_index),
    ("Exact classical", classical_index)
]:
    path = paths[index]

    st.write(
        f"**{label}:** {' → '.join(path['nodes'])} "
        f"| Base latency: {path['latency']:.2f} ms"
    )

st.subheader("QAOA vs Classical Benchmark")

comparison = pd.DataFrame([
    {
        "Algorithm": "QAOA",
        "Objective": qaoa["best"]["energy"],
        "Execution time (s)": qaoa_time
    },
    {
        "Algorithm": "Classical exhaustive search",
        "Objective": classical["energy"],
        "Execution time (s)": classical["time"]
    }
])

st.dataframe(comparison, use_container_width=True)

fig = px.bar(
    comparison,
    x="Algorithm",
    y="Objective",
    color="Algorithm",
    title="Optimization Objective Comparison"
)

st.plotly_chart(fig, use_container_width=True)

st.caption(
    "The classical exhaustive search is an exact reference for this "
    "small six-variable QUBO. Simulator timings are not a comparison "
    "against real quantum hardware."
)

st.subheader("QAOA Measurement Results")

counts_df = pd.DataFrame([
    {
        "Measured bitstring": bitstring,
        "Shots": count
    }
    for bitstring, count in qaoa["counts"].items()
]).sort_values("Shots", ascending=False)

fig = px.bar(
    counts_df,
    x="Measured bitstring",
    y="Shots",
    title="Quantum Measurement Distribution"
)

st.plotly_chart(fig, use_container_width=True)

st.dataframe(counts_df, use_container_width=True)

st.subheader("Quantum Circuit")

st.code(str(qaoa["circuit"].draw(output="text")))

# ---------------------------------------------------------
# TRAFFIC SIMULATION
# ---------------------------------------------------------

st.subheader("Cloud Network Load-Balancing Simulation")

traffic = simulate_traffic(
    graph,
    paths,
    qaoa_index,
    requests,
    demand
)

traffic_rows = []

for policy, result in traffic.items():
    traffic_rows.append({
        "Policy": policy,
        "Average latency (ms)": result["average_latency"],
        "Maximum latency (ms)": result["maximum_latency"],
        "Average link utilization (%)": result["average_utilization"],
        "Maximum link utilization (%)": result["maximum_utilization"]
    })

traffic_df = pd.DataFrame(traffic_rows)

st.dataframe(traffic_df, use_container_width=True)

fig = px.bar(
    traffic_df,
    x="Policy",
    y="Average latency (ms)",
    color="Policy",
    title="Average Network Latency"
)

st.plotly_chart(fig, use_container_width=True)

fig = px.bar(
    traffic_df,
    x="Policy",
    y="Maximum link utilization (%)",
    color="Policy",
    title="Maximum Link Utilization"
)

st.plotly_chart(fig, use_container_width=True)

st.subheader("Network Topology Visualization")

positions = nx.spring_layout(graph, seed=42)

fig = go.Figure()

for u, v in graph.edges():
    x = [positions[u][0], positions[v][0]]
    y = [positions[u][1], positions[v][1]]

    fig.add_trace(go.Scatter(
        x=x,
        y=y,
        mode="lines",
        line=dict(color="#94A3B8", width=1.5),
        hoverinfo="none",
        showlegend=False
    ))

for node in graph.nodes():
    fig.add_trace(go.Scatter(
        x=[positions[node][0]],
        y=[positions[node][1]],
        mode="markers+text",
        text=[node],
        textposition="top center",
        marker=dict(size=22),
        showlegend=False
    ))

fig.update_layout(
    height=550,
    title="16-Node Fat-Tree Data Center",
    xaxis=dict(visible=False),
    yaxis=dict(visible=False)
)

st.plotly_chart(fig, use_container_width=True)

st.subheader("Candidate Path Analysis")

path_table = pd.DataFrame([
    {
        "Path": i,
        "Route": " → ".join(path["nodes"]),
        "Latency (ms)": path["latency"],
        "Cost": path["cost"],
        "QUBO cost": route_costs[i]
    }
    for i, path in enumerate(paths)
])

st.dataframe(path_table, use_container_width=True)

st.subheader("Simulation Configuration")

st.write(f"- Network nodes: {graph.number_of_nodes()}")
st.write(f"- Network links: {graph.number_of_edges()}")
st.write(f"- Candidate paths: {len(paths)}")
st.write(f"- QAOA depth: {depth}")
st.write(f"- Measurement shots: {shots}")
st.write(f"- Traffic requests: {requests}")
st.write(f"- Traffic demand per request: {demand}")
st.write(f"- Random seed: {st.session_state['seed']}")

st.info(
    "Research note: QAOA optimizes a one-route selection QUBO. "
    "The adaptive load-balancing policy is a separate classical "
    "congestion-aware baseline. Traffic and latency are simulated, "
    "not measured from a physical data center."
)
# ================= COMPLETE PDF REPORT =================

import io
from datetime import datetime
from xml.sax.saxutils import escape

import matplotlib.pyplot as plt

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, Image as PDFImage, Preformatted
)


def make_chart_image(fig):
    """Convert a Matplotlib figure into an in-memory PNG for the PDF."""
    image_buffer = io.BytesIO()
    fig.savefig(image_buffer, format="png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    image_buffer.seek(0)
    return image_buffer


def make_pdf_report():
    """Create one complete PDF containing project documentation and live results."""
    buffer = io.BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=15 * mm,
        leftMargin=15 * mm,
        topMargin=16 * mm,
        bottomMargin=17 * mm,
        title="Q-NET Complete Project and Experiment Report",
        author="Q-NET"
    )

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="QNetTitle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=25,
        leading=30,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#087EAD"),
        spaceAfter=8
    ))
    styles.add(ParagraphStyle(
        name="QNetSubtitle",
        parent=styles["Normal"],
        alignment=TA_CENTER,
        fontSize=11,
        leading=15,
        spaceAfter=8
    ))
    styles.add(ParagraphStyle(
        name="QNetBody",
        parent=styles["BodyText"],
        fontSize=8.5,
        leading=12,
        spaceAfter=6
    ))
    styles.add(ParagraphStyle(
        name="QNetSmall",
        parent=styles["BodyText"],
        fontSize=7,
        leading=9
    ))
    styles.add(ParagraphStyle(
        name="QNetTableHeader",
        parent=styles["QNetSmall"],
        textColor=colors.white,
        fontName="Helvetica-Bold"
    ))
    styles.add(ParagraphStyle(
        name="QNetCode",
        parent=styles["Code"],
        fontName="Courier",
        fontSize=6.2,
        leading=7.5
    ))

    story = []

    def heading(text, level=1):
        story.append(Paragraph(escape(text), styles[f"Heading{level}"]))
        story.append(Spacer(1, 3))

    def para(text):
        story.append(Paragraph(escape(str(text)), styles["QNetBody"]))

    def add_table(headers, rows, widths=None, font_size=7):
        data = [[Paragraph(escape(str(v)), styles["QNetTableHeader"]) for v in headers]]
        for row in rows:
            data.append([
                Paragraph(escape(str(v)), styles["QNetSmall"])
                for v in row
            ])

        table = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#12304A")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#AAB4C0")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 5),
            ("RIGHTPADDING", (0, 0), (-1, -1), 5),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("FONTSIZE", (0, 0), (-1, -1), font_size),
        ]))
        story.append(table)
        story.append(Spacer(1, 8))

    def add_figure(fig, width=165 * mm):
        image_buffer = make_chart_image(fig)
        story.append(PDFImage(image_buffer, width=width, height=width * 0.53))
        story.append(Spacer(1, 7))

    # Cover and project documentation
    story.append(Spacer(1, 20 * mm))
    story.append(Paragraph("Q-NET", styles["QNetTitle"]))
    story.append(Paragraph(
        "QUANTUM NETWORK OPTIMIZATION PLATFORM",
        styles["QNetSubtitle"]
    ))
    story.append(Paragraph(
        "Complete Project Documentation and Experiment Report",
        styles["QNetSubtitle"]
    ))
    story.append(Spacer(1, 8))
    para(f"Report generated: {datetime.now().strftime('%d %B %Y, %I:%M:%S %p')}")
    para("Technology focus: QAOA, QUBO, quantum circuit simulation, "
         "Fat-Tree data-center networking, and classical benchmarking.")
    story.append(Spacer(1, 8))

    heading("Abstract")
    para(
        "Q-NET is a software-based experimental platform that models a small "
        "Fat-Tree data-center network and formulates selection of one candidate "
        "network route as a Quadratic Unconstrained Binary Optimization (QUBO) "
        "problem. The QUBO is evaluated using a Quantum Approximate Optimization "
        "Algorithm (QAOA) circuit executed on a simulator. An exact classical "
        "enumeration provides a reference optimum for the small problem. A "
        "separate simulated traffic experiment compares the QAOA-selected route, "
        "the shortest route, and a congestion-aware adaptive routing baseline."
    )

    heading("Problem Statement")
    para(
        "Selecting a network route requires balancing path latency and route "
        "cost while enforcing the requirement that exactly one candidate route "
        "is selected. This project represents that constrained selection task "
        "as a QUBO and investigates how a parameterized QAOA circuit samples "
        "solutions. It also models how different routing policies behave under "
        "synthetic traffic demand."
    )

    heading("Project Objectives")
    for item in [
        "Construct a reproducible 16-node, 28-link Fat-Tree-style network model.",
        "Generate up to six candidate paths between source H0 and destination H2.",
        "Create a weighted route objective and an exactly-one-route QUBO penalty.",
        "Build and execute a parameterized QAOA circuit with configurable depth and shots.",
        "Compare the sampled QAOA solution with exact classical enumeration.",
        "Simulate traffic and compare latency and link-utilization metrics across routing policies.",
        "Export the project description, configuration, calculations, tables, diagrams, and experiment outputs in one PDF."
    ]:
        para("- " + item)

    heading("Scope and Methodology")
    para(
        "The application builds the graph, enumerates candidate paths, computes "
        "route costs, constructs the QUBO, converts its terms into QAOA phase "
        "operations, optimizes circuit parameters with COBYLA using exact "
        "statevector expectation values, and samples the optimized circuit with "
        "Qiskit Aer. The classical benchmark enumerates every one-hot bitstring. "
        "Finally, synthetic traffic requests are passed through three routing "
        "policies and the resulting metrics are summarized."
    )

    heading("Software and Libraries")
    para(
        "Python, Streamlit, NetworkX, NumPy, pandas, Plotly, SciPy, Qiskit, "
        "Qiskit Aer, Matplotlib, and ReportLab."
    )

    # Current experiment settings
    heading("1. Experiment Configuration")
    add_table(
        ["Setting", "Current value"],
        [
            ["Network model", "16-node Fat-Tree-style graph"],
            ["Nodes", graph.number_of_nodes()],
            ["Links", graph.number_of_edges()],
            ["Source and destination", "H0 -> H2"],
            ["Candidate routes", len(paths)],
            ["QAOA depth (p)", depth],
            ["Measurement shots", shots],
            ["Traffic requests", requests],
            ["Demand per request", demand],
            ["Random seed", st.session_state.get("seed", "Not recorded")],
            ["Classical optimizer", "COBYLA; maximum 60 function evaluations"],
            ["Quantum execution", "Qiskit Aer simulator; statevector used for parameter expectation"]
        ],
        widths=[58 * mm, 105 * mm]
    )

    # Network topology and node/edge inventories
    heading("2. Network Topology")
    para(
        "The modeled topology contains four core nodes, four aggregation nodes, "
        "four edge nodes, and four host nodes. Link attributes in the model "
        "include latency, bandwidth, and cost."
    )

    node_rows = []
    for node, attributes in graph.nodes(data=True):
        node_rows.append([
            node,
            attributes.get("layer", ""),
            attributes.get("capacity", ""),
            attributes.get("load", "")
        ])
    add_table(
        ["Node", "Layer", "Capacity", "Initial load"],
        node_rows,
        widths=[30 * mm, 55 * mm, 38 * mm, 40 * mm]
    )

    edge_rows = []
    for u, v, attributes in graph.edges(data=True):
        edge_rows.append([
            u + " - " + v,
            attributes.get("latency", ""),
            attributes.get("bandwidth", ""),
            attributes.get("cost", "")
        ])
    add_table(
        ["Link", "Latency", "Bandwidth", "Cost"],
        edge_rows,
        widths=[48 * mm, 38 * mm, 38 * mm, 38 * mm]
    )

    positions = nx.spring_layout(graph, seed=42)
    fig, ax = plt.subplots(figsize=(9, 5))
    nx.draw_networkx(
        graph, positions, ax=ax, with_labels=True,
        node_size=650, font_size=8, width=1.0
    )
    ax.set_title("Q-NET Fat-Tree Network Topology")
    ax.axis("off")
    add_figure(fig)

    # Candidate route analysis
    heading("3. Candidate Path Analysis")
    path_rows = []
    for index, path_info in enumerate(paths):
        path_rows.append([
            index,
            " -> ".join(path_info["nodes"]),
            f'{path_info["latency"]:.4f}',
            f'{path_info["cost"]:.4f}',
            f'{route_costs[index]:.4f}',
            "Yes" if index == qaoa_index else "No",
            "Yes" if index == classical_index else "No"
        ])
    add_table(
        ["ID", "Route", "Latency", "Cost", "Weighted cost", "QAOA", "Exact"],
        path_rows,
        widths=[12 * mm, 55 * mm, 21 * mm, 18 * mm, 24 * mm, 16 * mm, 16 * mm]
    )

    # QUBO math and formulation
    heading("4. QUBO Mathematical Formulation")
    para(
        "Let x_i be a binary decision variable for candidate route i. "
        "x_i = 1 means that route i is selected; otherwise x_i = 0. "
        "The route score used by the implementation is c_i = 0.6 L_i + 0.4 K_i, "
        "where L_i is the sum of link latencies and K_i is the sum of link costs."
    )
    para("The exactly-one-route constraint is: sum_i x_i = 1.")
    para(
        "The implemented penalized objective is: "
        "E(x) = sum_i c_i x_i + P(1 - sum_i x_i)^2, with P = 50."
    )
    para(
        "Expanded QUBO form: E(x) = P + sum_i(c_i - P)x_i "
        "+ 2P sum_(i<j) x_i x_j. The constant P is retained in the energy "
        "function. The linear and pairwise coefficients are used to construct "
        "the QAOA cost-phase operations."
    )
    para(
        "For this small candidate set, the classical benchmark enumerates all "
        "2^n bitstrings and retains only the n bitstrings with exactly one "
        "selected route. It then chooses the lowest-energy feasible bitstring."
    )

    heading("5. QAOA Circuit and Optimization")
    para(
        f"The circuit uses {len(paths)} qubits, one per candidate route, and "
        f"QAOA depth p = {depth}. The initial state is prepared with Hadamard "
        "gates. Each layer applies the cost unitary using RZ and RZZ operations, "
        "followed by an X-mixer using RX operations."
    )
    para(
        "The gamma and beta parameters are initialized randomly using the "
        "configured seed. COBYLA minimizes the expected QUBO energy computed "
        "from a Qiskit Statevector. The optimized circuit is then measured "
        "using Qiskit Aer with the configured shot count."
    )
    para(
        f"Optimizer status: {'reported success' if qaoa['optimizer_success'] else 'did not report success'}. "
        f"Function evaluations: {qaoa['iterations']}."
    )

    parameter_values = np.asarray(qaoa["parameters"], dtype=float).tolist()
    gamma_values = parameter_values[:depth]
    beta_values = parameter_values[depth:depth * 2]
    add_table(
        ["Parameter", "Optimized value(s)"],
        [
            ["Gamma", ", ".join(f"{v:.8f}" for v in gamma_values)],
            ["Beta", ", ".join(f"{v:.8f}" for v in beta_values)]
        ],
        widths=[45 * mm, 118 * mm]
    )

    heading("Circuit Text Diagram")
    circuit_text = str(qaoa["circuit"].draw(output="text", fold=-1))
    circuit_text = circuit_text.encode("ascii", errors="replace").decode("ascii")
    story.append(Preformatted(circuit_text[:12000], styles["QNetCode"]))
    story.append(Spacer(1, 8))

    # Optimization results
    heading("6. Optimization Results")
    qaoa_best_energy = float(qaoa["best"]["energy"])
    classical_energy = float(classical["energy"])
    add_table(
        ["Metric", "QAOA", "Exact classical"],
        [
            ["Selected route ID", qaoa_index, classical_index],
            ["Selected route", " -> ".join(paths[qaoa_index]["nodes"]),
             " -> ".join(paths[classical_index]["nodes"])],
            ["Objective energy", f"{qaoa_best_energy:.8f}", f"{classical_energy:.8f}"],
            ["Execution time (s)", f"{qaoa_time:.6f}", f"{classical['time']:.6f}"],
            ["Expected QAOA energy", f"{float(qaoa['expected_energy']):.8f}", "Not applicable"],
            ["Feasible one-hot states evaluated", "Measured samples shown below",
             classical["evaluated"]]
        ],
        widths=[52 * mm, 56 * mm, 55 * mm]
    )

    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    ax.bar(
        ["QAOA sampled", "Exact classical"],
        [qaoa_best_energy, classical_energy]
    )
    ax.set_ylabel("QUBO objective energy")
    ax.set_title("Best sampled QAOA result vs exact classical reference")
    ax.grid(axis="y", alpha=0.25)
    add_figure(fig)

    para(
        "Interpretation: the exact classical result is the optimum for this "
        "small six-variable one-route QUBO. The QAOA result is the best feasible "
        "one-hot bitstring observed in the finite measurement sample. A simulator "
        "runtime is not evidence of a quantum speedup."
    )

    # Full measurement counts and sample states
    heading("7. QAOA Measurement Distribution")
    measurement_rows = []
    for sample in sorted(qaoa["samples"], key=lambda item: item["count"], reverse=True):
        bitstring = "".join(str(int(bit)) for bit in sample["bits"][::-1])
        measurement_rows.append([
            bitstring,
            sample["count"],
            f'{float(sample["energy"]):.8f}',
            "Valid one-hot" if sample["valid"] else "Invalid"
        ])
    add_table(
        ["Measured bitstring", "Shots", "Energy", "Constraint status"],
        measurement_rows,
        widths=[42 * mm, 25 * mm, 45 * mm, 51 * mm]
    )

    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    sorted_counts = sorted(qaoa["counts"].items(), key=lambda item: item[1], reverse=True)
    labels = [item[0] for item in sorted_counts]
    values = [item[1] for item in sorted_counts]
    ax.bar(labels, values)
    ax.set_xlabel("Measured bitstring")
    ax.set_ylabel("Measurement count")
    ax.set_title("QAOA measurement counts")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", alpha=0.25)
    add_figure(fig)

    # Traffic simulation: complete metrics and per-request observations
    heading("8. Traffic Simulation Method")
    para(
        f"The simulator generates {requests} requests, each adding demand "
        f"{demand} to the links on its selected route. The QAOA-selected-route "
        "policy sends every request over the route selected by QAOA. The shortest "
        "route policy sends every request over the minimum-base-latency route. "
        "The adaptive policy scores each candidate route using base path latency "
        "plus a congestion penalty derived from current link utilization."
    )

    heading("9. Traffic Simulation Results")
    traffic_summary_rows = []
    for policy, result in traffic.items():
        traffic_summary_rows.append([
            policy,
            f'{result["average_latency"]:.6f}',
            f'{result["maximum_latency"]:.6f}',
            f'{result["average_utilization"]:.4f}',
            f'{result["maximum_utilization"]:.4f}'
        ])
    add_table(
        ["Policy", "Avg latency (ms)", "Max latency (ms)",
         "Avg link use (%)", "Max link use (%)"],
        traffic_summary_rows,
        widths=[47 * mm, 30 * mm, 30 * mm, 29 * mm, 29 * mm]
    )

    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    policy_names = list(traffic.keys())
    avg_latencies = [traffic[name]["average_latency"] for name in policy_names]
    ax.bar(policy_names, avg_latencies)
    ax.set_ylabel("Average latency (ms)")
    ax.set_title("Average simulated network latency by policy")
    ax.tick_params(axis="x", rotation=12)
    ax.grid(axis="y", alpha=0.25)
    add_figure(fig)

    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    max_utils = [traffic[name]["maximum_utilization"] for name in policy_names]
    ax.bar(policy_names, max_utils)
    ax.set_ylabel("Maximum link utilization (%)")
    ax.set_title("Maximum simulated link utilization by policy")
    ax.tick_params(axis="x", rotation=12)
    ax.grid(axis="y", alpha=0.25)
    add_figure(fig)

    heading("10. Per-request Latency Records")
    request_rows = []
    for policy, result in traffic.items():
        for request_number, latency in enumerate(result["latencies"], start=1):
            request_rows.append([
                policy,
                request_number,
                f"{float(latency):.6f}"
            ])
    add_table(
        ["Routing policy", "Request number", "Simulated latency (ms)"],
        request_rows,
        widths=[75 * mm, 38 * mm, 50 * mm]
    )

    heading("11. Final Link Loads")
    link_load_rows = []
    for policy, result in traffic.items():
        for edge, load in result["edge_load"].items():
            edge_name = str(edge[0]) + " - " + str(edge[1])
            capacity = graph.edges[edge]["bandwidth"]
            utilization = min(float(load) / float(capacity), 1.0) * 100
            link_load_rows.append([
                policy,
                edge_name,
                f"{float(load):.4f}",
                f"{float(capacity):.4f}",
                f"{utilization:.2f}"
            ])
    add_table(
        ["Policy", "Link", "Final load", "Capacity", "Utilization (%)"],
        link_load_rows,
        widths=[48 * mm, 32 * mm, 27 * mm, 27 * mm, 29 * mm]
    )

    # Conclusions and limitations
    heading("12. Discussion")
    para(
        "The optimization component addresses a constrained single-route "
        "selection problem. QAOA produces a probability distribution over "
        "binary route selections; finite shots may include infeasible bitstrings. "
        "The report identifies the best feasible sampled state and compares it "
        "with an exact classical reference."
    )
    para(
        "The traffic simulation is a separate experiment. In particular, the "
        "QAOA-selected-route policy uses one route for all requests, while the "
        "adaptive classical policy can choose different routes as congestion "
        "changes. These policies therefore have different routing capabilities "
        "and should not be described as a like-for-like comparison of equivalent "
        "optimization algorithms."
    )

    heading("13. Limitations")
    for item in [
        "The quantum circuit is executed on a simulator, not physical quantum hardware.",
        "The network topology, link properties, traffic requests, and latency model are synthetic.",
        "The QUBO selects exactly one route; it does not solve a full multi-commodity or multi-path traffic allocation problem.",
        "The candidate route set is limited to at most six paths between H0 and H2.",
        "The exact classical enumeration is practical because the current problem is small; it is not a scalable benchmark for large networks.",
        "Simulation outcomes depend on the selected settings, seed, and simplified congestion model."
    ]:
        para("- " + item)

    heading("14. Conclusion")
    para(
        "Q-NET demonstrates an end-to-end software workflow for representing "
        "a small network route-selection problem as a QUBO, constructing and "
        "sampling a QAOA circuit, benchmarking against exact classical search, "
        "and evaluating separate simulated traffic-routing policies. The "
        "results provide an experimental baseline for future work on larger "
        "routing formulations, multi-path allocation, richer network models, "
        "and carefully controlled quantum-versus-classical comparisons."
    )

    heading("15. Future Scope")
    for item in [
        "Extend the formulation from one-route selection to multi-request or multi-path allocation.",
        "Include capacity constraints and request-specific bandwidth requirements directly in the optimization model.",
        "Evaluate larger candidate sets and compare QAOA with scalable classical heuristics.",
        "Test hardware execution only where suitable backend access and problem sizes permit.",
        "Add repeated-seed experiments and statistical summaries for more reliable evaluation."
    ]:
        para("- " + item)

    para(
        "End of report. All numerical values and charts in this document are "
        "generated from the current Q-NET application run and its selected settings."
    )

    def add_page_number(canvas, document):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#64748B"))
        canvas.drawString(15 * mm, 9 * mm, "Q-NET | Complete Experiment Report")
        canvas.drawRightString(
            A4[0] - 15 * mm, 9 * mm, f"Page {document.page}"
        )
        canvas.restoreState()

    doc.build(story, onFirstPage=add_page_number, onLaterPages=add_page_number)
    buffer.seek(0)
    return buffer.getvalue()


# One export button: the entire report is contained in this PDF.
st.divider()
st.header("Complete Project Report")
st.caption(
    "Download one PDF containing project documentation, topology, "
    "QUBO formulation, QAOA details, measurements, charts, and traffic results."
)

try:
    complete_pdf = make_pdf_report()
    st.download_button(
        label="Download Complete Q-NET PDF",
        data=complete_pdf,
        file_name="QNET_Complete_Project_and_Experiment_Report.pdf",
        mime="application/pdf",
        type="primary",
        use_container_width=True
    )
except Exception as pdf_error:
    st.error(f"Could not generate the PDF report: {pdf_error}")
    st.exception(pdf_error)
