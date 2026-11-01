# Q-NET: QAOA-Based Network Routing Optimization Simulator

Q-NET is a Python-based quantum computing project that explores network route selection using the Quantum Approximate Optimization Algorithm (QAOA).

It formulates route selection as a Quadratic Unconstrained Binary Optimization (QUBO) problem and compares the QAOA simulation with a classical exhaustive-search baseline.

## Features

- Interactive Streamlit dashboard
- Network topology visualization using NetworkX
- Candidate network path generation
- QUBO-based route selection
- QAOA circuit construction using Qiskit
- Quantum circuit simulation using Qiskit Aer
- Classical exhaustive-search comparison
- Simulated traffic evaluation
- Interactive Plotly visualizations
- Downloadable PDF project report

## Technology Stack

- Python
- Streamlit
- Qiskit
- Qiskit Aer
- NetworkX
- NumPy
- Pandas
- SciPy
- Plotly
- Matplotlib
- ReportLab

## Optimization Model

Each candidate route is represented by a binary decision variable:

- x_i = 1: route selected
- x_i = 0: route not selected

The route cost combines latency and routing cost:

c_i = 0.6 × latency_i + 0.4 × cost_i

The QUBO objective is:

E(x) = Σ(c_i × x_i) + P(1 - Σx_i)²

where P = 50 is the penalty coefficient used to encourage selection of exactly one route.

## QAOA Workflow

1. Generate the network topology.
2. Identify candidate paths.
3. Calculate route costs.
4. Formulate the QUBO problem.
5. Construct the QAOA circuit.
6. Optimize circuit parameters using COBYLA.
7. Simulate quantum measurements using Qiskit Aer.
8. Identify the best feasible route.
9. Compare with classical exhaustive search.

## Classical Comparison

The application includes an exhaustive-search baseline for the current small route-selection problem.

This comparison is intended for educational evaluation and does not demonstrate quantum advantage.

## Traffic Simulation

Q-NET demonstrates three simulated routing policies:

- QAOA-selected route
- Shortest-path route
- Adaptive congestion-aware classical routing

The traffic experiment is synthetic, and the policies do not all use identical route-selection behavior.

## Installation

### Prerequisites

- Python
- Git

### 1. Clone the repository

```bash
git clone https://github.com/yeswanth08-web/Q-NET-Quantum-Network-Optimizer.git
cd Q-NET-Quantum-Network-Optimizer
