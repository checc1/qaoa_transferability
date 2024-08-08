import pennylane as qml
from jax import numpy as jnp
import networkx as nx


def BetaCircuit(beta: jnp.array, qubits: int):
    for wire in range(qubits):
        qml.RX(phi=2*beta, wires=wire)


def GammaCircuit(gamma: jnp.array, graph: nx.Graph):
    for edge in graph:
        wire1 = edge[0]
        wire2 = edge[1]
        qml.CNOT(wires=[wire1, wire2])
        qml.RZ(phi=2*gamma, wires=wire2)
        qml.CNOT(wires=[wire1, wire2])