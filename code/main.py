import qiskit_aer as q_aer
import networkx as nx
from utilities import *
from maxcut import *
from circuit_QAOA import *


if __name__ == "__main__":
    shots = 10_000
    layers = 3
    backend = q_aer.Aer.get_backend("qasm_simulator")
    G = nx.Graph()

    # TODO: ....