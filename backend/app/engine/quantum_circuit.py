"""
quantum_circuit.py
====================
Builds the feature map + ansatz and reports structural circuit
statistics. Generalized to any qubit count (was hard-coded to 4 in the
original WDBC-only module).

Feature map: ZZFeatureMap (Q-REMED's standard choice for tabular data;
one feature per qubit).

Ansatz: EfficientSU2 by default (the platform's primary VQC per the
Q-REMED product spec: "ZZFeatureMap + EfficientSU2"). RealAmplitudes
remains available as the original project's low-depth, hardware-biased
alternative.
"""
from __future__ import annotations

from qiskit import QuantumCircuit
from qiskit.circuit.library import efficient_su2, real_amplitudes, zz_feature_map


def build_feature_map(num_features: int, reps: int = 1):
    return zz_feature_map(feature_dimension=num_features, reps=reps)


def build_ansatz(num_qubits: int, reps: int = 1, kind: str = "efficient_su2"):
    if kind == "real_amplitudes":
        return real_amplitudes(num_qubits=num_qubits, reps=reps)
    return efficient_su2(num_qubits=num_qubits, reps=reps)


def build_full_circuit(feature_map, ansatz):
    circuit = QuantumCircuit(feature_map.num_qubits)
    circuit.compose(feature_map, inplace=True)
    circuit.compose(ansatz, inplace=True)
    return circuit


def _gate_counts(circuit):
    ops = circuit.count_ops()
    two_qubit_gate_names = {"cx", "cz", "ecr", "cp", "rzz", "rxx", "ryy"}
    two_qubit_count = sum(v for k, v in ops.items() if k in two_qubit_gate_names)
    total_gates = sum(ops.values())
    return dict(ops), two_qubit_count, total_gates


def analyze_circuit(feature_map, ansatz, full_circuit) -> dict:
    fm_ops, fm_2q, fm_total = _gate_counts(feature_map)
    an_ops, an_2q, an_total = _gate_counts(ansatz)
    full_ops, full_2q, full_total = _gate_counts(full_circuit)

    return {
        "num_qubits": full_circuit.num_qubits,
        "feature_map": {"depth": feature_map.depth(), "gate_counts": fm_ops,
                         "two_qubit_gate_count": fm_2q, "total_gate_count": fm_total,
                         "num_parameters": feature_map.num_parameters},
        "ansatz": {"depth": ansatz.depth(), "gate_counts": an_ops,
                   "two_qubit_gate_count": an_2q, "total_gate_count": an_total,
                   "num_parameters": ansatz.num_parameters},
        "full_circuit": {"depth": full_circuit.depth(), "gate_counts": full_ops,
                          "two_qubit_gate_count": full_2q, "total_gate_count": full_total},
        "num_trainable_parameters": ansatz.num_parameters,
        "num_input_parameters": feature_map.num_parameters,
    }
