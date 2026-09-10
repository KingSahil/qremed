"""
quantum_circuit.py
====================
Builds the 4-qubit feature map + ansatz and reports structural circuit
statistics (depth, gate counts, parameter counts).

Feature map:  ZZFeatureMap, reps=1
    - Encodes each of the 4 features as a rotation angle on its own qubit,
      plus a ZZ entangling layer between qubits. This is the standard
      Qiskit-native choice for tabular data and keeps one feature per
      qubit, which matches Phase 1's "4 features -> 4 qubits" design and
      keeps the encoding directly explainable ("qubit 2 encodes 'mean
      concave points'").

Ansatz:  RealAmplitudes, reps=1
    - A shallow, real-amplitude (no complex phases) hardware-efficient
      ansatz using only RY rotations and CX entanglers. Chosen over more
      expressive ansätze (e.g. EfficientSU2) specifically because the
      Phase 2 objective is LOW DEPTH / FEW PARAMETERS / HARDWARE
      COMPATIBILITY, not maximum expressiveness.

reps=1 for both, per the Phase 2 instruction to start shallow and only
increase if there's a demonstrated reason to.
"""

from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import efficient_su2, real_amplitudes, zz_feature_map


def build_feature_map(num_features, reps=1):
    return zz_feature_map(feature_dimension=num_features, reps=reps)


def build_ansatz(num_qubits, reps=1):
    return real_amplitudes(num_qubits=num_qubits, reps=reps)


def build_alternative_ansatz(num_qubits, reps=1):
    """
    Phase 2B Experiment C: one alternative, still-shallow, hardware-efficient
    ansatz -- EfficientSU2 (RY+RZ rotation layers + linear CX entanglers).
    More expressive per layer than RealAmplitudes (adds an RZ rotation and
    uses reverse-linear entanglement by default), while remaining a
    standard, Qiskit-native, hardware-compatible circuit family.
    """
    return efficient_su2(num_qubits=num_qubits, reps=reps)


def build_full_circuit(feature_map, ansatz):
    """Compose feature map + ansatz into the full VQC circuit (unbound)."""
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


def analyze_circuit(feature_map, ansatz, full_circuit):
    """
    Structural analysis of the circuit as built (no transpilation yet --
    that's a separate hardware-compatibility check, see
    hardware_compatibility.py).
    """
    fm_ops, fm_2q, fm_total = _gate_counts(feature_map)
    an_ops, an_2q, an_total = _gate_counts(ansatz)
    full_ops, full_2q, full_total = _gate_counts(full_circuit)

    return {
        "num_qubits": full_circuit.num_qubits,
        "feature_map": {
            "depth": feature_map.depth(),
            "gate_counts": fm_ops,
            "two_qubit_gate_count": fm_2q,
            "total_gate_count": fm_total,
            "num_parameters": feature_map.num_parameters,
        },
        "ansatz": {
            "depth": ansatz.depth(),
            "gate_counts": an_ops,
            "two_qubit_gate_count": an_2q,
            "total_gate_count": an_total,
            "num_parameters": ansatz.num_parameters,
        },
        "full_circuit": {
            "depth": full_circuit.depth(),
            "gate_counts": full_ops,
            "two_qubit_gate_count": full_2q,
            "total_gate_count": full_total,
        },
        "num_trainable_parameters": ansatz.num_parameters,
        "num_input_parameters": feature_map.num_parameters,
    }
