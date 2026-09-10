"""
hardware_compatibility.py
============================
STATIC / STRUCTURAL checks only -- transpiles the trained circuit against
a generic IBM-typical basis gate set to inspect depth/gate structure.
Does NOT run a Fake Backend and does NOT connect to real IBM hardware.
Unchanged in spirit from the original WDBC module; works for any circuit.
"""
from __future__ import annotations

from qiskit import transpile

IBM_TYPICAL_BASIS_GATES = ["rz", "sx", "x", "cx"]


def check_hardware_compatibility(circuit, basis_gates=None, optimization_level: int = 1) -> dict:
    basis_gates = basis_gates or IBM_TYPICAL_BASIS_GATES

    transpiled = transpile(circuit, basis_gates=basis_gates, optimization_level=optimization_level)

    ops = transpiled.count_ops()
    two_qubit_gate_names = {"cx", "cz", "ecr"}
    two_qubit_count = sum(v for k, v in ops.items() if k in two_qubit_gate_names)

    unsupported_gates_found = [
        gate for gate in circuit.count_ops()
        if gate not in basis_gates and gate not in ("measure", "barrier")
    ]

    return {
        "basis_gates_used_for_check": basis_gates,
        "optimization_level": optimization_level,
        "num_qubits": circuit.num_qubits,
        "original_depth": circuit.depth(),
        "transpiled_depth": transpiled.depth(),
        "transpiled_gate_counts": dict(ops),
        "transpiled_two_qubit_gate_count": two_qubit_count,
        "gates_outside_basis_before_transpile": unsupported_gates_found,
        "transpilation_succeeded": True,
        "pipeline_stage": "Qiskit Aer simulation (current results) -> static IBM-basis transpilation check (this step) -> noisy simulation / Fake backend / real QPU (not yet performed)",
        "note": (
            "This is a STATIC transpilation check against a generic IBM-typical "
            "basis gate set -- it does NOT reflect a specific real device's "
            "coupling map, calibration, or noise, and no Fake Backend or real "
            "hardware was used. Current results reflect Qiskit Aer simulation only."
        ),
    }
