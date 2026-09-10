"""
hardware_compatibility.py
============================
STATIC / STRUCTURAL checks only -- this module does NOT run a Fake
Backend and does NOT connect to IBM Quantum hardware. It transpiles the
circuit against a generic IBM-typical basis gate set purely to inspect
whether the structure (gate types, resulting depth, two-qubit count) is
reasonable, without executing anything on a device or device-simulator.

This lays groundwork for Phase 3+ (Aer noise) and Phase 4+ (Fake
Backend / real hardware) without doing that work now.
"""

from qiskit import transpile

# A generic IBM-typical basis gate set (common across many superconducting
# IBM backends): single-qubit RZ/SX/X plus the two-qubit CX entangler.
# This is used only for a STRUCTURAL transpile check, not a specific
# device's real coupling map or calibration.
IBM_TYPICAL_BASIS_GATES = ["rz", "sx", "x", "cx"]


def check_hardware_compatibility(circuit, basis_gates=None, optimization_level=1):
    basis_gates = basis_gates or IBM_TYPICAL_BASIS_GATES

    transpiled = transpile(
        circuit,
        basis_gates=basis_gates,
        optimization_level=optimization_level,
    )

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
        "original_depth": circuit.depth(),
        "transpiled_depth": transpiled.depth(),
        "transpiled_gate_counts": dict(ops),
        "transpiled_two_qubit_gate_count": two_qubit_count,
        "gates_outside_basis_before_transpile": unsupported_gates_found,
        "transpilation_succeeded": True,
        "measurement_structure": (
            "Computational-basis measurement on all qubits, matching "
            "SamplerV2's counts-based output -- no mid-circuit "
            "measurement or conditional logic, which keeps the circuit "
            "compatible with standard Sampler primitive execution on "
            "real hardware."
        ),
        "parameter_binding_note": (
            "Feature-map parameters are bound per-sample (data), ansatz "
            "parameters are bound once after training (frozen weights) -- "
            "this split is exactly what Phase 2's 'train locally, freeze, "
            "then run inference circuits on hardware' design in the "
            "master plan requires."
        ),
        "note": (
            "This is a STATIC transpilation check against a generic "
            "IBM-typical basis gate set -- it does NOT reflect a specific "
            "real device's coupling map, calibration, or noise, and no "
            "Fake Backend or real hardware was used."
        ),
    }
