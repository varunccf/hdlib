"""
Two kernel estimation functions for quantum HDC classification.

kernel_estimation     → probability vector features  φ(v) = [P(000),...,P(111)]
kernel_estimation_exp → expectation value features   φ(v) = [<Z0>,...,<Zn-1>]
"""

import numpy as np
import statistics
from typing import List, Optional, Tuple
from contextlib import nullcontext

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import Sampler, Session, SamplerOptions


def _get_counts(res) -> dict:
    """
    Safely extract counts from a hardware result regardless of
    the classical register name.

    Iterates over all attributes of res.data and returns counts
    from the first one that has a get_counts() method.
    """
    for attr_name in vars(res.data):
        attr = getattr(res.data, attr_name)
        if hasattr(attr, 'get_counts'):
            return attr.get_counts()
    raise RuntimeError(
        f"No classical register with get_counts() found in result. "
        f"Available attributes: {list(vars(res.data).keys())}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# FUNCTION 1 — probability vector features
# ─────────────────────────────────────────────────────────────────────────────

def kernel_estimation(
    state_left_circs: List[QuantumCircuit],
    state_right_circs: List[QuantumCircuit],
    backend,
    gamma: float = 1.0,
    shots: int   = 2048,
    seed: int    = 42,
    sampler: Optional[Sampler] = None,
) -> Tuple[List[List[float]], List[np.ndarray]]:
    """
    Kernel estimation using PROBABILITY VECTOR features.

    Takes pre-built encoded circuits, runs them as-is,
    extracts probability distribution as feature vector.

    Feature: φ(v) = [P(b_0), P(b_1), ..., P(b_{2^n-1})]
        - Length = 2^n  (full probability distribution)
        - All entries >= 0, sum to 1

    Kernel: K(i,j) = exp(-γ × ||φ(vi) - φ(vj)||²)

    Parameters
    ----------
    state_left_circs  : pre-built query circuits (test samples)
    state_right_circs : pre-built prototype circuits (class prototypes)
    backend           : AerSimulator or IBM hardware backend
    gamma             : RBF bandwidth (default=1.0)
    shots             : measurement shots
    seed              : simulator random seed
    sampler           : Sampler for hardware (None for simulator)

    Returns
    -------
    similarities  : List[List[float]]  shape (n_left, n_right)
    left_features : List[np.ndarray]   probability vectors for query circuits
    """
    is_simulated = isinstance(backend, AerSimulator)
    n_qubits     = state_left_circs[0].num_qubits

    assert all(c.num_qubits == n_qubits
               for c in state_left_circs + state_right_circs), \
        "All circuits must have the same number of qubits."

    all_circs = state_left_circs + state_right_circs

    # ── Run all circuits in one batch ─────────────────────────────
    if is_simulated:
        tqcs       = transpile(all_circs, backend, optimization_level=3)
        result     = backend.run(tqcs, shots=shots,
                                 seed_simulator=seed).result()
        all_counts = [result.get_counts(i) for i in range(len(all_circs))]

    else:
        if sampler is None:
            raise ValueError("A Sampler must be provided for hardware execution.")

        tqcs = transpile(all_circs, backend, optimization_level=3)

        sampler.options.dynamical_decoupling.enable        = True
        sampler.options.dynamical_decoupling.sequence_type = "XpXm"
        sampler.options.twirling.enable_gates              = True

        results    = sampler.run(tqcs, shots=shots).result()
        all_counts = [_get_counts(res) for res in results]

    # ── Extract probability vector ────────────────────────────────
    def _prob_vector(counts: dict) -> np.ndarray:
        phi = np.zeros(2 ** n_qubits)
        for bitstring, freq in counts.items():
            phi[int(bitstring.zfill(n_qubits), 2)] += freq
        phi /= shots
        return phi

    all_features   = [_prob_vector(c) for c in all_counts]
    left_features  = all_features[:len(state_left_circs)]
    right_features = all_features[len(state_left_circs):]

    # ── RBF kernel ────────────────────────────────────────────────
    similarities = []
    for phi_i in left_features:
        row = []
        for phi_j in right_features:
            diff = phi_i - phi_j
            row.append(float(np.exp(-gamma * np.dot(diff, diff))))
        similarities.append(row)

    return similarities, left_features


# ─────────────────────────────────────────────────────────────────────────────
# FUNCTION 2 — expectation value features
# ─────────────────────────────────────────────────────────────────────────────

def kernel_estimation_exp(
    state_left_circs: List[QuantumCircuit],
    state_right_circs: List[QuantumCircuit],
    backend,
    gamma: float = 1.0,
    shots: int   = 2048,
    seed: int    = 42,
    sampler: Optional[Sampler] = None,
) -> Tuple[List[List[float]], List[np.ndarray]]:
    """
    Kernel estimation using PAULI-Z EXPECTATION VALUE features.

    Takes pre-built encoded circuits, runs them as-is,
    extracts per-qubit expectation values as feature vector.

    Feature: φ(v) = [<Z0>, <Z1>, ..., <Zn-1>]
        - Length = n  (more compact than probability vector)
        - <Zk> = P(qubit k=0) - P(qubit k=1)  ∈ [-1, +1]

    Qiskit bitstring ordering: rightmost bit = qubit 0.
        e.g. '011' → qubit0=1, qubit1=1, qubit2=0

    Kernel: K(i,j) = exp(-γ × ||φ(vi) - φ(vj)||²)

    Parameters
    ----------
    state_left_circs  : pre-built query circuits (test samples)
    state_right_circs : pre-built prototype circuits (class prototypes)
    backend           : AerSimulator or IBM hardware backend
    gamma             : RBF bandwidth (default=1.0)
    shots             : measurement shots
    seed              : simulator random seed
    sampler           : Sampler for hardware (None for simulator)

    Returns
    -------
    similarities  : List[List[float]]  shape (n_left, n_right)
    left_features : List[np.ndarray]   expectation vectors for query circuits
    """
    is_simulated = isinstance(backend, AerSimulator)
    n_qubits     = state_left_circs[0].num_qubits

    assert all(c.num_qubits == n_qubits
               for c in state_left_circs + state_right_circs), \
        "All circuits must have the same number of qubits."

    all_circs = state_left_circs + state_right_circs

    # ── Run all circuits in one batch ─────────────────────────────
    if is_simulated:
        tqcs       = transpile(all_circs, backend, optimization_level=3)
        result     = backend.run(tqcs, shots=shots,
                                 seed_simulator=seed).result()
        all_counts = [result.get_counts(i) for i in range(len(all_circs))]

    else:
        if sampler is None:
            raise ValueError("A Sampler must be provided for hardware execution.")

        tqcs = transpile(all_circs, backend, optimization_level=3)

        sampler.options.dynamical_decoupling.enable        = True
        sampler.options.dynamical_decoupling.sequence_type = "XpXm"
        sampler.options.twirling.enable_gates              = True

        results    = sampler.run(tqcs, shots=shots).result()
        all_counts = [_get_counts(res) for res in results]

    # ── Extract expectation value vector ──────────────────────────
    def _exp_vector(counts: dict) -> np.ndarray:
        count_0 = np.zeros(n_qubits)
        count_1 = np.zeros(n_qubits)
        for bitstring, freq in counts.items():
            bits = bitstring.zfill(n_qubits)
            for k in range(n_qubits):
                if bits[n_qubits - 1 - k] == '0':
                    count_0[k] += freq
                else:
                    count_1[k] += freq
        return (count_0 - count_1) / shots

    all_features   = [_exp_vector(c) for c in all_counts]
    left_features  = all_features[:len(state_left_circs)]
    right_features = all_features[len(state_left_circs):]

    # ── RBF kernel
    similarities = []
    for phi_i in left_features:
        row = []
        for phi_j in right_features:
            diff = phi_i - phi_j
            row.append(float(np.exp(-gamma * np.dot(diff, diff))))
        similarities.append(row)

    return similarities, left_features

def _build_swap_circuit(left_circ: QuantumCircuit,
                        right_circ: QuantumCircuit
                        ) -> QuantumCircuit:
    """
    Build SWAP test circuit from two pre-built encoded circuits.
 
    Structure:
        anc:  |0⟩ ─ H ─────────────────── H ─ Measure
        regA: |0⟩ ─ left_circ  ─ CSWAP ───────────────
        regB: |0⟩ ─ right_circ ─ CSWAP ───────────────
 
    P(anc=0) = (1 + |<L|R>|²) / 2
    Similarity = 2 × P(0) − 1  =  |<L|R>|²
 
    Total qubits = 2×n + 1
    """
    n    = left_circ.num_qubits
    anc  = QuantumRegister(1,  "anc")
    regA = QuantumRegister(n,  "regA")
    regB = QuantumRegister(n,  "regB")
    creg = ClassicalRegister(1, "c_swap")
    qc   = QuantumCircuit(anc, regA, regB, creg)
 
    # Encode both states into separate registers
    qc.compose(left_circ,  qubits=regA, inplace=True)
    qc.compose(right_circ, qubits=regB, inplace=True)
 
    # SWAP test
    qc.h(anc[0])
    for i in range(n):
        qc.cswap(anc[0], regA[i], regB[i])
    qc.h(anc[0])
 
    qc.measure(anc[0], creg[0])
 
    return qc
 
 
# ─────────────────────────────────────────────────────────────────────────────
# SWAP TEST SIMILARITY
# ─────────────────────────────────────────────────────────────────────────────
 
def swap_test_similarity(
    state_left_circs: List[QuantumCircuit],
    state_right_circs: List[QuantumCircuit],
    backend,
    shots: int = 2048,
    seed: int  = 42,
    sampler: Optional[Sampler] = None,
) -> Tuple[List[List[float]], List[dict]]:
    """
    SWAP test pairwise similarity between left and right circuits.
 
    Builds one SWAP test circuit per (left, right) pair,
    runs all N×M circuits in a single batch.
 
    Similarity = 2 × P(ancilla=0) − 1  =  |<L|R>|²
    Clipped to [0, 1].
 
    Same signature as run_compute_uncompute_test — direct drop-in.
 
    Parameters
    ----------
    state_left_circs  : pre-built query circuits (test samples)
    state_right_circs : pre-built prototype circuits (class prototypes)
    backend           : AerSimulator or IBM hardware backend
    shots             : measurement shots
    seed              : simulator random seed
    sampler           : Sampler for hardware (None for simulator)
 
    Returns
    -------
    similarities : List[List[float]]
        2D list shape (n_left, n_right) — |<L|R>|² values ∈ [0, 1]
    all_counts   : List[dict]
        Raw measurement counts for each circuit
    """
    is_simulated = isinstance(backend, AerSimulator)
    n_qubits     = state_left_circs[0].num_qubits
 
    assert all(c.num_qubits == n_qubits
               for c in state_left_circs + state_right_circs), \
        "All circuits must have the same number of qubits."
 
    # ── Build all N×M SWAP circuits ───────────────────────────────
    circuits = []
    for left_circ in state_left_circs:
        for right_circ in state_right_circs:
            circuits.append(_build_swap_circuit(left_circ, right_circ))
 
    # ── Run all circuits in one batch ─────────────────────────────
    if is_simulated:
        tqcs       = transpile(circuits, backend, optimization_level=3)
        result     = backend.run(tqcs, shots=shots,
                                 seed_simulator=seed).result()
        all_counts = [result.get_counts(i) for i in range(len(circuits))]
 
    else:
        if sampler is None:
            raise ValueError("A Sampler must be provided for hardware execution.")
 
        tqcs = transpile(circuits, backend, optimization_level=3)
 
        sampler.options.dynamical_decoupling.enable        = True
        sampler.options.dynamical_decoupling.sequence_type = "XpXm"
        sampler.options.twirling.enable_gates              = True
 
        results    = sampler.run(tqcs, shots=shots).result()
        all_counts = [_get_counts(res) for res in results]
 
    # ── Extract similarity from counts ────────────────────────────
    def _similarity(counts: dict) -> float:
        p0  = counts.get("0", 0) / shots
        return max(0.0, 2.0 * p0 - 1.0)
 
    # ── Group into 2D similarity list ─────────────────────────────
    n_left  = len(state_left_circs)
    n_right = len(state_right_circs)
    idx     = 0
 
    similarities = []
    for _ in range(n_left):
        row = []
        for _ in range(n_right):
            row.append(_similarity(all_counts[idx]))
            idx += 1
        similarities.append(row)
 
    return similarities, all_counts