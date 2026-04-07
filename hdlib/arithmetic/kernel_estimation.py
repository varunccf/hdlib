import numpy as np
import statistics
from math import ceil, log2
from typing import List, Optional, Tuple
from contextlib import nullcontext
 
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import Sampler, Session, SamplerOptions
 
 
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
 
    Circuit: H → DiagonalGate(v) → H → Measure
    Feature: φ(v) = [P(000), P(001), ..., P(111)]   length = 2^n
    Kernel:  K(i,j) = exp(-γ ||φ(vi) - φ(vj)||²)
 
    Runs N+M circuits total (N left + M right).
    Returns (similarities 2D list, left feature vectors).
    """
    is_simulated = isinstance(backend, AerSimulator)
    n_qubits     = state_left_circs[0].num_qubits
    all_circs    = state_left_circs + state_right_circs
 
    if is_simulated:
        tqcs       = transpile(all_circs, backend, optimization_level=1)
        result     = backend.run(tqcs, shots=shots,
                                 seed_simulator=seed).result()
        all_counts = [result.get_counts(i) for i in range(len(all_circs))]
 
    else:
        if sampler is None:
            raise ValueError("Sampler must be provided for hardware.")
 
        tqcs   = transpile(all_circs, backend, optimization_level=3)
 
        sampler.options.dynamical_decoupling.enable        = True
        sampler.options.dynamical_decoupling.sequence_type = "XpXm"
        sampler.options.twirling.enable_gates              = True
 
        results    = sampler.run(tqcs, shots=shots).result()
        all_counts = []
        for res in results:
            data = res.data
            if hasattr(data, 'c_meas'):
                all_counts.append(data.c_meas.get_counts())
            elif hasattr(data, 'c'):
                all_counts.append(data.c.get_counts())
            else:
                reg_name = list(vars(data).keys())[0]
                all_counts.append(getattr(data, reg_name).get_counts())
 
    def _prob_vector(counts):
        phi = np.zeros(2 ** n_qubits)
        for bitstring, freq in counts.items():
            phi[int(bitstring.zfill(n_qubits), 2)] += freq
        phi /= shots
        return phi
 
    all_features   = [_prob_vector(c) for c in all_counts]
    left_features  = all_features[:len(state_left_circs)]
    right_features = all_features[len(state_left_circs):]
 
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
 
    Circuit: H → DiagonalGate(v) → H → Measure
    Feature: φ(v) = [<Z0>, <Z1>, ..., <Zn-1>]   length = n
             <Zk> = P(qubit k=0) - P(qubit k=1)  ∈ [-1, +1]
    Kernel:  K(i,j) = exp(-γ ||φ(vi) - φ(vj)||²)
 
    Runs N+M circuits total (N left + M right).
    Returns (similarities 2D list, left feature vectors).
    """
    is_simulated = isinstance(backend, AerSimulator)
    n_qubits     = state_left_circs[0].num_qubits
    all_circs    = state_left_circs + state_right_circs
 
    if is_simulated:
        tqcs       = transpile(all_circs, backend, optimization_level=1)
        result     = backend.run(tqcs, shots=shots,
                                 seed_simulator=seed).result()
        all_counts = [result.get_counts(i) for i in range(len(all_circs))]
 
    else:
        if sampler is None:
            raise ValueError("Sampler must be provided for hardware.")
 
        tqcs   = transpile(all_circs, backend, optimization_level=3)
 
        sampler.options.dynamical_decoupling.enable        = True
        sampler.options.dynamical_decoupling.sequence_type = "XpXm"
        sampler.options.twirling.enable_gates              = True
 
        results    = sampler.run(tqcs, shots=shots).result()
        all_counts = []
        for res in results:
            data = res.data
            if hasattr(data, 'c_meas'):
                all_counts.append(data.c_meas.get_counts())
            elif hasattr(data, 'c'):
                all_counts.append(data.c.get_counts())
            else:
                reg_name = list(vars(data).keys())[0]
                all_counts.append(getattr(data, reg_name).get_counts())
 
    def _exp_vector(counts):
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
 
    similarities = []
    for phi_i in left_features:
        row = []
        for phi_j in right_features:
            diff = phi_i - phi_j
            row.append(float(np.exp(-gamma * np.dot(diff, diff))))
        similarities.append(row)
 
    return similarities, left_features
