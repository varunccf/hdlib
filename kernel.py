"""
kernel.py
=========
Hybrid Quantum Kernel Matrix using quantum feature extraction.

Approach:
─────────
1. Take 4 random bipolar vectors of dimension N
2. Encode each vector into a quantum state via DiagonalGate:
       |ψ(v)⟩ = DiagonalGate(v) |+⟩^n
3. Measure expectation values of Pauli-Z on each qubit:
       φ(v) = [<Z0>, <Z1>, ..., <Zn-1>]
   This gives a real-valued feature vector of length n = log2(N)
4. Compute RBF kernel on the quantum feature vectors:
       K(i,j) = exp(-γ × ||φ(vi) - φ(vj)||²)

Circuit per qubit measurement:
    |0⟩ → H → DiagonalGate(v) → Measure
    <Zk> = P(0 on qubit k) - P(1 on qubit k)

Three kernel matrices:
  • Classical : RBF on raw vectors v directly
  • Simulator : RBF on quantum features from AerSimulator
  • Hardware  : RBF on quantum features from IBM hardware

Run:
    python kernel.py
    python kernel.py --backend ibm_fez --dim 32
    python kernel.py --sim-only
    python kernel.py --gamma 2.0 --dim 16
"""

import numpy as np
from math import ceil, log2
from typing import List, Optional

from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import DiagonalGate
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import QiskitRuntimeService, Sampler


# ─────────────────────────────────────────────────────────────────────────────
# IBM CONNECTION
# ─────────────────────────────────────────────────────────────────────────────

QiskitRuntimeService.save_account(
    channel="ibm_cloud",
    token="v2WGCXrHMvu1Nh2tzr39Mocon2npcG_ogKxvgvKtTyg2",
    instance="crn:v1:bluemix:public:quantum-computing:us-east:a/813b37ffee14414ca81092ab94341434:1284900f-4e18-41c7-aadf-44278c5d44da::",
    set_as_default=True,
    overwrite=True,
)


# ─────────────────────────────────────────────────────────────────────────────
# COLOURS
# ─────────────────────────────────────────────────────────────────────────────

RESET   = "\033[0m"; BOLD = "\033[1m"; DIM = "\033[2m"
CYAN    = "\033[96m"; GREEN = "\033[92m"; YELLOW = "\033[93m"
RED     = "\033[91m"; BLUE  = "\033[94m"; MAGENTA = "\033[95m"


# ─────────────────────────────────────────────────────────────────────────────
# QUANTUM ENCODING CIRCUIT
# ─────────────────────────────────────────────────────────────────────────────

def build_feature_circuit(vec: np.ndarray, label: str = "v") -> QuantumCircuit:
    """
    Build quantum encoding circuit for a bipolar vector.

    Circuit:
        |0⟩^n → H^n → DiagonalGate(v) → H^n → Measure

    H → DiagonalGate → H  is the key structure:
      - First H:  creates uniform superposition
      - DiagGate: encodes vector values as phases
      - Second H: quantum interference converts phase differences
                  into amplitude differences → measurable probabilities

    Without the final H, all vectors give identical uniform distributions
    (phases are invisible to measurement). The final H makes the phase
    encoding visible via interference.

    Feature vector φ(v) = probability distribution over 2^n bitstrings.
    Different vectors → different interference patterns → different φ(v).
    """
    n = int(ceil(log2(len(vec))))
    v = np.ones(2**n)
    v[:len(vec)] = vec

    gate = DiagonalGate(v.tolist())
    qc   = QuantumCircuit(n, n, name=label)
    qc.h(range(n))            # Layer 1: create superposition
    qc.append(gate, range(n)) # Layer 2: encode phases via DiagonalGate
    qc.h(range(n))            # Layer 3: interference — converts phases to amplitudes
    qc.measure(range(n), range(n))
    return qc


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE EXTRACTION FROM COUNTS
# ─────────────────────────────────────────────────────────────────────────────

def extract_features(counts: dict, n_qubits: int, shots: int) -> np.ndarray:
    """
    Extract probability vector from measurement counts.

    Feature vector φ(v) ∈ R^{2^n}:
        φ(v)[k] = P(bitstring k)  for k = 0..2^n-1

    All entries ≥ 0 and sum to 1 — a proper probability distribution.
    Bitstrings ordered as '000', '001', '010', ..., '111' (binary order).
    """
    dim      = 2 ** n_qubits
    features = np.zeros(dim)

    for bitstring, freq in counts.items():
        # Convert bitstring to integer index
        idx             = int(bitstring.zfill(n_qubits), 2)
        features[idx]  += freq

    features /= shots   # normalise to probabilities
    return features


# ─────────────────────────────────────────────────────────────────────────────
# RBF KERNEL
# ─────────────────────────────────────────────────────────────────────────────

def rbf_kernel(phi_i: np.ndarray, phi_j: np.ndarray,
               gamma: float = 1.0) -> float:
    """
    RBF kernel between two feature vectors:
        K(i,j) = exp(-γ × ||φ(vi) - φ(vj)||²)
    """
    diff = phi_i - phi_j
    return float(np.exp(-gamma * np.dot(diff, diff)))


# ─────────────────────────────────────────────────────────────────────────────
# CLASSICAL KERNEL
# ─────────────────────────────────────────────────────────────────────────────

def classical_kernel_matrix(vectors: List[np.ndarray],
                             gamma: float = 1.0) -> np.ndarray:
    """
    Classical RBF kernel directly on raw vectors.
    Normalised to unit length first for fair comparison.

        K(i,j) = exp(-γ × ||v̂i - v̂j||²)
    """
    n  = len(vectors)
    K  = np.zeros((n, n))
    # Normalise each vector
    vn = [v / np.linalg.norm(v) for v in vectors]
    for i in range(n):
        for j in range(n):
            K[i, j] = rbf_kernel(vn[i], vn[j], gamma)
    return K


# ─────────────────────────────────────────────────────────────────────────────
# QUANTUM FEATURE EXTRACTION — SIMULATOR
# ─────────────────────────────────────────────────────────────────────────────

def quantum_features_simulator(vectors: List[np.ndarray],
                                shots: int = 2048,
                                seed: int  = 42) -> np.ndarray:
    """
    Extract quantum feature vectors for all input vectors using AerSimulator.
    Runs all circuits in a single batch.

    Returns feature matrix of shape (n_vecs, n_qubits).
    """
    backend   = AerSimulator()
    n_qubits  = int(ceil(log2(len(vectors[0]))))
    circuits  = [build_feature_circuit(v, label=f"v{i}")
                 for i, v in enumerate(vectors)]

    tqcs   = transpile(circuits, backend, optimization_level=1)
    result = backend.run(tqcs, shots=shots, seed_simulator=seed).result()

    features = []
    for i, qc in enumerate(circuits):
        counts = result.get_counts(i)
        phi    = extract_features(counts, n_qubits, shots)
        features.append(phi)

    return np.array(features)   # shape: (n_vecs, n_qubits)


def simulator_kernel_matrix(vectors: List[np.ndarray],
                             gamma: float = 1.0,
                             shots: int   = 2048,
                             seed: int    = 42) -> np.ndarray:
    """Compute kernel matrix from simulator quantum features."""
    features = quantum_features_simulator(vectors, shots=shots, seed=seed)
    n        = len(vectors)
    K        = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            K[i, j] = rbf_kernel(features[i], features[j], gamma)
    return K, features


# ─────────────────────────────────────────────────────────────────────────────
# QUANTUM FEATURE EXTRACTION — HARDWARE
# ─────────────────────────────────────────────────────────────────────────────

def quantum_features_hardware(vectors: List[np.ndarray],
                               backend,
                               sampler: Sampler,
                               shots: int = 2048) -> np.ndarray:
    """
    Extract quantum feature vectors using real IBM hardware.
    All circuits submitted in a single batch.
    DD (XpXm) + Gate Twirling ON.

    Returns feature matrix of shape (n_vecs, n_qubits).
    """
    n_qubits = int(ceil(log2(len(vectors[0]))))
    circuits = [build_feature_circuit(v, label=f"v{i}")
                for i, v in enumerate(vectors)]

    print(f"  Transpiling {len(circuits)} feature circuits ...")
    tqcs = transpile(circuits, backend, optimization_level=3)

    # Print circuit info
    d    = tqcs[0].depth()
    n_2q = sum(1 for inst in tqcs[0].data if len(inst.qubits) == 2)
    print(f"  Circuit: {n_qubits} qubits  |  depth={d}  |  2Q gates={n_2q}")

    # DD + Twirling
    sampler.options.dynamical_decoupling.enable        = True
    sampler.options.dynamical_decoupling.sequence_type = "XpXm"
    sampler.options.twirling.enable_gates              = True

    print(f"  Submitting to {backend.name} ...")
    job     = sampler.run(tqcs, shots=shots)
    results = job.result()
    print(f"  Done ✓")

    features = []
    for i in range(len(vectors)):
        # Get counts from the classical register
        counts = results[i].data.meas.get_counts() \
                 if hasattr(results[i].data, 'meas') \
                 else results[i].data.c.get_counts()
        phi = extract_features(counts, n_qubits, shots)
        features.append(phi)

    return np.array(features)


def hardware_kernel_matrix(vectors: List[np.ndarray],
                            backend,
                            sampler: Sampler,
                            gamma: float = 1.0,
                            shots: int   = 2048):
    """Compute kernel matrix from hardware quantum features."""
    features = quantum_features_hardware(vectors, backend, sampler, shots)
    n        = len(vectors)
    K        = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            K[i, j] = rbf_kernel(features[i], features[j], gamma)
    return K, features


# ─────────────────────────────────────────────────────────────────────────────
# PRINT HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def print_feature_matrix(features: np.ndarray, label: str, color: str):
    """Print the quantum feature vectors φ(v)."""
    print(f"\n  {BOLD}{color}{label} — Quantum feature vectors φ(v) = [<Z0>,...,<Zn>]{RESET}")
    print(f"  {color}{'─'*55}{RESET}")
    dim = features.shape[1]
    for i, phi in enumerate(features):
        # Show first 8 and last 2 entries if vector is long
        if dim <= 8:
            vals = "  ".join(f"{x:.4f}" for x in phi)
            print(f"  v{i} (dim={dim}): [{vals}]")
        else:
            head = "  ".join(f"{x:.4f}" for x in phi[:6])
            tail = "  ".join(f"{x:.4f}" for x in phi[-2:])
            print(f"  v{i} (dim={dim}): [{head}  ...  {tail}]  "
                  f"sum={phi.sum():.4f}  max={phi.max():.4f}")


def print_kernel_matrix(K: np.ndarray, title: str, color: str = CYAN):
    """Pretty print a kernel matrix."""
    n = K.shape[0]
    print(f"\n  {BOLD}{color}{title}{RESET}")
    print(f"  {color}{'─'*52}{RESET}")

    header = "         "
    for j in range(n):
        header += f"   v{j}     "
    print(f"  {header}")

    for i in range(n):
        row = f"  v{i}   │  "
        for j in range(n):
            val = K[i, j]
            if i == j:
                row += f"{GREEN}{val:7.4f}{RESET}   "
            elif val > 0.7:
                row += f"{YELLOW}{val:7.4f}{RESET}   "
            else:
                row += f"{val:7.4f}   "
        print(row)


def print_diff_matrix(K1: np.ndarray, K2: np.ndarray,
                      label1: str, label2: str):
    """Print |K1 - K2| difference matrix."""
    n    = K1.shape[0]
    diff = np.abs(K1 - K2)
    print(f"\n  {BOLD}|{label1} − {label2}|{RESET}")
    print(f"  {'─'*50}")

    header = "         "
    for j in range(n):
        header += f"   v{j}     "
    print(f"  {header}")

    for i in range(n):
        row = f"  v{i}   │  "
        for j in range(n):
            val = diff[i, j]
            c   = GREEN if val < 0.02 else YELLOW if val < 0.08 else RED
            row += f"{c}{val:7.4f}{RESET}   "
        print(row)
    print(f"  Max={diff.max():.4f}  Avg={diff.mean():.4f}  "
          f"Frobenius={np.linalg.norm(diff):.4f}")


def check_psd(K: np.ndarray, label: str):
    eigvals = np.linalg.eigvalsh(K)
    is_psd  = bool(np.all(eigvals >= -1e-6))
    print(f"  {label}: "
          f"PSD={GREEN+'YES ✓'+RESET if is_psd else RED+'NO ✗'+RESET}  "
          f"min_eig={eigvals.min():.4f}  "
          f"symmetric_err={np.max(np.abs(K-K.T)):.6f}")


# ─────────────────────────────────────────────────────────────────────────────
# VECTOR HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def generate_vectors(n_vecs: int, dim: int,
                     seed: int = 42) -> List[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [rng.choice([-1., 1.], size=dim) for _ in range(n_vecs)]


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

N_VECTORS = 4
DIM       = 64
GAMMA     = 1.0
SHOTS     = 2048
VEC_SEED  = 42


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def run(use_hardware: bool = True,
        backend_name: str  = "ibm_cleveland",
        dim: int           = DIM,
        gamma: float       = GAMMA,
        shots: int         = SHOTS,
        n_vecs: int        = N_VECTORS):

    print()
    print(f"{CYAN}{'═'*65}{RESET}")
    print(f"{BOLD}{CYAN}   HYBRID QUANTUM KERNEL MATRIX{RESET}")
    print(f"{DIM}   Encoding : H → DiagonalGate(v) → H → Measure  (interference makes phases measurable){RESET}")
    print(f"{DIM}   Features : φ(v) = [P(000), P(001), ..., P(111)]  (probability vector, length 2^n){RESET}")
    print(f"{DIM}   Kernel   : K(i,j) = exp(-γ ||φ(vi) − φ(vj)||²)  (RBF){RESET}")
    print(f"{DIM}   Vectors  : {n_vecs} random bipolar, dim={dim}, γ={gamma}{RESET}")
    print(f"{CYAN}{'═'*65}{RESET}")

    # ── Generate vectors ──────────────────────────────────────────
    vectors  = generate_vectors(n_vecs, dim, seed=VEC_SEED)
    n_qubits = int(ceil(log2(dim)))

    print(f"\n  {BOLD}Vectors:{RESET}")
    for i, v in enumerate(vectors):
        print(f"    v{i}: dim={dim}  n_qubits={n_qubits}  "
              f"first 6 = {v[:6].astype(int).tolist()}")

    # ── Classical kernel ──────────────────────────────────────────
    print(f"\n{MAGENTA}{'─'*65}{RESET}")
    print(f"{BOLD}{MAGENTA}  [1] CLASSICAL RBF KERNEL{RESET}")
    print(f"{DIM}  K(i,j) = exp(-γ ||v̂i − v̂j||²)  on normalised raw vectors{RESET}")
    K_cl = classical_kernel_matrix(vectors, gamma=gamma)
    print_kernel_matrix(K_cl, "Classical kernel matrix", CYAN)

    # ── Simulator kernel ──────────────────────────────────────────
    print(f"\n{MAGENTA}{'─'*65}{RESET}")
    print(f"{BOLD}{MAGENTA}  [2] SIMULATOR QUANTUM KERNEL{RESET}")
    print(f"{DIM}  Extract φ(v) from AerSimulator measurements, then RBF{RESET}")
    K_sim, phi_sim = simulator_kernel_matrix(vectors, gamma=gamma, shots=shots)
    print_feature_matrix(phi_sim, "Simulator", BLUE)
    print_kernel_matrix(K_sim, "Simulator kernel matrix", BLUE)

    # ── Hardware kernel ───────────────────────────────────────────
    K_hw    = None
    phi_hw  = None
    if use_hardware:
        print(f"\n{MAGENTA}{'─'*65}{RESET}")
        print(f"{BOLD}{MAGENTA}  [3] HARDWARE QUANTUM KERNEL{RESET}")
        print(f"{DIM}  Extract φ(v) from IBM hardware, then RBF{RESET}")
        try:
            service    = QiskitRuntimeService()
            hw_backend = service.backend(backend_name)
            sampler    = Sampler(mode=hw_backend)
            print(f"{GREEN}  Connected ✓  ({hw_backend.name}  "
                  f"| {hw_backend.num_qubits} qubits){RESET}")
            K_hw, phi_hw = hardware_kernel_matrix(
                vectors, hw_backend, sampler, gamma=gamma, shots=shots)
            print_feature_matrix(phi_hw, "Hardware", GREEN)
            print_kernel_matrix(K_hw, "Hardware kernel matrix", GREEN)
        except Exception as e:
            print(f"{RED}  Hardware failed: {e}{RESET}")

    # ── Comparison ────────────────────────────────────────────────
    print(f"\n{CYAN}{'═'*65}{RESET}")
    print(f"{BOLD}{CYAN}  KERNEL MATRIX COMPARISON{RESET}")
    print(f"{CYAN}{'═'*65}{RESET}")

    print_diff_matrix(K_cl, K_sim, "Classical", "Simulator")
    if K_hw is not None:
        print_diff_matrix(K_cl, K_hw,  "Classical", "Hardware ")
        print_diff_matrix(K_sim, K_hw, "Simulator", "Hardware ")

    # ── Feature comparison ────────────────────────────────────────
    if phi_hw is not None:
        print(f"\n  {BOLD}Quantum feature vectors — Simulator vs Hardware:{RESET}")
        print(f"  {'─'*55}")
        for i in range(n_vecs):
            diff = np.abs(phi_sim[i] - phi_hw[i])
            print(f"  v{i}: max_diff={diff.max():.4f}  "
                  f"avg_diff={diff.mean():.4f}")

    # ── Kernel properties ─────────────────────────────────────────
    print(f"\n  {BOLD}Kernel matrix properties:{RESET}")
    check_psd(K_cl,  "Classical")
    check_psd(K_sim, "Simulator")
    if K_hw is not None:
        check_psd(K_hw, "Hardware ")

    # ── Diagonal check ────────────────────────────────────────────
    print(f"\n  {BOLD}Diagonal (should be 1.0 — same vector with itself):{RESET}")
    print(f"    Classical : {np.diag(K_cl).round(4)}")
    print(f"    Simulator : {np.diag(K_sim).round(4)}")
    if K_hw is not None:
        print(f"    Hardware  : {np.diag(K_hw).round(4)}")

    print(f"\n{CYAN}{'═'*65}{RESET}\n")
    return K_cl, K_sim, K_hw


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Hybrid Quantum Kernel — DiagonalGate feature extraction + RBF"
    )
    parser.add_argument("--sim-only", action="store_true")
    parser.add_argument("--backend",  type=str,   default="ibm_cleveland")
    parser.add_argument("--dim",      type=int,   default=64)
    parser.add_argument("--gamma",    type=float, default=1.0)
    parser.add_argument("--shots",    type=int,   default=4096)
    parser.add_argument("--n-vecs",   type=int,   default=4)
    parser.add_argument("--seed",     type=int,   default=42)
    args     = parser.parse_args()
    VEC_SEED = args.seed

    run(
        use_hardware = not args.sim_only,
        backend_name = args.backend,
        dim          = args.dim,
        gamma        = args.gamma,
        shots        = args.shots,
        n_vecs       = args.n_vecs,
    )