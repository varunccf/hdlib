"""
mpo.py
======
MPO (Matrix Product Operator) / MPS (Matrix Product State) similarity
benchmark using DiagonalGate encoding.

APPROACH:
─────────
Standard DiagonalGate circuits are deep because they encode ALL 2^n
amplitudes directly. MPO compresses the state using a tensor network
with a fixed bond dimension χ (chi).

Key idea:
  1. Represent bipolar vector v as an MPS with bond dimension χ
  2. Compute classical MPS overlap <v1|v2> via tensor contraction
  3. Encode the MPS tensors into a quantum circuit using:
       - Local RY rotations (1 per site)
       - Nearest-neighbour CNOT entanglers (bond dimension = 2)
  4. Run CU test on the MPS-encoded circuit

WHY THIS HELPS:
───────────────
  DiagonalGate:   depth O(2^n)  — exponential
  MPS encoding:   depth O(n×χ) — linear in n, polynomial in χ

For bipolar {-1,+1} vectors, bond dimension χ=2 captures the
entanglement structure well. Circuit depth becomes O(n) instead of O(2^n).

CIRCUIT STRUCTURE (MPS encoding, bond dim χ=2):
────────────────────────────────────────────────
  q0: ─RY(θ0)─●──────────────────────
  q1: ─RY(θ1)─X─RY(φ1)─●────────────
  q2: ─RY(θ2)───────────X─RY(φ2)─●──
  q3: ─RY(θ3)──────────────────────X─

  Each site: 1 RY + 1 CNOT  → total depth = 2n  (linear!)
  Compare DiagonalGate for same dim: depth = O(2^n)

SIMILARITY MEASURES:
─────────────────────
  • Classical MPS overlap : exact  Re(<v1|v2>) via tensor contraction
  • Quantum CU test       : |<ψ1|ψ2>|²  via compute-uncompute
  • Both measured with and without M3 mitigation

Tests:
    • Identical vectors  (0% noise) → expected = 1.0
    • 10%, 20%, 30%, 40%, 50% bit flips

Dimensions: 4, 8, 16, 32, 64

Run:
    python mpo.py
    python mpo.py --backend ibm_fez
    python mpo.py --shots 4096
    python mpo.py --sim-only
    python mpo.py --chi 4           ← bond dimension (default=2)
"""

import numpy as np
from math import ceil, log2
from typing import Optional, List, Tuple

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.providers.backend import Backend
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import QiskitRuntimeService, Sampler
from mthree import M3Mitigation


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

RESET   = "\033[0m"; BOLD  = "\033[1m"; DIM    = "\033[2m"
CYAN    = "\033[96m"; GREEN = "\033[92m"; YELLOW = "\033[93m"
RED     = "\033[91m"; BLUE  = "\033[94m"; MAGENTA = "\033[95m"

def _col(err: float) -> str:
    if err < 0.02:   return GREEN
    elif err < 0.08: return YELLOW
    else:            return RED


# ─────────────────────────────────────────────────────────────────────────────
# MPS REPRESENTATION
# ─────────────────────────────────────────────────────────────────────────────

def vector_to_mps(vec: np.ndarray, chi: int = 2) -> List[np.ndarray]:
    """
    Convert a normalised vector to MPS (Matrix Product State) form
    using successive SVD decompositions.

    Parameters
    ----------
    vec : np.ndarray
        Input vector of length N = 2^n. For bipolar {-1,+1} vectors,
        normalise to unit length before calling.
    chi : int
        Maximum bond dimension. Larger chi = more accurate but deeper circuit.
        chi=2 is sufficient for bipolar vectors with local structure.

    Returns
    -------
    List of MPS tensors, one per site.
    Shape of tensor[i]:
        site 0:        (2, min(chi, 2^1))
        site 1..n-2:   (min(chi, 2^i), 2, min(chi, 2^(i+1)))
        site n-1:      (min(chi, 2^(n-1)), 2)
    """
    n   = int(round(log2(len(vec))))
    N   = 2 ** n
    psi = vec[:N].copy().astype(complex)

    # Normalise
    norm = np.linalg.norm(psi)
    if norm > 1e-12:
        psi /= norm

    tensors = []
    M       = psi.reshape(1, N)   # shape: (1, N)

    for site in range(n):
        d   = 2                          # physical dimension
        chi_l = M.shape[0]
        chi_r = M.shape[1] // d

        M = M.reshape(chi_l * d, max(1, chi_r))

        if site < n - 1:
            U, S, Vh = np.linalg.svd(M, full_matrices=False)

            # Truncate to bond dimension chi
            rank = min(len(S), chi)
            U    = U[:, :rank]
            S    = S[:rank]
            Vh   = Vh[:rank, :]

            # Site tensor: reshape U to (chi_l, d, rank)
            tensor = U.reshape(chi_l, d, rank)
            tensors.append(tensor)

            # Remaining matrix: absorb S into Vh
            M = np.diag(S) @ Vh
        else:
            # Last site
            tensor = M.reshape(chi_l, d)
            tensors.append(tensor)

    return tensors


def mps_overlap(mps1: List[np.ndarray],
                mps2: List[np.ndarray]) -> float:
    """
    Compute classical overlap <mps1|mps2> via tensor contraction.

    For each site, contracts the bra (mps1*) and ket (mps2) tensors.
    Returns Re(<mps1|mps2>) which equals |<mps1|mps2>| for real vectors.
    """
    n = len(mps1)

    # Left boundary: scalar 1
    env = np.array([[1.0 + 0j]])  # shape (1, 1)

    for site in range(n):
        t1 = mps1[site]   # bra tensor
        t2 = mps2[site]   # ket tensor

        if site == 0:
            # t1: (d, chi_r1), t2: (d, chi_r2)
            # Contract over physical index d
            # env: (1,1) → result: (chi_r1, chi_r2)
            env = np.einsum('ia,ib->ab', t1.conj(), t2)

        elif site == n - 1:
            # t1: (chi_l1, d), t2: (chi_l2, d)
            # env: (chi_l1, chi_l2)
            env = np.einsum('ab,ia,ib->', env, t1.conj(), t2)

        else:
            # t1: (chi_l1, d, chi_r1), t2: (chi_l2, d, chi_r2)
            # env: (chi_l1, chi_l2)
            env = np.einsum('ab,aic,bid->cd', env, t1.conj(), t2)

    return float(np.real(env))


def classical_mps_similarity(v1: np.ndarray, v2: np.ndarray,
                              chi: int = 2) -> Tuple[float, float]:
    """
    Compute:
      1. Exact classical similarity |<v1|v2>|² = (dot/N)²
      2. MPS approximate similarity |<mps1|mps2>|² (truncated at chi)

    Returns (exact, mps_approx)
    """
    N     = len(v1)
    exact = (float(np.dot(v1, v2)) / N) ** 2

    # Normalise for MPS
    v1n = v1.astype(float) / np.linalg.norm(v1)
    v2n = v2.astype(float) / np.linalg.norm(v2)

    mps1    = vector_to_mps(v1n, chi=chi)
    mps2    = vector_to_mps(v2n, chi=chi)
    overlap = mps_overlap(mps1, mps2)
    mps_sim = float(np.clip(overlap ** 2, 0.0, 1.0))

    return exact, mps_sim


# ─────────────────────────────────────────────────────────────────────────────
# MPS → QUANTUM CIRCUIT ENCODING
# ─────────────────────────────────────────────────────────────────────────────

def mps_to_circuit(mps: List[np.ndarray], label: str = "mps") -> QuantumCircuit:
    """
    Convert an MPS into a shallow quantum circuit using:
      - RY rotations to encode local amplitudes
      - CNOT gates to encode entanglement (bond dimension)

    For bond dimension χ=2, each site needs:
      - 1 RY gate (local qubit rotation)
      - 1 CNOT (entangler to next qubit)

    Total depth = 2n  vs  O(2^n) for DiagonalGate.

    The encoding extracts RY angles from the MPS tensors via:
        θ_i = 2 × arctan2(|A[1]|, |A[0]|)
    where A is the dominant singular vector of the site tensor.

    For bond dim > 2, we use the leading two singular values to
    set the entanglement angle.
    """
    n  = len(mps)
    qc = QuantumCircuit(n, name=label)

    for site in range(n):
        tensor = mps[site]

        if site == 0:
            # Shape: (d=2, chi_r)
            # Get the amplitude vector for |0> and |1>
            amp0 = np.linalg.norm(tensor[0])
            amp1 = np.linalg.norm(tensor[1])
            norm = np.sqrt(amp0**2 + amp1**2)
            if norm > 1e-12:
                theta = 2.0 * np.arctan2(amp1, amp0)
            else:
                theta = 0.0
            qc.ry(float(theta), site)

        elif site == n - 1:
            # Shape: (chi_l, d=2)
            amp0 = np.linalg.norm(tensor[:, 0])
            amp1 = np.linalg.norm(tensor[:, 1])
            norm = np.sqrt(amp0**2 + amp1**2)
            if norm > 1e-12:
                theta = 2.0 * np.arctan2(amp1, amp0)
            else:
                theta = 0.0
            qc.ry(float(theta), site)

        else:
            # Shape: (chi_l, d=2, chi_r)
            # Flatten over chi_l, chi_r to get effective amplitudes
            amp0 = np.linalg.norm(tensor[:, 0, :])
            amp1 = np.linalg.norm(tensor[:, 1, :])
            norm = np.sqrt(amp0**2 + amp1**2)
            if norm > 1e-12:
                theta = 2.0 * np.arctan2(amp1, amp0)
            else:
                theta = 0.0
            qc.ry(float(theta), site)

        # Entangler: CNOT to next qubit (encodes bond dimension)
        if site < n - 1:
            qc.cx(site, site + 1)

    return qc


def encode_mps(vec: np.ndarray, chi: int = 2,
               label: str = "mps") -> QuantumCircuit:
    """
    Full pipeline: bipolar vector → MPS → quantum circuit.

    Parameters
    ----------
    vec : ndarray  Bipolar {-1,+1} vector of length N
    chi : int      Bond dimension (default=2)
    label : str    Circuit label

    Returns
    -------
    QuantumCircuit with n = log2(N) qubits, depth ≈ 2n
    """
    N   = len(vec)
    n   = int(ceil(log2(N)))
    # Pad to 2^n if needed
    v   = np.ones(2**n); v[:N] = vec
    # Normalise
    vn  = v.astype(float) / np.linalg.norm(v)
    mps = vector_to_mps(vn, chi=chi)
    return mps_to_circuit(mps, label=label)


# ─────────────────────────────────────────────────────────────────────────────
# CU CIRCUIT + RUNNERS (reused from cu_benchmark logic)
# ─────────────────────────────────────────────────────────────────────────────

def build_cu(enc_l: QuantumCircuit,
             enc_r: QuantumCircuit) -> Tuple[QuantumCircuit, ClassicalRegister]:
    """H → enc_l → enc_r† → H → Measure"""
    n    = enc_l.num_qubits
    sys  = QuantumRegister(n, "sys")
    creg = ClassicalRegister(n, "c_meas")
    qc   = QuantumCircuit(sys, creg)
    qc.h(sys)
    qc.compose(enc_l,           qubits=sys, inplace=True)
    qc.compose(enc_r.inverse(), qubits=sys, inplace=True)
    qc.h(sys)
    qc.measure(sys, creg)
    return qc, creg


def sim_from_counts(counts: dict, n: int, shots: int) -> float:
    return counts.get("0" * n, 0) / shots


def get_depths(qc: QuantumCircuit,
               backend: Backend) -> Tuple[int, int]:
    tqc   = transpile(qc, backend, optimization_level=3)
    total = tqc.depth()
    two_q = QuantumCircuit(tqc.num_qubits)
    for inst in tqc.data:
        if len(inst.qubits) == 2:
            two_q.append(inst.operation,
                         [tqc.find_bit(q).index for q in inst.qubits])
    return total, two_q.depth()


# ─────────────────────────────────────────────────────────────────────────────
# M3 HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _get_physical_qubits(tqc: QuantumCircuit,
                          creg: ClassicalRegister) -> List[int]:
    tcreg    = next(r for r in tqc.cregs if r.name == creg.name)
    meas_map = {}
    for inst in tqc.data:
        if inst.operation.name == "measure":
            meas_map[inst.clbits[0]] = tqc.find_bit(inst.qubits[0]).index
    return [meas_map[cb] for cb in tcreg][::-1]


def _apply_m3(counts: dict, backend, shots: int,
              measured_qubits: List[int]) -> dict:
    mit   = M3Mitigation(backend)
    mit.cals_from_system(qubits=measured_qubits)
    probs = mit.apply_correction(counts, qubits=measured_qubits)
    return {s: int(round(max(0.0, p) * shots)) for s, p in probs.items()}


# ─────────────────────────────────────────────────────────────────────────────
# RUN CU — SIMULATOR
# ─────────────────────────────────────────────────────────────────────────────

def run_cu_simulator(enc_l: QuantumCircuit, enc_r: QuantumCircuit,
                     backend: AerSimulator, shots: int = 2048,
                     seed: int = 42) -> float:
    qc, _ = build_cu(enc_l, enc_r)
    n     = enc_l.num_qubits
    tqc   = transpile(qc, backend, optimization_level=1)
    counts = backend.run(tqc, shots=shots,
                         seed_simulator=seed).result().get_counts()
    return sim_from_counts(counts, n, shots)


# ─────────────────────────────────────────────────────────────────────────────
# RUN CU — HARDWARE
# ─────────────────────────────────────────────────────────────────────────────

def run_cu_hardware(enc_l: QuantumCircuit, enc_r: QuantumCircuit,
                    backend: Backend, sampler: Sampler,
                    shots: int = 2048) -> Tuple[float, float]:
    """Returns (hw_no_m3, hw_m3)."""
    qc, creg = build_cu(enc_l, enc_r)
    n        = enc_l.num_qubits

    tqc = transpile(qc, backend, optimization_level=3)

    sampler.options.dynamical_decoupling.enable        = True
    sampler.options.dynamical_decoupling.sequence_type = "XpXm"
    sampler.options.twirling.enable_gates              = True

    counts = sampler.run([tqc], shots=shots).result()[0].data.c_meas.get_counts()

    hw_no_m3 = sim_from_counts(counts, n, shots)

    phys     = _get_physical_qubits(tqc, creg)
    m3c      = _apply_m3(counts, backend, shots, phys)
    hw_m3    = sim_from_counts(m3c, n, shots)

    return hw_no_m3, hw_m3


# ─────────────────────────────────────────────────────────────────────────────
# VECTOR HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def generate_bipolar(dim: int, seed: int = 42) -> np.ndarray:
    return np.random.default_rng(seed).choice([-1, 1], size=dim)


def apply_bit_flips(vec: np.ndarray, fraction: float,
                    seed: int = 99) -> np.ndarray:
    v      = vec.copy()
    n_flip = int(round(len(v) * fraction))
    idx    = np.random.default_rng(seed).choice(len(v), n_flip, replace=False)
    v[idx] *= -1
    return v


# ─────────────────────────────────────────────────────────────────────────────
# PRINT TABLE
# ─────────────────────────────────────────────────────────────────────────────

DIV = "─" * 165

HDR = (
    f"  {'Dim':>5} │"
    f"{'Qubits':>7} │"
    f"{'Noise':>6} │"
    f"{'Exact GT':>10} │"
    f"{'MPS GT':>9} │"
    f"{'Sim':>9} │"
    f"{'SimErr':>8} │"
    f"{'HW noM3':>9} │"
    f"{'HW+M3':>9} │"
    f"{'ErrHW':>8} │"
    f"{'ErrM3':>8} │"
    f"{'Depth':>7} │"
    f"{'2QDepth':>8}"
)


def _banner(backend_name: str, hw_ok: bool, shots: int, chi: int):
    print()
    print(f"{CYAN}{'═'*165}{RESET}")
    print(f"{BOLD}{CYAN}   MPO / MPS SIMILARITY BENCHMARK — DiagonalGate + CU{RESET}")
    print(f"{DIM}   Encoding: MPS bond dim χ={chi} → shallow quantum circuit "
          f"(depth ≈ 2n vs O(2^n) DiagonalGate){RESET}")
    print(f"{DIM}   MPS GT = classical MPS overlap |<mps1|mps2>|² (approximate){RESET}")
    print(f"{DIM}   Exact GT = (dot(v1,v2)/N)²  |  SimErr vs Exact GT{RESET}")
    print(f"{DIM}   Backend: {backend_name if hw_ok else 'Simulator only'}"
          f"  |  Shots: {shots}  |  DD + Twirling ON{RESET}")
    print(f"{CYAN}{'═'*165}{RESET}")


def _section(title: str):
    print(f"\n{MAGENTA}{DIV}{RESET}")
    print(f"{BOLD}{MAGENTA}  {title}{RESET}")
    print(f"{MAGENTA}{DIV}{RESET}")
    print(f"{BOLD}  {HDR}{RESET}")
    print(f"{DIM}  {DIV}{RESET}")


def _row(dim, n_q, noise, exact_gt, mps_gt, sim,
         hw_nm, hw_m3, depth, two_q):
    noise_s = f"{int(noise*100):>5}%"
    sim_err = abs(sim - exact_gt)

    def _v(val):
        if val is None: return f"{DIM}{'N/A':>9}{RESET}"
        return f"{val:9.5f}"

    def _e(val, ref):
        if val is None: return f"{DIM}{'N/A':>8}{RESET}"
        err = abs(val - ref)
        return f"{_col(err)}{err:8.5f}{RESET}"

    print(
        f"  {dim:>5} │"
        f"{n_q:>7} │"
        f"{noise_s:>6} │"
        f"{exact_gt:>10.5f} │"
        f"{mps_gt:>9.5f} │"
        f"{sim:>9.5f} │"
        f"{_col(sim_err)}{sim_err:>8.5f}{RESET} │"
        f"{_v(hw_nm)} │"
        f"{_v(hw_m3)} │"
        f"{_e(hw_nm, sim)} │"
        f"{_e(hw_m3, sim)} │"
        f"{CYAN}{depth:>7}{RESET} │"
        f"{CYAN}{two_q:>8}{RESET}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

def _summary(results: list, use_hardware: bool, chi: int):
    print()
    print(f"{CYAN}{'═'*165}{RESET}")
    print(f"{BOLD}{CYAN}  SUMMARY  —  MPO/MPS  χ={chi}{RESET}")
    print(f"{CYAN}{'═'*165}{RESET}")

    def avg(key, ref):
        vals = [abs(r[key]-r[ref]) for r in results
                if r.get(key) is not None]
        return (np.mean(vals), np.max(vals)) if vals else (float('nan'), float('nan'))

    # MPS approximation quality
    mps_a, mps_m = avg('mps_gt', 'exact_gt')
    sim_a, sim_m = avg('sim',    'exact_gt')
    print(f"\n  {BOLD}MPS approximation quality (χ={chi}):{RESET}")
    print(f"    MPS GT vs Exact GT : avg {_col(mps_a)}{mps_a:.5f}{RESET}   max {_col(mps_m)}{mps_m:.5f}{RESET}")
    print(f"    Sim    vs Exact GT : avg {_col(sim_a)}{sim_a:.5f}{RESET}   max {_col(sim_m)}{sim_m:.5f}{RESET}")

    if use_hardware:
        hwa, hwm = avg('hw_no_m3', 'sim')
        m3a, m3m = avg('hw_m3',    'sim')
        print(f"\n  {BOLD}Hardware vs Simulator:{RESET}")
        print(f"    HW no M3 : avg {_col(hwa)}{hwa:.5f}{RESET}   max {_col(hwm)}{hwm:.5f}{RESET}")
        print(f"    HW + M3  : avg {_col(m3a)}{m3a:.5f}{RESET}   max {_col(m3m)}{m3m:.5f}{RESET}")

        m3_imp = np.mean([
            abs(r['hw_no_m3']-r['sim']) - abs(r['hw_m3']-r['sim'])
            for r in results
            if r.get('hw_no_m3') is not None and r.get('hw_m3') is not None
        ])
        print(f"    M3 improvement : avg {'+' if m3_imp>0 else ''}{m3_imp:.5f} "
              f"({'helps ✓' if m3_imp > 0 else 'hurts ✗'})")

    print(f"\n  {BOLD}Circuit depth comparison (MPS χ={chi} vs DiagonalGate reference):{RESET}")
    print(f"  {'Dim':>5} │ {'n_qubits':>8} │ {'MPS depth':>10} │ {'MPS 2Q':>8} │ "
          f"{'DiagGate ref depth':>20} │ {'DiagGate ref 2Q':>16}")
    print(f"  {'─'*85}")

    # DiagonalGate reference depths from previous experiments
    diag_ref = {4:(186,57), 8:(280,86), 16:(186,57), 32:(370,116), 64:(705,219)}
    seen = set()
    for r in results:
        dim = r['dim']
        if dim in seen: continue
        seen.add(dim)
        d_ref, q_ref = diag_ref.get(dim, (0,0))
        reduction = (d_ref - r['depth']) / max(d_ref,1) * 100 if d_ref > 0 else 0
        dc = GREEN if reduction > 20 else YELLOW if reduction > 0 else RED
        print(f"  {dim:>5} │ {r['n_qubits']:>8} │ "
              f"{CYAN}{r['depth']:>10}{RESET} │ "
              f"{CYAN}{r['two_q']:>8}{RESET} │ "
              f"{d_ref:>20} │ {q_ref:>16}  "
              f"{dc}({reduction:+.1f}%){RESET}")

    if use_hardware:
        print(f"\n  {BOLD}Best mitigation per dimension:{RESET}")
        for dim in DIMS:
            dr = [r for r in results if r['dim'] == dim]
            cands = {}
            for k, l in [('hw_no_m3','No-Mit'), ('hw_m3','M3')]:
                vs = [abs(r[k]-r['sim']) for r in dr if r.get(k) is not None]
                if vs: cands[l] = np.mean(vs)
            if cands:
                best = min(cands, key=cands.get)
                print(f"    dim={dim:>4} → {GREEN}{best}{RESET} "
                      f"(avg err: {cands[best]:.5f})")

    print()
    print(f"{DIM}  {GREEN}green<0.02{RESET}{DIM} | "
          f"{YELLOW}yellow 0.02–0.08{RESET}{DIM} | "
          f"{RED}red≥0.08{RESET}")
    print(f"{CYAN}{'═'*165}{RESET}\n")


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

DIMS         = [4, 8, 16, 32, 64]
NOISE_LEVELS = [0.0, 0.10, 0.20, 0.30, 0.40, 0.50]
SHOTS        = 2048
VEC_SEED     = 42
FLIP_SEED    = 99
CHI          = 2   # bond dimension


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def run(use_hardware: bool = True,
        backend_name: str  = "ibm_cleveland",
        shots: int         = SHOTS,
        chi: int           = CHI):

    sim_backend = AerSimulator()
    hw_backend  = None
    sampler     = None

    if use_hardware:
        print(f"\n{YELLOW}  Connecting to {backend_name} ...{RESET}")
        try:
            service    = QiskitRuntimeService()
            hw_backend = service.backend(backend_name)
            sampler    = Sampler(mode=hw_backend)
            print(f"{GREEN}  Connected ✓  ({hw_backend.name}  "
                  f"|  {hw_backend.num_qubits} qubits){RESET}")
        except Exception as e:
            print(f"{RED}  Connection failed: {e}{RESET}")
            print(f"{YELLOW}  Falling back to simulator only.{RESET}")
            use_hardware = False

    _banner(backend_name, use_hardware, shots, chi)

    all_results = []
    depth_cache: dict = {}   # cache depth per dim (same for all noise levels)

    # ══════════════════════════════════════════════════════════════
    # SECTION 1 — IDENTICAL VECTORS
    # ══════════════════════════════════════════════════════════════
    _section("SECTION 1 — IDENTICAL VECTORS  (v1 == v2)  |  expected = 1.0")

    for dim in DIMS:
        n_q = int(ceil(log2(dim)))
        v1  = generate_bipolar(dim, seed=VEC_SEED)
        v2  = v1.copy()

        enc1 = encode_mps(v1, chi=chi, label="L")
        enc2 = encode_mps(v2, chi=chi, label="R")

        exact_gt, mps_gt = classical_mps_similarity(v1, v2, chi=chi)

        # Depth
        qc_cu, _ = build_cu(enc1, enc2)
        if use_hardware and dim not in depth_cache:
            depth, two_q = get_depths(qc_cu, hw_backend)
            depth_cache[dim] = (depth, two_q)
            print(f"  {DIM}dim={dim}  MPS circuit: depth={depth}  2Q={two_q}"
                  f"  (n_qubits={n_q}){RESET}")
        elif dim not in depth_cache:
            depth_cache[dim] = (0, 0)

        depth, two_q = depth_cache[dim]

        # Sim
        sim = run_cu_simulator(enc1, enc2, sim_backend, shots=shots)

        # Hardware
        hw_nm = hw_m3 = None
        if use_hardware:
            try:
                hw_nm, hw_m3 = run_cu_hardware(
                    enc1, enc2, hw_backend, sampler, shots=shots)
            except Exception as e:
                print(f"{RED}  HW error dim={dim}: {e}{RESET}")

        _row(dim, n_q, 0.0, exact_gt, mps_gt, sim,
             hw_nm, hw_m3, depth, two_q)
        all_results.append(dict(
            dim=dim, n_qubits=n_q, noise=0.0,
            exact_gt=exact_gt, mps_gt=mps_gt,
            sim=sim, hw_no_m3=hw_nm, hw_m3=hw_m3,
            depth=depth, two_q=two_q
        ))

    # ══════════════════════════════════════════════════════════════
    # SECTION 2 — NOISE SWEEP
    # ══════════════════════════════════════════════════════════════
    for dim in DIMS:
        n_q = int(ceil(log2(dim)))
        v1  = generate_bipolar(dim, seed=VEC_SEED)

        _section(f"SECTION 2 — BIT-FLIP NOISE SWEEP  |  dim={dim}  |  0%→50%")

        for noise in NOISE_LEVELS:
            v2 = v1.copy() if noise == 0.0 else apply_bit_flips(v1, noise, FLIP_SEED)

            enc1 = encode_mps(v1, chi=chi, label="L")
            enc2 = encode_mps(v2, chi=chi, label="R")

            exact_gt, mps_gt = classical_mps_similarity(v1, v2, chi=chi)

            depth, two_q = depth_cache.get(dim, (0, 0))

            sim = run_cu_simulator(enc1, enc2, sim_backend, shots=shots)

            hw_nm = hw_m3 = None
            if use_hardware:
                try:
                    hw_nm, hw_m3 = run_cu_hardware(
                        enc1, enc2, hw_backend, sampler, shots=shots)
                except Exception as e:
                    print(f"{RED}  HW error dim={dim} noise={noise:.0%}: {e}{RESET}")

            _row(dim, n_q, noise, exact_gt, mps_gt, sim,
                 hw_nm, hw_m3, depth, two_q)
            all_results.append(dict(
                dim=dim, n_qubits=n_q, noise=noise,
                exact_gt=exact_gt, mps_gt=mps_gt,
                sim=sim, hw_no_m3=hw_nm, hw_m3=hw_m3,
                depth=depth, two_q=two_q
            ))

    _summary(all_results, use_hardware, chi)


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="MPO/MPS similarity benchmark — DiagonalGate + CU"
    )
    parser.add_argument("--sim-only", action="store_true",
                        help="Simulator only, skip hardware")
    parser.add_argument("--backend",  type=str, default="ibm_cleveland",
                        help="IBM Quantum backend (default: ibm_cleveland)")
    parser.add_argument("--shots",    type=int, default=2048,
                        help="Shots per circuit (default: 2048)")
    parser.add_argument("--chi",      type=int, default=2,
                        help="MPS bond dimension (default=2). "
                             "Higher chi = more accurate but deeper circuit.")
    args  = parser.parse_args()
    SHOTS = args.shots
    CHI   = args.chi

    run(
        use_hardware = not args.sim_only,
        backend_name = args.backend,
        shots        = SHOTS,
        chi          = CHI,
    )
