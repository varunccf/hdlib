"""
swap_test.py
============
SWAP test similarity benchmark using DiagonalGate encoding.

Similarity measure:
    P(ancilla=0) = (1 + |<v1|v2>|²) / 2
    → Similarity = 2 * P(0) - 1  = |<v1|v2>|²

Circuit structure:
    Ancilla:    |0> → H → ─────────────────── → H → Measure
    Register A: |0> → DiagonalGate(v1) → CSWAP ↕
    Register B: |0> → DiagonalGate(v2) → CSWAP ↕

Total qubits = 2 × log2(N) + 1  (two registers + 1 ancilla)

Tests:
    • Identical vectors  (0% noise)  → expected similarity = 1.0
    • 10% bit flips                  → partial similarity
    • 20% bit flips
    • 30% bit flips
    • 40% bit flips
    • 50% bit flips  (orthogonal)    → expected similarity ≈ 0.0

Dimensions: 4, 8, 16, 32, 64

Run:
    python swap_test.py
    python swap_test.py --backend ibm_fez
    python swap_test.py --shots 4096
    python swap_test.py --sim-only
"""

import numpy as np
from math import ceil, log2
from typing import Optional, List, Tuple, Dict

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.circuit.library import DiagonalGate
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
# ENCODE — DiagonalGate
# ─────────────────────────────────────────────────────────────────────────────

def encode(vec: np.ndarray, label: str = "v") -> QuantumCircuit:
    """
    Encode a bipolar {-1,+1} vector using DiagonalGate.
    N elements → log2(N) qubits.
    """
    vec = np.asarray(vec)
    if not np.all(np.isin(vec, [-1, 1])):
        raise ValueError("Vector must be bipolar {-1, +1}.")
    n = int(ceil(log2(len(vec))))
    if len(vec) < 2**n:
        vec = np.concatenate([vec, np.ones(2**n - len(vec))])
    gate = DiagonalGate(vec.tolist()); gate.label = label
    qc   = QuantumCircuit(n, name=label)
    qc.append(gate, range(n))
    return qc


# ─────────────────────────────────────────────────────────────────────────────
# SWAP TEST CIRCUIT
# ─────────────────────────────────────────────────────────────────────────────

def build_swap_circuit(enc_v1: QuantumCircuit,
                       enc_v2: QuantumCircuit
                       ) -> Tuple[QuantumCircuit, ClassicalRegister]:
    """
    Build the SWAP test circuit.

    Structure:
        anc:  |0> ─ H ─────────────── H ─ Measure
        regA: |0> ─ enc_v1 ─ CSWAP ─────────────
        regB: |0> ─ enc_v2 ─ CSWAP ─────────────

    P(anc=0) = (1 + |<v1|v2>|²) / 2
    Similarity = 2 * P(0) - 1

    Total qubits = 2*n + 1  where n = log2(N)
    """
    n    = enc_v1.num_qubits
    anc  = QuantumRegister(1, "anc")
    regA = QuantumRegister(n, "regA")
    regB = QuantumRegister(n, "regB")
    creg = ClassicalRegister(1, "c_swap")
    qc   = QuantumCircuit(anc, regA, regB, creg)

    # Encode v1 into regA
    qc.compose(enc_v1, qubits=regA, inplace=True)

    # Encode v2 into regB
    qc.compose(enc_v2, qubits=regB, inplace=True)

    # SWAP test
    qc.h(anc[0])
    for i in range(n):
        qc.cswap(anc[0], regA[i], regB[i])   # controlled-SWAP per qubit
    qc.h(anc[0])

    qc.measure(anc[0], creg[0])

    return qc, creg


# ─────────────────────────────────────────────────────────────────────────────
# CIRCUIT DEPTH
# ─────────────────────────────────────────────────────────────────────────────

def get_depths(qc: QuantumCircuit,
               backend: Backend) -> Tuple[int, int]:
    """
    Transpile to hardware and return (total_depth, two_q_depth).
    """
    tqc = transpile(qc, backend, optimization_level=3)
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
# SIMILARITY FROM COUNTS
# ─────────────────────────────────────────────────────────────────────────────

def similarity_from_counts(counts: dict, shots: int) -> float:
    """
    SWAP test: P(0) = (1 + |<v1|v2>|²) / 2
    → Similarity = 2 * P(0) - 1  ∈ [-1, 1]
    Clipped to [0, 1] since similarity is non-negative.
    """
    p0  = counts.get("0", 0) / shots
    sim = max(0.0, 2 * p0 - 1)
    return sim


# ─────────────────────────────────────────────────────────────────────────────
# CLASSICAL GROUND TRUTH
# ─────────────────────────────────────────────────────────────────────────────

def classical_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Classical ground truth for SWAP test:
        |<v1|v2>|² = (dot(v1,v2) / N)²
    Same as CU test ground truth.
    """
    N = len(v1)
    return (float(np.dot(v1, v2)) / N) ** 2


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
# RUN SWAP TEST — SIMULATOR
# ─────────────────────────────────────────────────────────────────────────────

def run_swap_simulator(enc_v1: QuantumCircuit,
                       enc_v2: QuantumCircuit,
                       backend: AerSimulator,
                       shots: int = 2048,
                       seed: int  = 42) -> float:
    qc, _ = build_swap_circuit(enc_v1, enc_v2)
    tqc   = transpile(qc, backend, optimization_level=1)
    counts = backend.run(tqc, shots=shots,
                         seed_simulator=seed).result().get_counts()
    return similarity_from_counts(counts, shots)


# ─────────────────────────────────────────────────────────────────────────────
# RUN SWAP TEST — HARDWARE
# ─────────────────────────────────────────────────────────────────────────────

def run_swap_hardware(enc_v1: QuantumCircuit,
                      enc_v2: QuantumCircuit,
                      backend: Backend,
                      sampler: Sampler,
                      shots: int = 2048
                      ) -> Tuple[float, float]:
    """
    Run SWAP test on hardware.
    Returns (sim_no_m3, sim_m3).
    """
    qc, creg = build_swap_circuit(enc_v1, enc_v2)

    tqc = transpile(qc, backend, optimization_level=3)

    sampler.options.dynamical_decoupling.enable        = True
    sampler.options.dynamical_decoupling.sequence_type = "XpXm"
    sampler.options.twirling.enable_gates              = True

    job      = sampler.run([tqc], shots=shots)
    result   = job.result()
    bit_data = result[0].data.c_swap
    counts_1 = int(np.count_nonzero(bit_data.array == 1))
    counts   = {"0": shots - counts_1, "1": counts_1}

    # No mitigation
    sim_no_m3 = similarity_from_counts(counts, shots)

    # M3 mitigation
    phys      = _get_physical_qubits(tqc, creg)
    m3_counts = _apply_m3(counts, backend, shots, phys)
    sim_m3    = similarity_from_counts(m3_counts, shots)

    return sim_no_m3, sim_m3


# ─────────────────────────────────────────────────────────────────────────────
# PRINT TABLE
# ─────────────────────────────────────────────────────────────────────────────

DIV = "─" * 145

HDR = (
    f"  {'Dim':>5} │"
    f"{'Qubits':>7} │"
    f"{'Noise':>6} │"
    f"{'Classical':>11} │"
    f"{'Sim':>9} │"
    f"{'SimErr':>8} │"
    f"{'HW no M3':>10} │"
    f"{'HW + M3':>9} │"
    f"{'Err noM3':>9} │"
    f"{'Err M3':>8} │"
    f"{'Depth':>7} │"
    f"{'2QDepth':>8}"
)


def _banner(backend_name: str, hw_ok: bool, shots: int):
    print()
    print(f"{CYAN}{'═'*145}{RESET}")
    print(f"{BOLD}{CYAN}   SWAP TEST BENCHMARK — DiagonalGate Encoding{RESET}")
    print(f"{DIM}   Similarity = 2×P(ancilla=0) − 1  =  |<v1|v2>|²{RESET}")
    print(f"{DIM}   Qubits = 2×log₂(N) + 1  |  Backend: "
          f"{backend_name if hw_ok else 'Simulator only'}"
          f"  |  Shots: {shots}  |  DD + Twirling ON{RESET}")
    print(f"{CYAN}{'═'*145}{RESET}")


def _section(title: str):
    print(f"\n{MAGENTA}{DIV}{RESET}")
    print(f"{BOLD}{MAGENTA}  {title}{RESET}")
    print(f"{MAGENTA}{DIV}{RESET}")
    print(f"{BOLD}  {HDR}{RESET}")
    print(f"{DIM}  {DIV}{RESET}")


def _row(dim, n_qubits, noise, classical, sim, hw_nm, hw_m3,
         depth, two_q):
    noise_s = f"{int(noise*100):>5}%"
    sim_err = abs(sim - classical)

    def _v(val):
        if val is None: return f"{DIM}{'N/A':>9}{RESET}"
        return f"{val:9.5f}"

    def _e(val, ref):
        if val is None: return f"{DIM}{'N/A':>9}{RESET}"
        err = abs(val - ref)
        return f"{_col(err)}{err:9.5f}{RESET}"

    print(
        f"  {dim:>5} │"
        f"{n_qubits:>7} │"
        f"{noise_s:>6} │"
        f"{classical:>11.5f} │"
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

def _summary(results: list, use_hardware: bool):
    print()
    print(f"{CYAN}{'═'*145}{RESET}")
    print(f"{BOLD}{CYAN}  SUMMARY{RESET}")
    print(f"{CYAN}{'═'*145}{RESET}")

    def avg(key, ref):
        vals = [abs(r[key]-r[ref]) for r in results
                if r.get(key) is not None]
        return (np.mean(vals), np.max(vals)) if vals else (float('nan'), float('nan'))

    sa, sm = avg('sim', 'classical')
    print(f"\n  Simulator vs Classical : avg {_col(sa)}{sa:.5f}{RESET}   max {_col(sm)}{sm:.5f}{RESET}")

    if use_hardware:
        hwa, hwm = avg('hw_no_m3', 'sim')
        m3a, m3m = avg('hw_m3',    'sim')
        print(f"  HW no M3 vs Sim       : avg {_col(hwa)}{hwa:.5f}{RESET}   max {_col(hwm)}{hwm:.5f}{RESET}")
        print(f"  HW + M3  vs Sim       : avg {_col(m3a)}{m3a:.5f}{RESET}   max {_col(m3m)}{m3m:.5f}{RESET}")

        m3_imp = np.mean([abs(r['hw_no_m3']-r['sim']) - abs(r['hw_m3']-r['sim'])
                          for r in results
                          if r.get('hw_no_m3') is not None and r.get('hw_m3') is not None])
        sign = "+" if m3_imp > 0 else ""
        print(f"  M3 improvement        : avg {sign}{m3_imp:.5f} "
              f"({'helps' if m3_imp > 0 else 'hurts'})")

    print(f"\n  Circuit depth per dimension:")
    seen = set()
    for r in results:
        if r['dim'] not in seen:
            seen.add(r['dim'])
            print(f"    dim={r['dim']:>4}  ({r['n_qubits']} enc qubits, "
                  f"{2*r['n_qubits']+1} total)  →  "
                  f"depth={r['depth']:>5}   2Q depth={r['two_q']:>4}")

    if use_hardware:
        print(f"\n  Best HW result per dimension (no M3):")
        for dim in DIMS:
            dr = [r for r in results if r['dim'] == dim and r.get('hw_no_m3') is not None]
            if not dr: continue
            best = max(dr, key=lambda r: r['hw_no_m3'])
            print(f"    dim={dim:>4}  noise={int(best['noise']*100):>2}%  "
                  f"hw={best['hw_no_m3']:.5f}  classical={best['classical']:.5f}")

    print()
    print(f"{DIM}  {GREEN}green err<0.02{RESET}{DIM} | "
          f"{YELLOW}yellow 0.02–0.08{RESET}{DIM} | "
          f"{RED}red ≥0.08{RESET}")
    print(f"{CYAN}{'═'*145}{RESET}\n")


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

DIMS         = [4, 8, 16, 32, 64]
NOISE_LEVELS = [0.0, 0.10, 0.20, 0.30, 0.40, 0.50]
SHOTS        = 2048
VEC_SEED     = 42
FLIP_SEED    = 99


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def run(use_hardware: bool = True,
        backend_name: str  = "ibm_cleveland",
        shots: int         = SHOTS):

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

    _banner(backend_name, use_hardware, shots)

    all_results = []

    # ══════════════════════════════════════════════════════════════
    # SECTION 1 — IDENTICAL VECTORS
    # ══════════════════════════════════════════════════════════════
    _section("SECTION 1 — IDENTICAL VECTORS  (v1 == v2)  |  expected = 1.0")

    for dim in DIMS:
        n_q = int(ceil(log2(dim)))
        v1  = generate_bipolar(dim, seed=VEC_SEED)
        v2  = v1.copy()

        enc1 = encode(v1, label="L")
        enc2 = encode(v2, label="R")
        cl   = classical_similarity(v1, v2)

        # Depth
        qc_swap, _ = build_swap_circuit(enc1, enc2)
        depth, two_q = (get_depths(qc_swap, hw_backend)
                        if use_hardware else (None, None))

        # Sim
        sim = run_swap_simulator(enc1, enc2, sim_backend, shots=shots)

        # Hardware
        hw_nm = hw_m3 = None
        if use_hardware:
            try:
                hw_nm, hw_m3 = run_swap_hardware(
                    enc1, enc2, hw_backend, sampler, shots=shots)
            except Exception as e:
                print(f"{RED}  HW error dim={dim}: {e}{RESET}")

        _row(dim, n_q, 0.0, cl, sim, hw_nm, hw_m3,
             depth or 0, two_q or 0)
        all_results.append(dict(
            dim=dim, n_qubits=n_q, noise=0.0,
            classical=cl, sim=sim, hw_no_m3=hw_nm, hw_m3=hw_m3,
            depth=depth or 0, two_q=two_q or 0
        ))

    # ══════════════════════════════════════════════════════════════
    # SECTION 2 — BIT-FLIP NOISE SWEEP
    # ══════════════════════════════════════════════════════════════
    for dim in DIMS:
        n_q = int(ceil(log2(dim)))
        v1  = generate_bipolar(dim, seed=VEC_SEED)

        _section(f"SECTION 2 — BIT-FLIP NOISE SWEEP  |  dim = {dim}  |  0% → 50%")

        for noise in NOISE_LEVELS:
            v2   = v1.copy() if noise == 0.0 else apply_bit_flips(v1, noise, FLIP_SEED)
            enc1 = encode(v1, label="L")
            enc2 = encode(v2, label="R")
            cl   = classical_similarity(v1, v2)

            # Depth (compute once per dim, reuse)
            qc_swap, _ = build_swap_circuit(enc1, enc2)
            if use_hardware and noise == 0.0:
                depth, two_q = get_depths(qc_swap, hw_backend)
            elif noise == 0.0:
                depth, two_q = 0, 0

            # Sim
            sim = run_swap_simulator(enc1, enc2, sim_backend, shots=shots)

            # Hardware
            hw_nm = hw_m3 = None
            if use_hardware:
                try:
                    hw_nm, hw_m3 = run_swap_hardware(
                        enc1, enc2, hw_backend, sampler, shots=shots)
                except Exception as e:
                    print(f"{RED}  HW error dim={dim} noise={noise:.0%}: {e}{RESET}")

            _row(dim, n_q, noise, cl, sim, hw_nm, hw_m3, depth, two_q)
            all_results.append(dict(
                dim=dim, n_qubits=n_q, noise=noise,
                classical=cl, sim=sim, hw_no_m3=hw_nm, hw_m3=hw_m3,
                depth=depth, two_q=two_q
            ))

    _summary(all_results, use_hardware)


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="SWAP test similarity benchmark — DiagonalGate encoding"
    )
    parser.add_argument("--sim-only", action="store_true",
                        help="Simulator only, skip hardware")
    parser.add_argument("--backend",  type=str, default="ibm_cleveland",
                        help="IBM Quantum backend (default: ibm_cleveland)")
    parser.add_argument("--shots",    type=int, default=2048,
                        help="Shots per circuit (default: 2048)")
    args  = parser.parse_args()
    SHOTS = args.shots

    run(
        use_hardware = not args.sim_only,
        backend_name = args.backend,
        shots        = SHOTS,
    )
