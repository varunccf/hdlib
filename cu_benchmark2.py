"""
comparison_table.py
===================
Runs Compute-Uncompute + Hadamard test for identical vectors (v1 == v2)
across dimensions 4, 8, 16, 32, 64.

Prints a clean comparison table with:
  - HW no M3  and  HW + M3  for both tests
  - Circuit depth and 2Q depth for both tests
  - Error vs expected (1.0 for CU, 1.0 for Hadamard)

Run:
    python comparison_table.py
    python comparison_table.py --backend ibm_fez
    python comparison_table.py --shots 4096
"""

import numpy as np
from math import ceil, log2
from typing import Optional, List, Tuple

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

RESET   = "\033[0m"; BOLD  = "\033[1m"; DIM = "\033[2m"
CYAN    = "\033[96m"; GREEN = "\033[92m"; YELLOW = "\033[93m"
RED     = "\033[91m"; BLUE  = "\033[94m"; MAGENTA = "\033[95m"

def _col(err: float) -> str:
    if err < 0.05:   return GREEN
    elif err < 0.15: return YELLOW
    else:            return RED


# ─────────────────────────────────────────────────────────────────────────────
# ENCODE
# ─────────────────────────────────────────────────────────────────────────────

def encode(vec: np.ndarray, label: str = "v") -> QuantumCircuit:
    vec = np.asarray(vec)
    n   = int(ceil(log2(len(vec))))
    if len(vec) < 2**n:
        vec = np.concatenate([vec, np.ones(2**n - len(vec))])
    gate = DiagonalGate(vec.tolist()); gate.label = label
    qc   = QuantumCircuit(n, name=label)
    qc.append(gate, range(n))
    return qc


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


def _apply_m3(counts, backend, shots, measured_qubits):
    mit   = M3Mitigation(backend)
    mit.cals_from_system(qubits=measured_qubits)
    probs = mit.apply_correction(counts, qubits=measured_qubits)
    return {s: int(round(max(0.0, p) * shots)) for s, p in probs.items()}


# ─────────────────────────────────────────────────────────────────────────────
# CIRCUIT DEPTH HELPER
# ─────────────────────────────────────────────────────────────────────────────

def _two_q_depth(tqc: QuantumCircuit) -> int:
    two_q = QuantumCircuit(tqc.num_qubits)
    for inst in tqc.data:
        if len(inst.qubits) == 2:
            two_q.append(inst.operation,
                         [tqc.find_bit(q).index for q in inst.qubits])
    return two_q.depth()


def get_depths(qc: QuantumCircuit, backend: Backend) -> Tuple[int, int]:
    """Transpile to HW and return (total_depth, two_q_depth)."""
    tqc = transpile(qc, backend, optimization_level=3)
    return tqc.depth(), _two_q_depth(tqc)


# ─────────────────────────────────────────────────────────────────────────────
# CU CIRCUIT + RUN
# ─────────────────────────────────────────────────────────────────────────────

def build_cu(enc_l: QuantumCircuit, enc_r: QuantumCircuit):
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


def run_cu(enc_l, enc_r, backend, sampler, shots) -> Tuple[float, float]:
    """Returns (hw_no_m3, hw_m3). Both are P('000...0') / shots."""
    qc, creg = build_cu(enc_l, enc_r)
    n        = enc_l.num_qubits
    target   = "0" * n

    tqc = transpile(qc, backend, optimization_level=3)
    sampler.options.dynamical_decoupling.enable        = True
    sampler.options.dynamical_decoupling.sequence_type = "XpXm"
    sampler.options.twirling.enable_gates              = True

    counts   = sampler.run([tqc], shots=shots).result()[0].data.c_meas.get_counts()
    hw_no_m3 = counts.get(target, 0) / shots

    phys     = _get_physical_qubits(tqc, creg)
    m3c      = _apply_m3(counts, backend, shots, phys)
    hw_m3    = m3c.get(target, 0) / shots

    return hw_no_m3, hw_m3


# ─────────────────────────────────────────────────────────────────────────────
# HADAMARD CIRCUIT + RUN
# ─────────────────────────────────────────────────────────────────────────────

def build_hadamard(enc_l: QuantumCircuit, enc_r: QuantumCircuit):
    """
    Hadamard test circuit:
        H(anc) → C-enc_L† → X(anc) → C-enc_R → H(anc) → Measure(anc)

    Measures Re(<L|R>) = P(0) - P(1).
    """
    n    = enc_l.num_qubits
    anc  = QuantumRegister(1, "anc")
    sys  = QuantumRegister(n, "sys")
    creg = ClassicalRegister(1, "c_had")
    qc   = QuantumCircuit(anc, sys, creg)

    qc.h(anc[0])
    qc.h(sys)
    qc.append(enc_l.inverse().to_gate(label="L†").control(1),
              [anc[0]] + list(sys))
    qc.x(anc[0])
    qc.append(enc_r.to_gate(label="R").control(1),
              [anc[0]] + list(sys))
    qc.h(anc[0])
    qc.h(sys)
    qc.measure(anc[0], creg[0])

    return qc, creg


def run_hadamard(enc_l, enc_r, backend, sampler, shots) -> Tuple[float, float]:
    """Returns (hw_no_m3, hw_m3). Both are P(0)-P(1)."""
    qc, creg = build_hadamard(enc_l, enc_r)

    tqc = transpile(qc, backend, optimization_level=3)
    sampler.options.dynamical_decoupling.enable        = True
    sampler.options.dynamical_decoupling.sequence_type = "XpXm"
    sampler.options.twirling.enable_gates              = True

    job      = sampler.run([tqc], shots=shots)
    bit_data = job.result()[0].data.c_had
    counts_1 = int(np.count_nonzero(bit_data.array == 1))
    counts   = {"0": shots - counts_1, "1": counts_1}

    def _sim(c): return c.get("0",0)/shots - c.get("1",0)/shots

    hw_no_m3 = _sim(counts)

    phys  = _get_physical_qubits(tqc, creg)
    m3c   = _apply_m3(counts, backend, shots, phys)
    hw_m3 = _sim(m3c)

    return hw_no_m3, hw_m3


# ─────────────────────────────────────────────────────────────────────────────
# PRINT TABLE
# ─────────────────────────────────────────────────────────────────────────────

def print_table(results: list):
    """
    results: list of dicts with keys:
        dim, n_qubits,
        cu_no_m3, cu_m3, cu_depth, cu_2q,
        had_no_m3, had_m3, had_depth, had_2q
    """
    W = 175
    print()
    print(f"{CYAN}{'═'*W}{RESET}")
    print(f"{BOLD}{CYAN}   CU vs HADAMARD — Identical vectors (v1 == v2) | Expected = 1.0{RESET}")
    print(f"{DIM}   IBM Cleveland | DD + Twirling ON | M3 readout mitigation{RESET}")
    print(f"{CYAN}{'═'*W}{RESET}")
    print()

    # Header row 1
    print(f"  {'':>5} │{'':>7} │"
          f"{BLUE}{'── COMPUTE-UNCOMPUTE  |<v1|v2>|²/N²  ──':^55}{RESET} │"
          f"{GREEN}{'── HADAMARD TEST  Re(<v1|v2>)  ─────────':^55}{RESET}")

    # Header row 2
    print(
        f"  {'Dim':>5} │{'Qubits':>7} │"
        f"{BLUE}{'HW no M3':>12} │{'HW + M3':>10} │{'Err no M3':>10} │{'Err M3':>8} │{'Depth':>7} │{'2QDepth':>8}{RESET} │"
        f"{GREEN}{'HW no M3':>12} │{'HW + M3':>10} │{'Err no M3':>10} │{'Err M3':>8} │{'Depth':>7} │{'2QDepth':>8}{RESET}"
    )
    print(f"  {'─'*W}")

    for r in results:
        dim = r['dim']
        n_q = r['n_qubits']

        # CU values
        cu_nm  = r.get('cu_no_m3')
        cu_m3  = r.get('cu_m3')
        cu_enm = abs(cu_nm  - 1.0) if cu_nm  is not None else None
        cu_em3 = abs(cu_m3  - 1.0) if cu_m3  is not None else None
        cu_d   = r.get('cu_depth',  'N/A')
        cu_2q  = r.get('cu_2q',     'N/A')

        # Hadamard values
        had_nm  = r.get('had_no_m3')
        had_m3  = r.get('had_m3')
        had_enm = abs(had_nm  - 1.0) if had_nm  is not None else None
        had_em3 = abs(had_m3  - 1.0) if had_m3  is not None else None
        had_d   = r.get('had_depth', 'N/A')
        had_2q  = r.get('had_2q',    'N/A')

        def _v(val):
            if val is None: return f"{DIM}{'N/A':>10}{RESET}"
            return f"{val:>10.5f}"

        def _e(err):
            if err is None: return f"{DIM}{'N/A':>10}{RESET}"
            return f"{_col(err)}{err:>10.5f}{RESET}"

        def _d(d):
            if isinstance(d, str): return f"{DIM}{d:>8}{RESET}"
            return f"{CYAN}{d:>8}{RESET}"

        print(
            f"  {dim:>5} │{n_q:>7} │"
            f"{BLUE}{_v(cu_nm)} │{_v(cu_m3)} │{_e(cu_enm)} │{_e(cu_em3)} │{_d(cu_d)} │{_d(cu_2q)}{RESET} │"
            f"{GREEN}{_v(had_nm)} │{_v(had_m3)} │{_e(had_enm)} │{_e(had_em3)} │{_d(had_d)} │{_d(had_2q)}{RESET}"
        )

    print(f"  {'─'*W}")
    print()
    print(f"{DIM}  Colour: {GREEN}green err<0.05{RESET}{DIM} | "
          f"{YELLOW}yellow 0.05–0.15{RESET}{DIM} | "
          f"{RED}red >0.15{RESET}")
    print(f"{DIM}  CU ground truth    = (dot(v1,v2)/N)²  → 1.0 for identical vectors{RESET}")
    print(f"{DIM}  Hadamard g.t.      = Re(<v1|v2>)       → 1.0 for identical vectors{RESET}")
    print(f"{DIM}  Err = |HW result - 1.0|{RESET}")
    print(f"{CYAN}{'═'*W}{RESET}\n")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

DIMS     = [4, 8, 16, 32, 64]
SHOTS    = 2048
VEC_SEED = 42


def run(backend_name: str = "ibm_cleveland", shots: int = SHOTS):

    print(f"\n{YELLOW}  Connecting to {backend_name} ...{RESET}")
    service    = QiskitRuntimeService()
    backend    = service.backend(backend_name)
    sampler    = Sampler(mode=backend)
    print(f"{GREEN}  Connected ✓  ({backend.name}){RESET}\n")

    results = []

    for dim in DIMS:
        n_q = int(ceil(log2(dim)))
        v   = np.random.default_rng(VEC_SEED).choice([-1, 1], size=dim)
        enc = encode(v, label="v")   # encode v once — both left and right are same

        print(f"  dim={dim} ({n_q} qubits) ...", end="", flush=True)

        r = {'dim': dim, 'n_qubits': n_q}

        # ── CU depth ─────────────────────────────────────────────
        qc_cu, _ = build_cu(enc, enc)
        cu_d, cu_2q = get_depths(qc_cu, backend)
        r['cu_depth'] = cu_d
        r['cu_2q']    = cu_2q

        # ── CU hardware run ───────────────────────────────────────
        try:
            cu_nm, cu_m3 = run_cu(enc, enc, backend, sampler, shots)
            r['cu_no_m3'] = cu_nm
            r['cu_m3']    = cu_m3
            print(f"  CU done (hw={cu_nm:.3f})", end="", flush=True)
        except Exception as e:
            print(f"\n{RED}  CU error dim={dim}: {e}{RESET}")

        # ── Hadamard depth ────────────────────────────────────────
        qc_had, _ = build_hadamard(enc, enc)
        had_d, had_2q = get_depths(qc_had, backend)
        r['had_depth'] = had_d
        r['had_2q']    = had_2q

        # ── Hadamard hardware run ─────────────────────────────────
        try:
            had_nm, had_m3 = run_hadamard(enc, enc, backend, sampler, shots)
            r['had_no_m3'] = had_nm
            r['had_m3']    = had_m3
            print(f"  HAD done (hw={had_nm:.3f})", end="", flush=True)
        except Exception as e:
            print(f"\n{RED}  HAD error dim={dim}: {e}{RESET}")

        print()
        results.append(r)

    print_table(results)


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="CU vs Hadamard comparison table — identical vectors")
    parser.add_argument("--backend", type=str, default="ibm_cleveland")
    parser.add_argument("--shots",   type=int, default=2048)
    args = parser.parse_args()
    SHOTS = args.shots
    run(backend_name=args.backend, shots=SHOTS)
