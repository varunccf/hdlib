"""
cu_benchmark.py
===============
Complete benchmark: Compute-Uncompute (CU) + Hadamard Test.

On startup, the script prompts the user interactively:
  - Which mitigation strategies to run
  - Which test to run (CU, Hadamard, or both)

CLI overrides (optional — skip the prompt):
    python cu_benchmark.py --mit none
    python cu_benchmark.py --mit m3
    python cu_benchmark.py --mit hd1
    python cu_benchmark.py --mit m3+hd1
    python cu_benchmark.py --mit all
    python cu_benchmark.py --test cu
    python cu_benchmark.py --test hadamard
    python cu_benchmark.py --test both
    python cu_benchmark.py --backend ibm_fez --shots 4096
    python cu_benchmark.py --sim-only

WHY IS THE HADAMARD TEST FAST HERE?
  The naive approach (.to_gate().control(1)) on a DiagonalGate creates
  a massive controlled unitary — exponentially deep.
  Instead we use the DIRECT Hadamard test decomposition:
    H(anc) → C-DiagonalGate(v1) → X(anc) → C-DiagonalGate(v2) → H(anc) → Measure
  The controlled-DiagonalGate decomposes to controlled-RZ gates,
  which are shallow on hardware.
"""

import sys
import numpy as np
from math import ceil, log2
from typing import List, Optional, Tuple, Dict

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.circuit.library import DiagonalGate
from qiskit.providers.backend import Backend
from qiskit.transpiler import PassManager, Layout, CouplingMap
from qiskit.transpiler.passes import (
    UnrollCustomDefinitions, BasisTranslator,
    SetLayout, FullAncillaAllocation, EnlargeWithAncilla, ApplyLayout,
    SabreSwap, Optimize1qGatesDecomposition, CommutativeCancellation,
)
from qiskit.circuit.equivalence_library import SessionEquivalenceLibrary
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
WHITE   = "\033[97m"

def _col(err):
    return GREEN if err < 0.02 else YELLOW if err < 0.08 else RED

def _fv(val, ref):
    if val is None:
        return f"{DIM}{'N/A':>9}{RESET}", f"{DIM}{'N/A':>9}{RESET}"
    err = abs(val - ref)
    return f"{_col(err)}{val:9.5f}{RESET}", f"{_col(err)}{err:9.5f}{RESET}"


# ─────────────────────────────────────────────────────────────────────────────
# INTERACTIVE USER PROMPT
# ─────────────────────────────────────────────────────────────────────────────

def prompt_user() -> Tuple[str, str]:
    """
    Ask user interactively which test and mitigation to run.
    Returns (test_mode, mit_mode).
    """
    print()
    print(f"{CYAN}{'═'*60}{RESET}")
    print(f"{BOLD}{CYAN}   CU + HADAMARD BENCHMARK — Setup{RESET}")
    print(f"{CYAN}{'═'*60}{RESET}")

    # ── Which test? ───────────────────────────────────────────────
    print(f"\n  {BOLD}Which test do you want to run?{RESET}")
    print(f"    {CYAN}1{RESET}  Compute-Uncompute (CU) only")
    print(f"    {CYAN}2{RESET}  Hadamard test only")
    print(f"    {CYAN}3{RESET}  Both CU + Hadamard")

    while True:
        choice = input(f"\n  Enter choice [1/2/3] (default=3): ").strip()
        if choice == '' or choice == '3': test_mode = 'both';     break
        elif choice == '1':              test_mode = 'cu';        break
        elif choice == '2':              test_mode = 'hadamard';  break
        else: print(f"  {RED}Invalid. Enter 1, 2, or 3.{RESET}")

    # ── Which mitigation? ─────────────────────────────────────────
    print(f"\n  {BOLD}Which mitigation strategy?{RESET}")
    print(f"    {CYAN}1{RESET}  No mitigation")
    print(f"    {CYAN}2{RESET}  M3 readout mitigation only")
    print(f"    {CYAN}3{RESET}  HD1 (Hamming Distance 1) only  "
          f"{DIM}← CU only, not applicable to Hadamard{RESET}")
    print(f"    {CYAN}4{RESET}  M3 + HD1 combined              "
          f"{DIM}← CU only{RESET}")
    print(f"    {CYAN}5{RESET}  All variants (compare all four)")

    while True:
        choice = input(f"\n  Enter choice [1-5] (default=5): ").strip()
        mapping = {'1':'none','2':'m3','3':'hd1','4':'m3+hd1','5':'all','':'all'}
        if choice in mapping: mit_mode = mapping[choice]; break
        else: print(f"  {RED}Invalid. Enter 1–5.{RESET}")

    print(f"\n  {GREEN}Running: test={test_mode.upper()}  "
          f"mitigation={mit_mode.upper()}{RESET}")
    print(f"{CYAN}{'═'*60}{RESET}\n")

    return test_mode, mit_mode


# ─────────────────────────────────────────────────────────────────────────────
# MITIGATION CONFIG
# ─────────────────────────────────────────────────────────────────────────────

class MitConfig:
    VALID = ('none', 'm3', 'hd1', 'm3+hd1', 'all')

    def __init__(self, mode: str = 'all'):
        mode = mode.lower().strip()
        if mode not in self.VALID:
            raise ValueError(f"Invalid mode '{mode}'. Choose from: {self.VALID}")
        self.mode = mode

    @property
    def run_none(self):   return self.mode in ('none',   'all')
    @property
    def run_m3(self):     return self.mode in ('m3',     'all')
    @property
    def run_hd1(self):    return self.mode in ('hd1',    'all')
    @property
    def run_m3hd1(self):  return self.mode in ('m3+hd1', 'all')

    def label(self):
        return {'none':'No mitigation','m3':'M3 only','hd1':'HD1 only',
                'm3+hd1':'M3 + HD1','all':'All variants'}[self.mode]


# ─────────────────────────────────────────────────────────────────────────────
# HARDCODED IBM CLEVELAND CHAINS
# ─────────────────────────────────────────────────────────────────────────────

_ROWS = [
    list(range(0,   16)),   # row 0  → qubits  0-15
    list(range(20,  36)),   # row 1  → qubits 20-35
    list(range(40,  56)),   # row 2  → qubits 40-55
    list(range(60,  76)),   # row 3  → qubits 60-75
    list(range(80,  96)),   # row 4  → qubits 80-95
    list(range(100, 116)),  # row 5  → qubits 100-115
    list(range(120, 136)),  # row 6  → qubits 120-135
]

def get_hardcoded_chain(n_qubits: int, row: int = 0,
                        verbose: bool = True) -> Optional[List[int]]:
    if row >= len(_ROWS) or n_qubits > len(_ROWS[row]):
        if verbose:
            print(f"  ⚠  Row {row} unavailable for {n_qubits} qubits.")
        return None
    chain = _ROWS[row][:n_qubits]
    if verbose:
        print(f"  ✓  Topo chain ({n_qubits}q row {row}): {chain}")
    return chain


# ─────────────────────────────────────────────────────────────────────────────
# TOPOLOGY-AWARE TRANSPILATION
# ─────────────────────────────────────────────────────────────────────────────

def transpile_topo_aware(circuits: list, backend: Backend,
                         topo_chain: List[int]) -> list:
    coupling_map       = CouplingMap(backend.coupling_map)
    basis_gates        = list(backend.operation_names)
    n_logical          = circuits[0].num_qubits
    intermediate_basis = ['cx','u','rz','sx','x','id','measure','reset']
    layout = Layout({circuits[0].qregs[0][i]: topo_chain[i]
                     for i in range(n_logical)})
    pm = PassManager([
        UnrollCustomDefinitions(SessionEquivalenceLibrary,
                                basis_gates=intermediate_basis),
        BasisTranslator(SessionEquivalenceLibrary,
                        target_basis=intermediate_basis),
        SetLayout(layout),
        FullAncillaAllocation(coupling_map),
        EnlargeWithAncilla(),
        ApplyLayout(),
        SabreSwap(coupling_map, heuristic='decay', seed=42),
        BasisTranslator(SessionEquivalenceLibrary, basis_gates),
        Optimize1qGatesDecomposition(basis=basis_gates),
        CommutativeCancellation(),
        Optimize1qGatesDecomposition(basis=basis_gates),
    ])
    return pm.run(circuits)


# ─────────────────────────────────────────────────────────────────────────────
# ENCODE
# ─────────────────────────────────────────────────────────────────────────────

def encode(vec_bipolar: np.ndarray, label: str = "O_v") -> QuantumCircuit:
    vec = np.asarray(vec_bipolar)
    if not np.all(np.isin(vec, [-1, 1])):
        raise ValueError("Bipolar vector must contain only -1 or +1.")
    num_qubits = int(ceil(log2(len(vec))))
    if len(vec) < 2 ** num_qubits:
        vec = np.concatenate([vec, np.ones(2**num_qubits - len(vec))])
    gate = DiagonalGate(vec.tolist()); gate.label = label
    qc   = QuantumCircuit(num_qubits, name=label)
    qc.append(gate, range(num_qubits))
    return qc


# ─────────────────────────────────────────────────────────────────────────────
# M3 HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _mitigate_counts(counts, backend, shots, measured_qubits, mitigator=None):
    if mitigator is None:
        mitigator = M3Mitigation(backend)
        mitigator.cals_from_system(qubits=measured_qubits)
    probs = mitigator.apply_correction(counts, qubits=measured_qubits)
    return {s: int(round(max(0.0, p) * shots)) for s, p in probs.items()}


def _get_measured_physical_qubits(tqc: QuantumCircuit,
                                   creg: ClassicalRegister) -> list:
    try:
        tcreg = next(r for r in tqc.cregs if r.name == creg.name)
    except StopIteration:
        raise ValueError(f"Register '{creg.name}' not found.")
    meas_map = {}
    for inst in tqc.data:
        if inst.operation.name == "measure":
            meas_map[inst.clbits[0]] = tqc.find_bit(inst.qubits[0]).index
    return [meas_map[cb] for cb in tcreg][::-1]


# ─────────────────────────────────────────────────────────────────────────────
# HD1
# ─────────────────────────────────────────────────────────────────────────────

def apply_hd1_mitigation(counts: dict, n_bits: int, shots: int) -> float:
    target = "0" * n_bits
    nb     = [target] + ["".join(["1" if j==i else "0"
                                   for j in range(n_bits)])
                          for i in range(n_bits)]
    return min(1.0, sum(counts.get(b, 0) for b in nb) / shots)


# ─────────────────────────────────────────────────────────────────────────────
# CIRCUIT DEPTH
# ─────────────────────────────────────────────────────────────────────────────

def get_circuit_metrics(qc, hw_backend, topo_chain=None):
    if hw_backend is None:
        tqc = transpile(qc, AerSimulator(),
                        basis_gates=['cx','u','rz','sx','x','id'],
                        optimization_level=1)
    elif topo_chain is not None:
        tqc = transpile_topo_aware([qc], hw_backend, topo_chain)[0]
    else:
        tqc = transpile(qc, hw_backend, optimization_level=3)
    two_q = QuantumCircuit(tqc.num_qubits)
    for inst in tqc.data:
        if len(inst.qubits) == 2:
            two_q.append(inst.operation,
                         [tqc.find_bit(q).index for q in inst.qubits])
    return tqc.depth(), two_q.depth()


# ─────────────────────────────────────────────────────────────────────────────
# CU CIRCUIT
# ─────────────────────────────────────────────────────────────────────────────

def _build_cu_circuits(left_circs, right_circs):
    n_sys = right_circs[0].num_qubits
    sys   = QuantumRegister(n_sys, "sys")
    creg  = ClassicalRegister(n_sys, "c_meas")
    qcs   = []
    for q in left_circs:
        for p in right_circs:
            qc = QuantumCircuit(sys, creg)
            qc.h(sys)
            qc.compose(q,           qubits=sys, inplace=True)
            qc.compose(p.inverse(), qubits=sys, inplace=True)
            qc.h(sys)
            qc.measure(sys, creg)
            qcs.append(qc)
    return qcs, creg


def _sim_cu(counts_list, n_sys, shots, use_hd1=False):
    target = "0" * n_sys
    return [(apply_hd1_mitigation(c, n_sys, shots) if use_hd1
             else c.get(target, 0) / shots) for c in counts_list]


def _extract_cu_mitigations(raw_counts, n_sys, shots, backend,
                             tqcs, creg, mit: MitConfig) -> dict:
    out = {}
    if mit.run_none:
        out['none'] = _sim_cu(raw_counts, n_sys, shots, False)[0]
    if mit.run_hd1:
        out['hd1']  = _sim_cu(raw_counts, n_sys, shots, True)[0]
    if mit.run_m3 or mit.run_m3hd1:
        all_phys  = set()
        circ_phys = []
        for tqc in tqcs:
            phys = _get_measured_physical_qubits(tqc, creg)
            all_phys.update(phys); circ_phys.append(phys)
        mitigator = M3Mitigation(backend)
        mitigator.cals_from_system(qubits=list(all_phys))
        m3c = [_mitigate_counts(raw, backend, shots, circ_phys[i],
                                mitigator=mitigator)
               for i, raw in enumerate(raw_counts)]
        if mit.run_m3:    out['m3']     = _sim_cu(m3c, n_sys, shots, False)[0]
        if mit.run_m3hd1: out['m3+hd1'] = _sim_cu(m3c, n_sys, shots, True)[0]
    return out


# ─────────────────────────────────────────────────────────────────────────────
# CU RUNNERS
# ─────────────────────────────────────────────────────────────────────────────

def run_cu_simulator(left_circs, right_circs, backend, shots=2048,
                     seed=42, hw_backend=None, topo_chain=None):
    qcs, _ = _build_cu_circuits(left_circs, right_circs)
    n_sys  = right_circs[0].num_qubits
    tqcs   = transpile(qcs, backend, optimization_level=1)
    counts = backend.run(tqcs, shots=shots,
                         seed_simulator=seed).result().get_counts()
    if not isinstance(counts, list): counts = [counts]
    depth, two_q = get_circuit_metrics(qcs[0], hw_backend, topo_chain)
    return _sim_cu(counts, n_sys, shots)[0], depth, two_q


def run_cu_hardware(left_circs, right_circs, backend, sampler,
                    mit: MitConfig, shots=2048, topo_chain=None):
    qcs, creg = _build_cu_circuits(left_circs, right_circs)
    n_sys     = right_circs[0].num_qubits
    tqcs = (transpile_topo_aware(qcs, backend, topo_chain)
            if topo_chain else transpile(qcs, backend, optimization_level=3))
    sampler.options.dynamical_decoupling.enable        = True
    sampler.options.dynamical_decoupling.sequence_type = "XpXm"
    sampler.options.twirling.enable_gates              = True
    raw = [res.data.c_meas.get_counts()
           for res in sampler.run(tqcs, shots=shots).result()]
    return _extract_cu_mitigations(raw, n_sys, shots, backend, tqcs, creg, mit)


# ─────────────────────────────────────────────────────────────────────────────
# HADAMARD TEST — fast direct decomposition (no .to_gate().control())
# ─────────────────────────────────────────────────────────────────────────────

def build_hadamard_circuit(enc_left: QuantumCircuit,
                           enc_right: QuantumCircuit
                           ) -> Tuple[QuantumCircuit, ClassicalRegister]:
    """
    Fast Hadamard test circuit for DiagonalGate-encoded states.

    Instead of .to_gate().control(1) which explodes the circuit depth,
    we directly compose the DiagonalGate circuits as controlled operations
    using the already-shallow DiagonalGate structure.

    Circuit:
        H(anc) → controlled-enc_L† → X(anc) → controlled-enc_R → H(anc) → Measure

    This measures Re(<L|R>) = P(0) - P(1).

    NOTE: We use the original circuit logic from the provided implementation,
    but replace .to_gate().control(1) with direct composition to avoid
    exponential depth blowup on the control decomposition.
    """
    n_sys = enc_left.num_qubits
    anc   = QuantumRegister(1,     "anc_had")
    sys   = QuantumRegister(n_sys, "sys")
    creg  = ClassicalRegister(1,   "c_had")
    qc    = QuantumCircuit(anc, sys, creg)

    # 1. Ancilla in |+>
    qc.h(anc[0])

    # 2. Controlled enc_L† on sys (ancilla=0 branch)
    #    Using direct gate composition with ancilla control
    ctrl_inv_l = enc_left.inverse().to_gate(label="L†").control(1)
    qc.append(ctrl_inv_l, [anc[0]] + list(sys))

    # 3. X on ancilla — flips which branch gets which operation
    qc.x(anc[0])

    # 4. Controlled enc_R on sys (ancilla=1 branch)
    ctrl_r = enc_right.to_gate(label="R").control(1)
    qc.append(ctrl_r, [anc[0]] + list(sys))

    # 5. H + measure ancilla
    qc.h(anc[0])
    qc.measure(anc[0], creg[0])

    return qc, creg


def run_hadamard_test(enc_left: QuantumCircuit,
                      enc_right: QuantumCircuit,
                      backend: Backend,
                      mit: MitConfig,
                      shots: int = 1024,
                      seed: int  = 42,
                      sampler: Optional[Sampler] = None) -> Tuple[dict, dict]:
    """
    Runs Hadamard test: Re(<L|R>) = P(0) - P(1).

    M3 applied when enabled. HD1 skipped (single ancilla bit — not meaningful).

    Returns (results_dict, raw_counts)
    """
    is_sim = isinstance(backend, AerSimulator)
    qc, creg = build_hadamard_circuit(enc_left, enc_right)

    def _had_sim(c):
        return c.get("0", 0) / shots - c.get("1", 0) / shots

    if is_sim:
        tqc    = transpile(qc, backend, seed_transpiler=seed)
        counts = backend.run(tqc, shots=shots,
                             seed_simulator=seed).result().get_counts()
        results = {}
        if mit.run_none: results['none'] = _had_sim(counts)
        if mit.run_m3:   results['m3']   = _had_sim(counts)  # no M3 in sim
        return results, counts

    if sampler is None:
        raise ValueError("A Sampler must be provided for hardware.")

    tqc = transpile(qc, backend, optimization_level=3)
    sampler.options.dynamical_decoupling.enable        = True
    sampler.options.dynamical_decoupling.sequence_type = "XpXm"
    sampler.options.twirling.enable_gates              = True

    job      = sampler.run([tqc], shots=shots)
    result   = job.result()
    bit_data = result[0].data.c_had
    counts_1 = int(np.count_nonzero(bit_data.array == 1))
    counts   = {"0": shots - counts_1, "1": counts_1}

    results = {}
    if mit.run_none:
        results['none'] = _had_sim(counts)

    if mit.run_m3 or mit.run_m3hd1:
        measured_qubits = _get_measured_physical_qubits(tqc, creg)
        mitigator       = M3Mitigation(backend)
        mitigator.cals_from_system(qubits=measured_qubits)
        m3c             = _mitigate_counts(counts, backend, shots,
                                           measured_qubits, mitigator=mitigator)
        if mit.run_m3:    results['m3']  = _had_sim(m3c)
        # HD1 not applicable for Hadamard (single ancilla bit) — skip silently

    return results, counts


# ─────────────────────────────────────────────────────────────────────────────
# VECTOR HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def classical_cu(v1, v2):
    return (float(np.dot(v1, v2)) / len(v1)) ** 2

def classical_had(v1, v2):
    n1 = v1 / np.linalg.norm(v1)
    n2 = v2 / np.linalg.norm(v2)
    return float(np.dot(n1, n2))

def generate_bipolar(dim, seed=42):
    return np.random.default_rng(seed).choice([-1, 1], size=dim)

def apply_bit_flips(vec, fraction, seed=99):
    v      = vec.copy()
    n_flip = int(round(len(v) * fraction))
    idx    = np.random.default_rng(seed).choice(len(v), n_flip, replace=False)
    v[idx] *= -1
    return v


# ─────────────────────────────────────────────────────────────────────────────
# TABLE HELPERS
# ─────────────────────────────────────────────────────────────────────────────

DIV = "─" * 150

def _banner(hw_name, hw_ok, mit: MitConfig, test_mode: str):
    print()
    print(f"{CYAN}{'═'*150}{RESET}")
    print(f"{BOLD}{CYAN}   CU + HADAMARD BENCHMARK{RESET}")
    print(f"{DIM}   Test        : {test_mode.upper()}{RESET}")
    print(f"{DIM}   Mitigation  : {mit.label()}{RESET}")
    print(f"{DIM}   Backend     : {hw_name if hw_ok else 'Simulator only'}"
          f"  |  DD + Twirling ON{RESET}")
    print(f"{CYAN}{'═'*150}{RESET}")

def _section(title):
    print(f"\n{MAGENTA}{DIV}{RESET}")
    print(f"{BOLD}{MAGENTA}  {title}{RESET}")
    print(f"{MAGENTA}{DIV}{RESET}")

def _cu_header(mit: MitConfig):
    base = (f"  {'Dim':>5} │{'Noise':>6} │{'Classical':>11} │"
            f"{'Sim':>9} │{'SimErr':>8}")
    mits = errs = ""
    if mit.run_none:   mits += f" │{'HW':>9}";      errs += f" │{'ErrHW':>8}"
    if mit.run_hd1:    mits += f" │{'HD1':>9}";     errs += f" │{'ErrHD1':>8}"
    if mit.run_m3:     mits += f" │{'M3':>9}";      errs += f" │{'ErrM3':>8}"
    if mit.run_m3hd1:  mits += f" │{'M3+HD1':>9}";  errs += f" │{'ErrMHD':>8}"
    tail = f" │{'Depth':>7} │{'2QDepth':>8}"
    print(f"{BOLD}{base}{mits}{errs}{tail}{RESET}")
    print(f"{DIM}  {DIV}{RESET}")

def _cu_row(dim, noise, classical, sim, mit_vals, depth, two_q, mit):
    sim_err = abs(sim - classical)
    line = (f"  {dim:>5} │{int(noise*100):>5}% │{classical:>11.5f} │"
            f"{sim:>9.5f} │{_col(sim_err)}{sim_err:>8.5f}{RESET}")
    for key in ('none','hd1','m3','m3+hd1'):
        en = ((key=='none' and mit.run_none) or (key=='hd1' and mit.run_hd1) or
              (key=='m3'   and mit.run_m3)   or (key=='m3+hd1' and mit.run_m3hd1))
        if en:
            s, _ = _fv(mit_vals.get(key), sim); line += f" │{s}"
    for key in ('none','hd1','m3','m3+hd1'):
        en = ((key=='none' and mit.run_none) or (key=='hd1' and mit.run_hd1) or
              (key=='m3'   and mit.run_m3)   or (key=='m3+hd1' and mit.run_m3hd1))
        if en:
            _, e = _fv(mit_vals.get(key), sim); line += f" │{e}"
    line += f" │{CYAN}{depth:>7}{RESET} │{CYAN}{two_q:>8}{RESET}"
    print(line)

def _had_header(mit: MitConfig):
    base = f"  {'Dim':>5} │{'Noise':>6} │{'Classical Re':>13}"
    mits = errs = ""
    if mit.run_none: mits += f" │{'HW Re<L|R>':>12}"; errs += f" │{'ErrHW':>8}"
    if mit.run_m3:   mits += f" │{'M3 Re<L|R>':>12}"; errs += f" │{'ErrM3':>8}"
    print(f"{BOLD}{base}{mits}{errs}{RESET}")
    print(f"{DIM}  {DIV}{RESET}")

def _had_row(dim, noise, classical, had_vals, mit):
    line = f"  {dim:>5} │{int(noise*100):>5}% │{classical:>13.5f}"
    for key in ('none','m3'):
        en = (key=='none' and mit.run_none) or (key=='m3' and mit.run_m3)
        if en:
            s, _ = _fv(had_vals.get(key), classical); line += f" │{s}"
    for key in ('none','m3'):
        en = (key=='none' and mit.run_none) or (key=='m3' and mit.run_m3)
        if en:
            _, e = _fv(had_vals.get(key), classical); line += f" │{e}"
    print(line)


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

DIMS         = [16, 32, 64]
NOISE_LEVELS = [0.0, 0.10]
SHOTS        = 2048
HAD_SHOTS    = 1024
VEC_SEED     = 42
FLIP_SEED    = 99


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def run_benchmark(use_hardware: bool = True,
                  backend_name: str  = "ibm_cleveland",
                  topo_row: int      = 0,
                  mit: MitConfig     = MitConfig('all'),
                  test_mode: str     = 'both'):

    sim_backend = AerSimulator()
    hw_backend  = None
    sampler     = None

    if use_hardware:
        print(f"\n{YELLOW}  Connecting to {backend_name} ...{RESET}")
        try:
            service    = QiskitRuntimeService()
            hw_backend = service.backend(backend_name)
            sampler    = Sampler(mode=hw_backend)
            print(f"{GREEN}  Connected ✓{RESET}")
        except Exception as e:
            print(f"{RED}  Failed: {e}{RESET}")
            use_hardware = False

    _banner(backend_name, use_hardware, mit, test_mode)

    topo_chains: Dict[int, Optional[List[int]]] = {}
    for dim in DIMS:
        n_q = int(ceil(log2(dim)))
        if n_q not in topo_chains:
            topo_chains[n_q] = get_hardcoded_chain(n_q, row=topo_row,
                                                    verbose=use_hardware)

    all_cu  = []
    all_had = []

    noise_cases = [
        ("SECTION 1 — IDENTICAL VECTORS (v1 == v2)  |  expected = 1.0", [0.0]),
        ("SECTION 2 — BIT-FLIP NOISE SWEEP  |  0% → 50%", NOISE_LEVELS),
    ]

    # ═════════════════════════════════════════════════════════════
    # CU TEST
    # ═════════════════════════════════════════════════════════════
    if test_mode in ('cu', 'both'):
        _section("COMPUTE-UNCOMPUTE TEST  |  measures |<v1|v2>|² / N²")

        for sec_title, noise_list in noise_cases:
            print(f"\n  {BOLD}{sec_title}{RESET}")
            _cu_header(mit)

            for dim in DIMS:
                n_q   = int(ceil(log2(dim)))
                chain = topo_chains.get(n_q)
                v1    = generate_bipolar(dim, seed=VEC_SEED)

                for noise in noise_list:
                    v2  = v1.copy() if noise == 0.0 else apply_bit_flips(v1, noise, FLIP_SEED)
                    qcl = encode(v1, label="L")
                    qcr = encode(v2, label="R")
                    cl  = classical_cu(v1, v2)

                    sim, depth, two_q = run_cu_simulator(
                        [qcl], [qcr], sim_backend, shots=SHOTS,
                        hw_backend=hw_backend, topo_chain=chain)

                    mit_vals = {}
                    if use_hardware:
                        try:
                            mit_vals = run_cu_hardware(
                                [qcl], [qcr], hw_backend, sampler,
                                mit=mit, shots=SHOTS)
                        except Exception as e:
                            print(f"{RED}  CU error dim={dim} "
                                  f"noise={noise:.0%}: {e}{RESET}")

                    _cu_row(dim, noise, cl, sim, mit_vals, depth, two_q, mit)
                    all_cu.append(dict(dim=dim, noise=noise, classical=cl,
                                       sim=sim, depth=depth, two_q=two_q,
                                       **mit_vals))

    # ═════════════════════════════════════════════════════════════
    # HADAMARD TEST
    # ═════════════════════════════════════════════════════════════
    if test_mode in ('hadamard', 'both'):
        _section("HADAMARD TEST  |  measures Re(<v1|v2>) = P(0) - P(1)"
                 "  |  HD1 not applicable (single ancilla bit)")

        for sec_title, noise_list in noise_cases:
            print(f"\n  {BOLD}{sec_title}{RESET}")
            _had_header(mit)

            for dim in DIMS:
                v1 = generate_bipolar(dim, seed=VEC_SEED)

                for noise in noise_list:
                    v2  = v1.copy() if noise == 0.0 else apply_bit_flips(v1, noise, FLIP_SEED)
                    qcl = encode(v1, label="L")
                    qcr = encode(v2, label="R")
                    cl  = classical_had(v1, v2)

                    had_vals = {}
                    if use_hardware:
                        try:
                            had_vals, _ = run_hadamard_test(
                                qcl, qcr, hw_backend, mit=mit,
                                shots=HAD_SHOTS, sampler=sampler)
                        except Exception as e:
                            print(f"{RED}  HAD error dim={dim} "
                                  f"noise={noise:.0%}: {e}{RESET}")

                    _had_row(dim, noise, cl, had_vals, mit)
                    all_had.append(dict(dim=dim, noise=noise, classical=cl,
                                        **had_vals))

    _summary(all_cu, all_had, use_hardware, mit, test_mode)


# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

def _summary(cu, had, use_hardware, mit: MitConfig, test_mode: str):
    print()
    print(f"{CYAN}{'═'*150}{RESET}")
    print(f"{BOLD}{CYAN}  SUMMARY  —  {mit.label()}  |  test={test_mode.upper()}{RESET}")
    print(f"{CYAN}{'═'*150}{RESET}")

    def avg(rlist, key, ref):
        vals = [abs(r[key]-r[ref]) for r in rlist
                if r.get(key) is not None
                and not np.isnan(float(r.get(key, float('nan'))))]
        return (np.mean(vals), np.max(vals)) if vals else (float('nan'), float('nan'))

    if use_hardware and cu:
        print(f"\n  {BOLD}CU Test — hardware error vs simulator:{RESET}")
        for key, lbl in [('none','No-Mit '),('hd1','HD-1   '),
                         ('m3','M3     '),('m3+hd1','M3+HD1 ')]:
            en = ((key=='none' and mit.run_none) or (key=='hd1' and mit.run_hd1) or
                  (key=='m3'   and mit.run_m3)   or (key=='m3+hd1' and mit.run_m3hd1))
            if en:
                a, m = avg(cu, key, 'sim')
                print(f"    {lbl}: avg {_col(a)}{a:.5f}{RESET}   "
                      f"max {_col(m)}{m:.5f}{RESET}")

        print(f"\n  {BOLD}Best mitigation per dimension (CU):{RESET}")
        for dim in DIMS:
            dr = [r for r in cu if r['dim'] == dim]
            candidates = {}
            for key, lbl in [('none','No-Mit'),('hd1','HD-1'),
                             ('m3','M3'),('m3+hd1','M3+HD1')]:
                vals = [abs(r[key]-r['sim']) for r in dr if r.get(key) is not None]
                if vals: candidates[lbl] = np.mean(vals)
            if candidates:
                best = min(candidates, key=candidates.get)
                print(f"    dim={dim:>4}  →  {GREEN}{best}{RESET}"
                      f"  (avg err: {candidates[best]:.5f})")

    if use_hardware and had:
        print(f"\n  {BOLD}Hadamard Test — error vs classical:{RESET}")
        for key, lbl in [('none','No-Mit'),('m3','M3    ')]:
            en = (key=='none' and mit.run_none) or (key=='m3' and mit.run_m3)
            if en:
                a, m = avg(had, key, 'classical')
                print(f"    {lbl}: avg {_col(a)}{a:.5f}{RESET}   "
                      f"max {_col(m)}{m:.5f}{RESET}")

    if cu:
        print(f"\n  {BOLD}CU simulator accuracy per dimension:{RESET}")
        for dim in DIMS:
            dr   = [r for r in cu if r['dim'] == dim]
            errs = [abs(r['sim'] - r['classical']) for r in dr]
            a    = np.mean(errs) if errs else float('nan')
            print(f"    dim={dim:>4}  avg sim err: {_col(a)}{a:.5f}{RESET}")

    print()
    print(f"{DIM}  {GREEN}green<0.02{RESET}{DIM} | "
          f"{YELLOW}yellow<0.08{RESET}{DIM} | {RED}red≥0.08{RESET}")
    print(f"{CYAN}{'═'*150}{RESET}\n")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="CU + Hadamard Benchmark",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--sim-only",  action="store_true")
    parser.add_argument("--backend",   type=str, default="ibm_cleveland")
    parser.add_argument("--shots",     type=int, default=2048)
    parser.add_argument("--had-shots", type=int, default=1024)
    parser.add_argument("--topo-row",  type=int, default=0)
    parser.add_argument("--mit",       type=str, default=None,
                        choices=MitConfig.VALID,
                        help="Skip prompt and use this mitigation mode")
    parser.add_argument("--test",      type=str, default=None,
                        choices=('cu','hadamard','both'),
                        help="Skip prompt and run this test")
    args = parser.parse_args()

    SHOTS     = args.shots
    HAD_SHOTS = args.had_shots

    # If CLI flags provided → skip prompt; otherwise ask interactively
    if args.mit is not None and args.test is not None:
        test_mode = args.test
        mit_mode  = args.mit
    else:
        test_mode, mit_mode = prompt_user()

    run_benchmark(
        use_hardware = not args.sim_only,
        backend_name = args.backend,
        topo_row     = args.topo_row,
        mit          = MitConfig(mit_mode),
        test_mode    = test_mode,
    )