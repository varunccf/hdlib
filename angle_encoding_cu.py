"""
angle_encoding_cu.py
====================
Angle encoding + Compute-Uncompute for dim=64 and dim=128.
Hardware only — simulator skipped (too many qubits for AerSimulator).

Angle encoding:
    v[i] ∈ {-1,+1}  →  RY(arccos(v[i])) on qubit i
    N elements → N qubits, depth = 1, ZERO 2Q gates

CU circuit:
    H → RY(v1) → RY(v2)† → H → Measure
    P("000...0") = |<v1|v2>|² / N²

Tests:
    1. Identical vectors (v1 == v2)   → expected ≈ 1.0
    2. 50% bit flips                  → expected ≈ 0.0
"""

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_ibm_runtime import QiskitRuntimeService, Sampler

QiskitRuntimeService.save_account(
    channel="ibm_cloud",
    token="v2WGCXrHMvu1Nh2tzr39Mocon2npcG_ogKxvgvKtTyg2",
    instance="crn:v1:bluemix:public:quantum-computing:us-east:a/813b37ffee14414ca81092ab94341434:1284900f-4e18-41c7-aadf-44278c5d44da::",
    set_as_default=True,
    overwrite=True,
)

SHOTS = 2048
DIMS  = [64, 128]


# ─────────────────────────────────────────────────────────────────────────────
# ENCODING
# ─────────────────────────────────────────────────────────────────────────────

def encode_angle(vec: np.ndarray, label: str = "enc") -> QuantumCircuit:
    """
    Angle encoding: v[i] → RY(arccos(v[i])) on qubit i.
    All gates are single-qubit and fully parallel → depth = 1, 0 CNOTs.
    """
    N  = len(vec)
    qc = QuantumCircuit(N, name=label)
    for i, v in enumerate(vec):
        theta = float(np.arccos(np.clip(v, -1.0, 1.0)))
        if abs(theta) > 1e-9:
            qc.ry(theta, i)
    return qc


def build_cu_circuit(qc_left: QuantumCircuit,
                     qc_right: QuantumCircuit) -> QuantumCircuit:
    """H → encode(v1) → encode(v2)† → H → Measure"""
    n    = qc_left.num_qubits
    sys  = QuantumRegister(n, "sys")
    creg = ClassicalRegister(n, "c")
    qc   = QuantumCircuit(sys, creg)
    qc.h(sys)
    qc.compose(qc_left,           qubits=sys, inplace=True)
    qc.compose(qc_right.inverse(), qubits=sys, inplace=True)
    qc.h(sys)
    qc.measure(sys, creg)
    return qc


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def similarity(counts: dict, n: int, shots: int) -> float:
    return counts.get("0" * n, 0) / shots


def classical_ground_truth(v1: np.ndarray, v2: np.ndarray) -> float:
    return (float(np.dot(v1, v2)) / len(v1)) ** 2


def generate_bipolar(dim: int, seed: int = 42) -> np.ndarray:
    return np.random.default_rng(seed).choice([-1, 1], size=dim)


def apply_bit_flips(vec: np.ndarray, fraction: float,
                    seed: int = 99) -> np.ndarray:
    v      = vec.copy()
    n_flip = int(round(len(v) * fraction))
    idx    = np.random.default_rng(seed).choice(len(v), n_flip, replace=False)
    v[idx] *= -1
    return v


def get_circuit_depths(qc: QuantumCircuit, backend) -> tuple:
    """
    Returns (raw_depth, raw_2q, hw_depth, hw_2q_depth)
    after transpiling to hardware backend.
    """
    raw_depth = qc.depth()
    raw_2q    = sum(1 for inst in qc.data if len(inst.qubits) == 2)

    tqc = transpile(qc, backend, optimization_level=3)
    hw_depth = tqc.depth()

    two_q_circ = QuantumCircuit(tqc.num_qubits)
    for inst in tqc.data:
        if len(inst.qubits) == 2:
            idxs = [tqc.find_bit(q).index for q in inst.qubits]
            two_q_circ.append(inst.operation, idxs)
    hw_2q = two_q_circ.depth()

    return raw_depth, raw_2q, hw_depth, hw_2q


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def run():
    print("\nConnecting to IBM Quantum ...")
    service    = QiskitRuntimeService()
    backend    = service.backend("ibm_cleveland")
    sampler    = Sampler(mode=backend)
    sampler.options.dynamical_decoupling.enable        = True
    sampler.options.dynamical_decoupling.sequence_type = "XpXm"
    sampler.options.twirling.enable_gates              = True
    print(f"Connected ✓  ({backend.name}  |  {backend.num_qubits} qubits)\n")

    print("=" * 80)
    print("  ANGLE ENCODING — Compute-Uncompute  (hardware only)")
    print("  N elements → N qubits | depth = O(1) | ZERO 2Q gates")
    print("=" * 80)

    all_results = []

    for dim in DIMS:
        v1       = generate_bipolar(dim, seed=42)
        v2_same  = v1.copy()
        v2_flip  = apply_bit_flips(v1, 0.5, seed=99)

        cases = [
            ("identical  (0% flip)",  v1, v2_same),
            ("orthogonal (50% flip)", v1, v2_flip),
        ]

        print(f"\n{'─'*80}")
        print(f"  dim = {dim}  |  {dim} qubits")
        print(f"{'─'*80}")

        for case_name, va, vb in cases:
            classical = classical_ground_truth(va, vb)

            qc_a = encode_angle(va, label="L")
            qc_b = encode_angle(vb, label="R")
            qc   = build_cu_circuit(qc_a, qc_b)

            raw_d, raw_2q, hw_d, hw_2q = get_circuit_depths(qc, backend)

            print(f"\n  [{case_name}]")
            print(f"    Classical ground truth : {classical:.5f}")
            print(f"    Circuit depth (raw)    : total={raw_d}   2Q gates={raw_2q}")
            print(f"    Circuit depth (HW)     : total={hw_d}   2Q depth={hw_2q}")
            print(f"    Submitting to hardware ...")

            tqc = transpile(qc, backend, optimization_level=3)
            job = sampler.run([tqc], shots=SHOTS)
            res = job.result()

            counts = res[0].data.c.get_counts()
            hw_val = similarity(counts, dim, SHOTS)

            print(f"    HW result              : {hw_val:.5f}  "
                  f"(err vs classical: {abs(hw_val - classical):.5f})")

            all_results.append(dict(
                dim=dim, case=case_name,
                classical=classical, hw=hw_val,
                raw_depth=raw_d, raw_2q=raw_2q,
                hw_depth=hw_d, hw_2q=hw_2q,
            ))

    # ── Summary ──────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("  SUMMARY")
    print(f"{'='*80}")
    print(f"  {'dim':>5} │ {'case':<24} │ {'classical':>10} │ {'HW result':>10} │ "
          f"{'err':>8} │ {'raw depth':>10} │ {'raw 2Q':>7} │ {'HW depth':>9} │ {'HW 2Q':>7}")
    print(f"  {'─'*5}─┼─{'─'*24}─┼─{'─'*10}─┼─{'─'*10}─┼─"
          f"{'─'*8}─┼─{'─'*10}─┼─{'─'*7}─┼─{'─'*9}─┼─{'─'*7}")
    for r in all_results:
        print(f"  {r['dim']:>5} │ {r['case']:<24} │ {r['classical']:>10.5f} │ "
              f"{r['hw']:>10.5f} │ {abs(r['hw']-r['classical']):>8.5f} │ "
              f"{r['raw_depth']:>10} │ {r['raw_2q']:>7} │ "
              f"{r['hw_depth']:>9} │ {r['hw_2q']:>7}")

    print(f"\n  For comparison — DiagonalGate (measured earlier):")
    diag_ref = [
        (16,  186,  57),
        (32,  370, 116),
        (64,  705, 219),
    ]
    for dim, d, q in diag_ref:
        print(f"    dim={dim:>4}  DiagonalGate:  HW depth={d:>5}   HW 2Q depth={q:>4}")

    print(f"{'='*80}\n")


if __name__ == "__main__":
    run()