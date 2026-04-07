"""
diagnose_topo.py
================
Diagnose why topo-aware depth is higher than standard.
Prints detailed circuit info at each PassManager stage.
"""

import numpy as np
from math import ceil, log2

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.circuit.library import DiagonalGate
from qiskit.transpiler import PassManager, Layout, CouplingMap
from qiskit.transpiler.passes import (
    UnrollCustomDefinitions,
    BasisTranslator,
    SetLayout,
    FullAncillaAllocation,
    EnlargeWithAncilla,
    ApplyLayout,
    SabreSwap,
    Optimize1qGatesDecomposition,
    CommutativeCancellation,
)
from qiskit.circuit.equivalence_library import SessionEquivalenceLibrary
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import QiskitRuntimeService

QiskitRuntimeService.save_account(
    channel="ibm_cloud",
    token="v2WGCXrHMvu1Nh2tzr39Mocon2npcG_ogKxvgvKtTyg2",
    instance="crn:v1:bluemix:public:quantum-computing:us-east:a/813b37ffee14414ca81092ab94341434:1284900f-4e18-41c7-aadf-44278c5d44da::",
    set_as_default=True, overwrite=True,
)

service    = QiskitRuntimeService()
backend    = service.backend("ibm_cleveland")
basis_gates = list(backend.operation_names)
coupling_map = CouplingMap(backend.coupling_map)

print(f"Backend     : {backend.name}")
print(f"Basis gates : {basis_gates}")
print(f"Num qubits  : {backend.num_qubits}")

# ── Build a simple dim=32 (5 qubit) CU circuit ───────────────────
dim = 32
n_q = int(ceil(log2(dim)))
v1  = np.random.default_rng(42).choice([-1, 1], size=dim)

vec = v1.copy()
if len(vec) < 2**n_q:
    vec = np.concatenate([vec, np.ones(2**n_q - len(vec))])

gate = DiagonalGate(vec.tolist())
qc_enc = QuantumCircuit(n_q, name="enc")
qc_enc.append(gate, range(n_q))

sys  = QuantumRegister(n_q, "sys")
creg = ClassicalRegister(n_q, "c_meas")
qc   = QuantumCircuit(sys, creg)
qc.h(sys)
qc.compose(qc_enc, qubits=sys, inplace=True)
qc.compose(qc_enc.inverse(), qubits=sys, inplace=True)
qc.h(sys)
qc.measure(sys, creg)

print(f"\nOriginal circuit: depth={qc.depth()}, gates={qc.count_ops()}")

# ── STEP 1: Decompose DiagonalGate ───────────────────────────────
intermediate_basis = ['cx', 'u', 'rz', 'sx', 'x', 'id', 'measure', 'reset']

pm_decompose = PassManager([
    UnrollCustomDefinitions(SessionEquivalenceLibrary,
                            basis_gates=intermediate_basis),
    BasisTranslator(SessionEquivalenceLibrary,
                    target_basis=intermediate_basis),
])
qc_decomposed = pm_decompose.run(qc)
print(f"\nAfter decompose: depth={qc_decomposed.depth()}")
print(f"  Gate counts: {qc_decomposed.count_ops()}")

# Count CNOT pairs
cnot_pairs = {}
for inst in qc_decomposed.data:
    if inst.operation.name == 'cx':
        q0 = qc_decomposed.find_bit(inst.qubits[0]).index
        q1 = qc_decomposed.find_bit(inst.qubits[1]).index
        pair = (min(q0,q1), max(q0,q1))
        cnot_pairs[pair] = cnot_pairs.get(pair, 0) + 1

print(f"\n  CNOT pairs in decomposed circuit:")
for pair, count in sorted(cnot_pairs.items(), key=lambda x: -x[1]):
    print(f"    {pair}: {count} CNOTs")

# ── STEP 2: Check if [0,1,2,3,4] satisfies all CNOT pairs ────────
chain = [0, 1, 2, 3, 4]
print(f"\nChain: {chain}")
print(f"CNOT pair vs chain adjacency:")
adj = {}
for a, b in backend.coupling_map:
    adj.setdefault(a, []).append(b)
    adj.setdefault(b, []).append(a)

for (a, b), count in sorted(cnot_pairs.items(), key=lambda x: -x[1]):
    # a and b are LOGICAL qubits in the decomposed circuit
    # after ApplyLayout they map to physical: chain[a] and chain[b]
    phys_a = chain[a] if a < len(chain) else a
    phys_b = chain[b] if b < len(chain) else b
    connected = phys_b in adj.get(phys_a, [])
    hop = abs(chain.index(phys_a) - chain.index(phys_b)) if phys_a in chain and phys_b in chain else 999
    print(f"    logical ({a},{b}) → physical ({phys_a},{phys_b}) "
          f"| connected={connected} | hop={hop} "
          f"| count={count}")

# ── STEP 3: Full PassManager topo ────────────────────────────────
layout = Layout({qc.qregs[0][i]: chain[i] for i in range(n_q)})

pm_topo = PassManager([
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

qc_topo = pm_topo.run(qc)
print(f"\nTopo-aware result: depth={qc_topo.depth()}")
print(f"  Gate counts: {qc_topo.count_ops()}")

# Count SWAPs
n_swaps = qc_topo.count_ops().get('swap', 0)
print(f"  SWAPs inserted: {n_swaps}")

# Count 2Q depth
from qiskit import QuantumCircuit as QC2
two_q = QC2(qc_topo.num_qubits)
for inst in qc_topo.data:
    if len(inst.qubits) == 2:
        idxs = [qc_topo.find_bit(q).index for q in inst.qubits]
        two_q.append(inst.operation, idxs)
print(f"  2Q depth: {two_q.depth()}")

# ── STEP 4: Standard transpile for comparison ────────────────────
qc_std = transpile(qc, backend, optimization_level=3)
print(f"\nStandard result: depth={qc_std.depth()}")
print(f"  Gate counts: {qc_std.count_ops()}")
n_swaps_std = qc_std.count_ops().get('swap', 0)
print(f"  SWAPs inserted: {n_swaps_std}")

two_q_std = QC2(qc_std.num_qubits)
for inst in qc_std.data:
    if len(inst.qubits) == 2:
        idxs = [qc_std.find_bit(q).index for q in inst.qubits]
        two_q_std.append(inst.operation, idxs)
print(f"  2Q depth: {two_q_std.depth()}")

# ── STEP 5: What physical qubits did std choose? ──────────────────
print(f"\nStandard layout chosen by Qiskit:")
if hasattr(qc_std, '_layout') and qc_std._layout:
    layout_info = qc_std._layout
    print(f"  {layout_info}")
else:
    # Read from circuit properties
    print("  (checking transpiled circuit for qubit mapping...)")
    meas_qubits = set()
    for inst in qc_std.data:
        if inst.operation.name == 'measure':
            meas_qubits.add(qc_std.find_bit(inst.qubits[0]).index)
    print(f"  Measured physical qubits: {sorted(meas_qubits)}")
