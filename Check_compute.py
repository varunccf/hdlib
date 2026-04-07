"""Quantum implementation of the MAP arithmetic operators."""

import re
from math import atan2, sqrt, ceil, log2, pi
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from mthree import M3Mitigation
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.circuit import Gate, Qubit
from qiskit.circuit.library import DiagonalGate, XGate
from qiskit.quantum_info import Statevector
from qiskit.providers.backend import Backend
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import QiskitRuntimeService , Session
from qiskit_ibm_runtime import Sampler, Options

QiskitRuntimeService.save_account(
channel="ibm_cloud",
token= "v2WGCXrHMvu1Nh2tzr39Mocon2npcG_ogKxvgvKtTyg2", # Use the 44-character API_KEY you created and saved from the IBM Quantum Platform Home dashboard
instance="crn:v1:bluemix:public:quantum-computing:us-east:a/813b37ffee14414ca81092ab94341434:1284900f-4e18-41c7-aadf-44278c5d44da::" ,
set_as_default= True,
overwrite = True,
)
service = QiskitRuntimeService()

#backends = service.backends()
#print(backends)


def statevector_to_bipolar(circuit: QuantumCircuit) -> np.ndarray:
    """Extracts a classical bipolar vector from the phases of a quantum statevector.

    This function provides a method to decode a quantum state back into a classical vector.
    It assumes the information is encoded in the sign of the real part of the amplitudes,
    mapping positive signs to +1 and negative signs to -1.

    Automatically detects if the data is in standard (0/pi) or symmetric (+/- delta) encoding
    and rotates if necessary.

    Parameters
    ----------
    circuit : QuantumCircuit
        A quantum circuit to simulate and retrieve the classical bipolar vector from.

    Returns
    -------
    numpy.ndarray
        The corresponding classical bipolar vector of integers (+1 or -1).
    """

    statevector = Statevector.from_instruction(circuit.decompose())
    statevector_data = np.asarray(statevector.data)

    # Heuristic: determine encoding based on the presence of negative real components.
    # Standard encoding (0/pi): has amplitudes ~ +1 and ~ -1. Min real < -0.5.
    # Symmetric encoding (+/- delta): has amplitudes e^(+id) and e^(-id).
    # For small delta, real part is cos(d) ~ 1 (always positive).
    min_real = np.min(np.real(statevector_data))

    # Adapt
    data = statevector_data

    # If all real parts are non-negative, the information must be in the phase.
    # Rotate by -90 degrees to project phase (imag) onto real axis for decoding.
    if min_real > -1e-5:
        data = statevector_data * -1j

    # Decode
    reals = np.real(data)
    tolerance = 1e-9

    vec = np.ones(len(reals), dtype=int)

    # Positive real +1, negative real -1
    vec[reals < -tolerance] = -1

    return vec.astype(int)






def __mitigate_counts(counts, backend, shots, measured_qubits, mitigator: Optional[M3Mitigation]=None):
    """Apply readout error mitigation using mthree to a single-qubit measurement.
    """

    if mitigator is None:
        # Initialize mitigator from backend
        mitigator = M3Mitigation(backend)
        mitigator.cals_from_system(qubits=measured_qubits)

    # Apply correction to get mitigated probabilities
    probs = mitigator.apply_correction(counts, qubits=measured_qubits)

    # Dynamically convert all output states back to pseudo-counts
    mitigated_pseudo_counts = dict()

    for state, prob in probs.items():
        # Use max(0, prob) because M3 can sometimes output tiny negative quasi-probabilities
        safe_prob = max(0.0, prob)
        mitigated_pseudo_counts[state] = int(round(safe_prob * shots))

    # Return mitigated probabilities as pseudo-counts
    return mitigated_pseudo_counts

def __get_measured_physical_qubits(
    transpiled_circuit: QuantumCircuit,
    measured_register: ClassicalRegister
) -> list[int]:
    """Returns physical qubits corresponding to the measured classical bits."""

    try:
        transpiled_creg = next(
            reg for reg in transpiled_circuit.cregs
            if reg.name == measured_register.name
        )
    except StopIteration:  # more specific than bare except
        raise ValueError(f"Register '{measured_register.name}' not found in transpiled circuit.")

    # Map classical bits → physical qubit indices
    meas_map = {}
    for inst in transpiled_circuit.data:
        if inst.operation.name == "measure":
            qbit_idx = transpiled_circuit.find_bit(inst.qubits[0]).index
            meas_map[inst.clbits[0]] = qbit_idx

    # Extract physical qubits in register order
    physical_qubits = []
    for cbit in transpiled_creg:
        if cbit not in meas_map:
            raise ValueError(f"Classical bit {cbit} has no measurement mapped to it.")
        physical_qubits.append(meas_map[cbit])

    # Reverse: Qiskit bitstrings are MSB→LSB (left to right = c_{N-1}...c_0)
    # M3 expects physical_qubits to match that same left-to-right order
    return physical_qubits[::-1]

def run_compute_uncompute_test_M3(
    state_left_circs: List[QuantumCircuit],
    state_right_circs: List[QuantumCircuit],
    backend: Backend,
    shots: int = 1024,
    seed: int = 42,
    sampler: Optional[Sampler] = None
) -> Tuple[List[List[float]], List[dict]]:
    """Performs a Compute-Uncompute (Inversion) test to measure |<L|R>|^2 in batch mode.

    This avoids all controlled operations, making it exponentially cheaper
    to transpile and execute compared to the Hadamard Test.
    """
    is_simulated = isinstance(backend, AerSimulator)
    n_sys = state_right_circs[0].num_qubits

    if state_left_circs[0].num_qubits != n_sys:
        raise ValueError("Left and Right circuits must have the exact same number of qubits for Inversion test.")

    sys = QuantumRegister(n_sys, "sys")
    creg = ClassicalRegister(n_sys, "c_meas")

    qcs = list()
    for query_circ in state_left_circs:
        for prototype_circ in state_right_circs:
            qc = QuantumCircuit(sys, creg)

            # 1. Initialize uniform superposition
            qc.h(sys)

            # 2. Compute: Apply query state (R)
            qc.compose(query_circ, qubits=sys, inplace=True)

            # 3. Uncompute: Apply inverse of prototype state (L)
            qc.compose(prototype_circ.inverse(), qubits=sys, inplace=True)

            # 4. Map phases back to amplitudes for measurement
            qc.h(sys)

            # 5. Measure all qubits
            qc.measure(sys, creg)

            qcs.append(qc)
    #  SIMULATOR
    if is_simulated:
        tqcs = transpile(qcs, backend, optimization_level=1)

        result = backend.run(tqcs, shots=shots, seed_simulator=seed).result()
        counts = result.get_counts()

        if not isinstance(counts, list):
            counts = [counts]

    #  HARDWARE
    else:
        if sampler is None:
            raise ValueError("Sampler required for hardware execution.")


        tqcs = transpile(qcs, backend , optimization_level=3)

        # Debug check (VERY useful)
        #print("\n===== DEBUG CIRCUIT =====")
        #print(tqcs[0].draw())
        sampler.options.dynamical_decoupling.enable = True
        sampler.options.dynamical_decoupling.sequence_type = "XpXm"
        sampler.options.twirling.enable_gates = True

        job = sampler.run(tqcs, shots=shots)
        results = job.result()
        counts = []

        #Get measured qubits for mitigation
        all_measured_qubits = set()
        circuit_measured_qubits = []

        for tqc in tqcs:
            phys_qubits = __get_measured_physical_qubits(tqc,creg)
            all_measured_qubits.update(phys_qubits)
            circuit_measured_qubits.append(phys_qubits)

        #Initialize M3 mitigator
        mitigator = M3Mitigation(backend)
        mitigator.cals_from_system(qubits=list(all_measured_qubits))

        for i, res in enumerate(results):
            counts_res = res.data.c_meas.get_counts()
            mitigated = __mitigate_counts(
                counts_res,
                backend,
                shots,
                circuit_measured_qubits[i],
                mitigator=mitigator
            )
            counts.append(mitigated)

    #Convert to similarity matrix
    similarities = []
    idx = 0
    target_state = "0" * n_sys

    for _ in state_left_circs:
        row = []
        for _ in state_right_circs:
            prob = counts[idx].get(target_state, 0) / shots
            row.append(prob)
            idx += 1
        similarities.append(row)

    return similarities, counts


def run_hadamard_test_M3(
    state_left_circ: QuantumCircuit,
    state_right_circ: QuantumCircuit,
    backend: Backend,
    shots: int = 1024,
    seed: int = 42,
    sampler: Optional[Sampler] = None
) -> Tuple[float, dict]:

    is_simulated = isinstance(backend, AerSimulator)

    n_total = state_right_circ.num_qubits
    n_sys = state_left_circ.num_qubits

    if n_total < n_sys:
        raise ValueError("Right circuit must have ≥ qubits than left.")

    num_anc_pad = n_total - n_sys

    # Convert to gates (safe conversion)
    v_l_gate = state_left_circ.to_gate(label="Prep_L")
    v_r_gate = state_right_circ.to_gate(label="Prep_R")

    # Registers
    anc = QuantumRegister(1, "anc")
    sys = QuantumRegister(n_total, "sys")
    creg = ClassicalRegister(1, "c")

    qc = QuantumCircuit(anc, sys, creg)

    # 1. Hadamard on ancilla
    qc.h(anc[0])

    # 2. Controlled L (on padded system)
    qc.append(
        v_l_gate.control(1),
        [anc[0]] + list(sys[num_anc_pad:])
    )

    qc.barrier()

    # 3. X on ancilla
    qc.x(anc[0])

    # 4. Controlled R
    qc.append(
        v_r_gate.control(1),
        [anc[0]] + list(sys[:])
    )

    qc.barrier()

    # 5. Final Hadamard
    qc.h(anc[0])

    qc.measure(anc[0], creg[0])

    #  SIMULATOR
    if is_simulated:
        tqc = transpile(qc, backend)
        counts = backend.run(tqc, shots=shots, seed_simulator=seed).result().get_counts()

    #  HARDWARE
    else:
        if sampler is None:
            raise ValueError("Sampler required for hardware.")

        tqc = transpile(qc, backend, optimization_level=3)

        sampler.options.dynamical_decoupling.enable = True
        sampler.options.dynamical_decoupling.sequence_type = "XpXm"
        sampler.options.twirling.enable_gates = True

        #print("\n===== HADAMARD CIRCUIT =====")
        #print(tqc.draw())

        job = sampler.run([tqc], shots=shots)
        result = job.result()

        # SamplerV2 extraction
        bit_data = result[0].data.c
        counts_1 = np.count_nonzero(bit_data.array == 1)
        counts_0 = shots - counts_1
        counts = {"0": counts_0, "1": counts_1}
        print(counts)

        # M3 mitigation
        measured_qubits = [tqc.find_bit(tqc.qubits[0]).index]  # ancilla only

        mitigator = M3Mitigation(backend)
        mitigator.cals_from_system(qubits=measured_qubits)

        counts = __mitigate_counts(
            counts,
            backend,
            shots,
            measured_qubits,
            mitigator=mitigator
        )
        print(counts)


    # Compute similarity
    p0 = counts.get("0", 0) / shots
    p1 = counts.get("1", 0) / shots

    similarity = p0 - p1  # Re(<L|R>)

    return similarity, counts

def encode(vec_bipolar: np.ndarray, label: str="O_v") -> QuantumCircuit:
    """Creates a circuit containing a diagonal phase oracle.
    This function is a core component for encoding classical bipolar vectors into the phase of a quantum state.

    Parameters
    ----------
    vec_bipolar : numpy.ndarray
        A classical vector containing only -1 and +1 values.
    label : str, default "O_v"
        An optional label for the created Qiskit gate.

    Returns
    -------
    qiskit.QuantumCircuit
        A quantum circuit containing the diagonal gate.

    Raises
    ------
    ValueError
        If the input `vec_bipolar` contains values other than -1 or +1.
    """

    vec = np.asarray(vec_bipolar)
    if not np.all(np.isin(vec, [-1, 1])):
        raise ValueError("Bipolar vector must contain only -1 or +1.")

    num_qubits = int(ceil(log2(len(vec))))
    # Pad vector if necessary to match 2^N
    if len(vec) < 2**num_qubits:
        padding = np.ones(2**num_qubits - len(vec))
        vec = np.concatenate([vec, padding])
    # Convert to complex diagonal entries
    gate = DiagonalGate(vec.tolist())
    gate.label = label
    qc = QuantumCircuit(num_qubits, name=label)
    qc.append(gate, range(num_qubits))

    return qc




def negate_circuits(circuits: List[QuantumCircuit]) -> List[QuantumCircuit]:
    """Flips the bipolar phase of the circuits for subtraction.
    Multiplying the complex eigenvalues of the DiagonalGate by -1 reflects the vector.

    Parameters
    ----------
    circuits : list
        The input circuits.

    Returns
    -------
    list
        List of phase-flipped circuits.
    """

    negated = list()

    for circuit in circuits:
        neg_circuit = QuantumCircuit(*circuit.qregs, name=f"{circuit.name}_neg")

        for instr in circuit.data:
            if isinstance(instr.operation, DiagonalGate):
                # -1 inverts the bipolar phases
                new_phases = np.array(instr.operation.params, dtype=complex) * -1.0
                neg_circuit.append(DiagonalGate(new_phases.tolist()), instr.qubits)

            else:
                neg_circuit.append(instr)

        negated.append(neg_circuit)

    return negated

def run_compute_uncompute_test(
    state_left_circs: List[QuantumCircuit],
    state_right_circs: List[QuantumCircuit],
    backend: Backend,
    shots: int = 1024,
    seed: int = 42,
    sampler: Optional[Sampler] = None
) -> Tuple[List[List[float]], List[dict]]:
    """Performs a Compute-Uncompute (Inversion) test to measure |<L|R>|^2 in batch mode.

    Avoids all controlled operations, making it exponentially cheaper
    to transpile and execute compared to the Hadamard Test.
    """
    is_simulated = isinstance(backend, AerSimulator)
    n_sys = state_right_circs[0].num_qubits

    if state_left_circs[0].num_qubits != n_sys:
        raise ValueError("Left and Right circuits must have the same number of qubits.")

    sys = QuantumRegister(n_sys, "sys")
    creg = ClassicalRegister(n_sys, "c_meas")

    qcs = []
    for query_circ in state_left_circs:
        for prototype_circ in state_right_circs:
            qc = QuantumCircuit(sys, creg)
            qc.h(sys)                                             # 1. Uniform superposition
            qc.compose(query_circ, qubits=sys, inplace=True)     # 2. Compute: Apply query (R)
            qc.compose(prototype_circ.inverse(), qubits=sys, inplace=True)  # 3. Uncompute: Apply L†
            qc.h(sys)                                             # 4. Map phases back to amplitudes
            qc.measure(sys, creg)                                 # 5. Measure
            qcs.append(qc)

    # ── SIMULATOR ────────────────────────────────────────────────
    if is_simulated:
        tqcs = transpile(qcs, backend, optimization_level=1)
        result = backend.run(tqcs, shots=shots, seed_simulator=seed).result()
        counts = result.get_counts()
        if not isinstance(counts, list):
            counts = [counts]

    # ── HARDWARE ─────────────────────────────────────────────────
    else:
        if sampler is None:
            raise ValueError("Sampler required for hardware execution.")

        tqcs = transpile(qcs, backend, optimization_level=3)

        sampler.options.dynamical_decoupling.enable = True
        sampler.options.dynamical_decoupling.sequence_type = "XpXm"
        sampler.options.twirling.enable_gates = True

        job = sampler.run(tqcs, shots=shots)
        results = job.result()

        counts = []
        for res in results:
            counts.append(res.data.c_meas.get_counts())

    # ── Convert to similarity matrix ──────────────────────────────
    similarities = []
    idx = 0
    target_state = "0" * n_sys

    for _ in state_left_circs:
        row = []
        for _ in state_right_circs:
            prob = counts[idx].get(target_state, 0) / shots
            row.append(prob)
            idx += 1
        similarities.append(row)

    return similarities, counts


def run_hadamard_test(
    state_left_circ: QuantumCircuit,
    state_right_circ: QuantumCircuit,
    backend: Backend,
    shots: int = 1024,
    seed: int = 42,
    sampler: Optional[Sampler] = None
) -> Tuple[float, dict]:
    """Performs a Hadamard Test to measure Re(<L|R>).

    Uses an ancilla qubit to extract the real part of the inner product.
    """
    is_simulated = isinstance(backend, AerSimulator)

    n_total = state_right_circ.num_qubits
    n_sys   = state_left_circ.num_qubits

    if n_total < n_sys:
        raise ValueError("Right circuit must have >= qubits than left.")

    num_anc_pad = n_total - n_sys

    v_l_gate = state_left_circ.to_gate(label="Prep_L")
    v_r_gate = state_right_circ.to_gate(label="Prep_R")

    anc  = QuantumRegister(1, "anc")
    sys  = QuantumRegister(n_total, "sys")
    creg = ClassicalRegister(1, "c")

    qc = QuantumCircuit(anc, sys, creg)
    qc.h(anc[0])                                                        # 1. Hadamard on ancilla
    qc.append(v_l_gate.control(1), [anc[0]] + list(sys[num_anc_pad:])) # 2. Controlled-L
    qc.barrier()
    qc.x(anc[0])                                                        # 3. X on ancilla
    qc.append(v_r_gate.control(1), [anc[0]] + list(sys[:]))            # 4. Controlled-R
    qc.barrier()
    qc.h(anc[0])                                                        # 5. Final Hadamard
    qc.measure(anc[0], creg[0])                                         # 6. Measure ancilla only

    # ── SIMULATOR ────────────────────────────────────────────────
    if is_simulated:
        tqc = transpile(qc, backend, optimization_level=1)
        counts = backend.run(tqc, shots=shots, seed_simulator=seed).result().get_counts()

    # ── HARDWARE ─────────────────────────────────────────────────
    else:
        if sampler is None:
            raise ValueError("Sampler required for hardware execution.")

        tqc = transpile(qc, backend, optimization_level=3)

        sampler.options.dynamical_decoupling.enable = True
        sampler.options.dynamical_decoupling.sequence_type = "XpXm"
        sampler.options.twirling.enable_gates = True

        job     = sampler.run([tqc], shots=shots)
        results = job.result()

        # SamplerV2 extraction
        bit_data  = results[0].data.c
        counts_1  = int(np.count_nonzero(bit_data.array == 1))
        counts_0  = shots - counts_1
        counts    = {"0": counts_0, "1": counts_1}

    # ── Compute similarity: Re(<L|R>) = P(0) - P(1) ──────────────
    p0 = counts.get("0", 0) / shots
    p1 = counts.get("1", 0) / shots
    similarity = p0 - p1

    return similarity, counts


# =========================================================
#  Utility Functions
# =========================================================

def generate_bipolar(dim: int) -> np.ndarray:
    """Generate a random bipolar vector {-1, +1}."""
    return np.random.choice([-1, 1], size=dim)


def normalize(vec: np.ndarray) -> np.ndarray:
    """Normalize vector to unit length."""
    return vec / np.linalg.norm(vec)


def classical_similarity(v1: np.ndarray, v2: np.ndarray):
    """Compute classical similarity metrics."""
    v1_n = normalize(v1)
    v2_n = normalize(v2)

    inner = np.dot(v1_n, v2_n)
    return inner, inner**2


def quantum_ground_truth(qc_left, qc_right):
    """Compute exact quantum overlap from statevectors."""
    sv_L = Statevector.from_instruction(qc_left)
    sv_R = Statevector.from_instruction(qc_right)

    inner = np.vdot(sv_L.data, sv_R.data)
    return inner, abs(inner)**2


# =========================================================
#  Main Experiment Pipeline
# =========================================================

def run_experiment(dim=16, shots=2048):

    print("\n==============================")
    print(" GENERATING DATA")
    print("==============================")

    v1 = generate_bipolar(dim)
    v2 = generate_bipolar(dim)

    print("v1:", v1)
    print("v2:", v2)

    # Encode into quantum circuits
    qc_left = encode(v1, label="L")
    qc_right = encode(v1, label="R")

    # Debug encoding
    print("\nEncoded L:", statevector_to_bipolar(qc_left))
    print("Encoded R:", statevector_to_bipolar(qc_right))

    # =====================================================
    #  Classical Ground Truth
    # =====================================================
    print("\n==============================")
    print(" CLASSICAL RESULTS")
    print("==============================")

    inner_classical, inner_sq_classical = classical_similarity(v1, v2)

    print("⟨v1|v2⟩:", inner_classical)
    print("|⟨v1|v2⟩|²:", inner_sq_classical)

    # =====================================================
    #  Quantum Ground Truth (Statevector)
    # =====================================================
    print("\n==============================")
    print(" QUANTUM (STATEVECTOR)")
    print("==============================")

    inner_q, inner_sq_q = quantum_ground_truth(qc_left, qc_right)

    print("⟨L|R⟩:", inner_q)
    print("|⟨L|R⟩|²:", inner_sq_q)

    # =====================================================
    #  Simulator Results
    # =====================================================
    print("\n==============================")
    print(" SIMULATOR RESULTS")
    print("==============================")

    sim_backend = AerSimulator()

    had_sim_sim, _ = run_hadamard_test(
        qc_left, qc_right, sim_backend, shots=shots
    )

    cu_sim_sim, _ = run_compute_uncompute_test(
        [qc_left], [qc_right], sim_backend, shots=shots
    )

    print("Hadamard (Sim):", had_sim_sim)
    print("Compute–Uncompute (Sim):", cu_sim_sim[0][0])

    # =====================================================
    #  Hardware Results
    # =====================================================
    print("\n==============================")
    print(" HARDWARE RESULTS")
    print("==============================")

    service = QiskitRuntimeService()
    backend = service.backend("ibm_cleveland")

    sampler = Sampler(mode=backend)

    had_sim_hw, _ = run_hadamard_test(
        qc_left,
        qc_right,
        backend,
        shots=shots,
        sampler=sampler
    )

    cu_sim_hw, _ = run_compute_uncompute_test(
        [qc_left],
        [qc_right],
        backend,
        shots=shots,
        sampler=sampler
    )

    cu_sim_hw_m3, _ = run_compute_uncompute_test_M3(
        [qc_left],
        [qc_right],
        backend,
        shots=shots,
        sampler=sampler
    )
    print("CU with M3: ", cu_sim_hw_m3 )
    print("Hadamard (HW):", had_sim_hw)
    print("Compute–Uncompute (HW):", cu_sim_hw[0][0])

    # =====================================================
    #  Comparison Summary
    # =====================================================
    print("\n==============================")
    print(" SUMMARY")
    print("==============================")

    print(f"{'Method':<25} {'Value':<15}")
    print("-" * 40)

    print(f"{'Classical ⟨v1|v2⟩':<25} {inner_classical:.6f}")
    print(f"{'Quantum ⟨L|R⟩':<25} {np.real(inner_q):.6f}")
    print(f"{'Hadamard (Sim)':<25} {had_sim_sim:.6f}")
    print(f"{'Hadamard (HW)':<25} {had_sim_hw:.6f}")

    print("-" * 40)

    print(f"{'Classical |.|²':<25} {inner_sq_classical:.6f}")
    print(f"{'Quantum |.|²':<25} {inner_sq_q:.6f}")
    print(f"{'CU (Sim)':<25} {cu_sim_sim[0][0]:.6f}")
    print(f"{'CU (HW)':<25} {cu_sim_hw[0][0]:.6f}")

    def safe_rel_error(hw, ref):
        return abs(hw - ref) / (abs(ref) + 1e-12)

    cu_hw = cu_sim_hw[0][0]
    cu_sim = cu_sim_sim[0][0]

    print("\nRELATIVE ERRORS")
    print("-" * 45)

    print(f"{'Hadamard vs Classical':<30} {safe_rel_error(had_sim_hw, inner_classical):.6f}")
    print(f"{'Hadamard vs Quantum StateVector':<30} {safe_rel_error(had_sim_hw, np.real(inner_q)):.6f}")
    print(f"{'Hadamard vs AerSimulator':<30} {safe_rel_error(had_sim_hw, had_sim_sim):.6f}")

    print("-" * 45)

    print(f"{'CU vs Classical':<30} {safe_rel_error(cu_hw, inner_sq_classical):.6f}")
    print(f"{'CU vs Quantum Statevector':<30} {safe_rel_error(cu_hw, inner_sq_q):.6f}")
    print(f"{'CU vs AerSimulator':<30} {safe_rel_error(cu_hw, cu_sim):.6f}")

    print("\nDone.\n")

# =========================================================
#  Run
# =========================================================

if __name__ == "__main__":
    run_experiment(dim=16, shots=2048)