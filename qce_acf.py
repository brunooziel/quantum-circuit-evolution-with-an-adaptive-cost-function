import time
import numpy as np
import random
from typing import List, Tuple, Optional, Callable, Any
from qiskit import transpile
from _utility import to_qubo, qubo_evaluation, check_keys

from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import RXGate, RYGate, RZGate, RXXGate, RYYGate, RZZGate, CRXGate, CRYGate, CRZGate
from qiskit.primitives import Estimator
from qiskit_aer.primitives import Estimator as AerEstimator
from qiskit.result import sampled_expectation_value
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer import AerSimulator



# QCE - Quantum Circuit Evolutionary

single_gates = [RXGate, RYGate, RZGate]  
double_gates = [CRXGate, CRYGate, RXXGate, RYYGate, RZZGate]


def initial_circuit(qubits: int) -> QuantumCircuit:
    """
    Cria um circuito quântico inicial com uma única porta de rotação aleatória.

    Parameters:
        qubits (int): Número de qubits do circuito.

    Returns:
        QuantumCircuit: Circuito inicial com uma porta aleatória.
    """
    qc = QuantumCircuit(qubits)
    qubit = np.random.randint(qubits)
    gate_choice = random.choice(['rx', 'ry', 'rz'])
    theta = np.random.uniform(0, 2 * np.pi)

    if gate_choice == 'rx':
        qc.rx(theta, qubit)
    elif gate_choice == 'ry':
        qc.ry(theta, qubit)
    elif gate_choice == 'rz':
        qc.rz(theta, qubit)
    
    return qc

def choose(circuit):

    all_operations = [(i, instr.operation, instr.qubits) for i, instr in enumerate(circuit.data)]
    draw = np.random.choice(len(all_operations))
    return all_operations[draw]

def modify(circuit):

    index, gate, qubits = choose(circuit)
    
    # Verifica se a porta possui parâmetros
    if hasattr(gate, 'params') and gate.params:
        epsilon = np.random.normal(0, 0.1)
        new_theta = gate.params[0] + epsilon
        new_gate = gate.__class__(new_theta)
    else:
        new_gate = gate  # Se não houver parâmetros, nada é alterado
    
    circuit.data[index] = (new_gate, qubits, [])
    return circuit

def delete(circuit):

    index, gate, _ = choose(circuit)
    circuit.data.pop(index)
    return circuit

def get_gate(circuit, gate_kind):

    if gate_kind is None:
        gate_kind = random.choice(['single', 'double'])
    
    if gate_kind == 'single':
        gate_class = random.choice(single_gates)
        angle = random.uniform(-2 * np.pi, 2 * np.pi)
        gate = gate_class(angle)
        qubit = random.choice(range(circuit.num_qubits))
        qubits = [circuit.qubits[qubit]]
    else:
        gate_class = random.choice(double_gates)
        angle = random.uniform(-2 * np.pi, 2 * np.pi)
        gate = gate_class(angle)
        control = random.choice(range(circuit.num_qubits))
        target = random.choice([q for q in range(circuit.num_qubits) if q != control])
        qubits = [circuit.qubits[control], circuit.qubits[target]]
    
    return gate, qubits

def insert(circuit):

    gate_kind = random.choice(['single', 'double'])
    random_gate, qubits = get_gate(circuit, gate_kind)
    insert_position = random.choice(range(len(circuit.data) + 1))
    circuit.data.insert(insert_position, (random_gate, qubits, []))
    return circuit

def swap(circuit):

    index, _, _ = choose(circuit)
    circuit.data.pop(index)
    gate_kind = random.choice(['single', 'double'])
    new_gate, qubits = get_gate(circuit, gate_kind)
    circuit.data.insert(index, (new_gate, qubits, []))
    return circuit

def mutation(circuit):
    
    child = circuit.copy()
    actions = ["INSERT", "DELETE", "SWAP", "MODIFY"]
    actions_prob = [0.25, 0.25, 0.25, 0.25] 

    if len(child.data) <= 1:
        child = insert(child)
        mutation_type = "INSERT"
    else:
        mutation_type = np.random.choice(actions, p=actions_prob)
        if mutation_type == "INSERT":
            child = insert(child)
        elif mutation_type == "DELETE":
            child = delete(child)
        elif mutation_type == "SWAP":
            child = swap(child)
        elif mutation_type == "MODIFY":
            child = modify(child)

    return child


def calculate(estimator, circuit, hamiltonian, ref_value, Q, const, A):
    """
    Calculate the objective function value for a quantum circuit.
    
    Args:
        estimator: Quantum estimator to run the circuit
        circuit: Quantum circuit to evaluate
        hamiltonian: Hamiltonian operator
        ref_value: Reference value for normalization
        Q: QUBO matrix
        const: Constant term in QUBO
        A: Constraint matrix for feasibility checking
        
    Returns:
        tuple: (objective_value, best_key) where:
            - objective_value (float): Objective function value
            - best_key (str): Bitstring of the most significant valid state
    """
    # Create a copy of the circuit and add measurements
    
    # Total shots
    shots = 1024
    
    child = circuit.copy()
    child.measure_all()
    
    transpiled_circuit = transpile(child, estimator)
    
    # Run the circuit and get measurement counts  ON AER
    job = estimator.run(circuits=transpiled_circuit, shots=shots)
    counts = job.result().get_counts()
    
    # Run the circuit and get measurement counts  ON SAMPLER
    # Check for constraint violations
    violations = check_keys(A, counts)
    
    # If all states violate constraints but we have multiple states
    if len(violations) == len(counts) and len(counts) > 1 :
        # Use the most frequent state as the representative
        string_max = max(counts, key=counts.get)
        violations = [string_max]
        
    # Calculate valid shots (those not violating constraints)
    valid_shots = shots - sum(counts.get(key, 0) for key in violations)
    
    # If no valid shots, return a high penalty value and the most frequent state
    if valid_shots == 0:
        best_key = max(counts, key=counts.get)
        return 1e10, best_key
    
    # Calculate the objective function value
    objective_value = 0
    valid_counts = {}
    
    for key, value in counts.items():
        # Skip states that violate constraints
        if key in violations:
            continue
        
        # Store valid counts for determining the best key
        valid_counts[key] = value
        
        # Add contribution of this state to the objective
        objective_value += qubo_evaluation(Q, const, key) * value / valid_shots
    
    # Find the most significant valid state (highest count)
    best_key = max(valid_counts, key=valid_counts.get) if valid_counts else max(counts, key=counts.get)
    
    return objective_value, best_key



def minimize(estimator: Estimator,
             hamiltonian: SparsePauliOp,
             generations: int,
             population: int,
             qc: Optional[QuantumCircuit] = None,
             cd_qaoa: Optional[Callable[[QuantumCircuit], QuantumCircuit]] = None,
             ref_value: float = 0,
             tol: float = 1e-0,
             A: list = None,
             verbose: bool = True):
    """
    Minimize a Hamiltonian using evolutionary quantum circuit optimization.
    
    Args:
        estimator: Quantum estimator to run circuits
        hamiltonian: Hamiltonian operator to minimize
        generations: Maximum number of generations for evolution
        population: Number of circuits in each generation
        qc: Optional initial quantum circuit (if None, a random circuit is created)
        cd_qaoa: Optional function to compose with each circuit (e.g., for QAOA)
        ref_value: Reference value for the optimization target
        tol: Tolerance for early stopping based on proximity to ref_value
        A: Constraint matrix for feasibility checking
        verbose: Whether to print progress information
        
    Returns:
        dict: Results of the optimization including best circuit, value, and history
    """
    if A is None:
        A = []
    
    start = time.time()
    
    # Convert Hamiltonian to QUBO form
    Q, const = to_qubo(hamiltonian)
    
    # Initialize tracking variables
    values_list = []
    circuit_list = []
    best_keys_list = []
    depth_list = []
    
    
    current_value = float('inf')
    current_parent = None
    current_best_key = None
    
    try:
        for step in range(generations):
            # Generate initial population or mutate from current best
            if step == 0:
                # Create initial population
                if qc is not None:
                    parent = [qc.copy() for _ in range(population)]
                else:
                    parent = [initial_circuit(hamiltonian.num_qubits) for _ in range(population)]
            else:
                # Create mutations of current best circuit
                parent = [mutation(current_parent.copy()) for _ in range(population)]
            
            # Apply optional composition (e.g., for QAOA)
            cd_parent = parent
            if cd_qaoa is not None:
                cd_parent = [cd_qaoa.compose(cir) for cir in parent]
            
            # Evaluate all circuits in parallel
            results = [calculate(estimator, child, hamiltonian, ref_value, Q, const, A) for child in cd_parent]
            loss_list = [result[0] for result in results]
            key_list = [result[1] for result in results]
            
            # Include current best in the population (elitism)
            if step > 0 and current_parent is not None:
                parent.append(current_parent)
                loss_list.append(current_value)
                key_list.append(current_best_key)
            
            # Find the best circuit in this generation
            
            index = np.argmin(loss_list)
            
            # Update current best if improvement found
            if loss_list[index] <= current_value:
                current_parent = parent[index]
                current_value = loss_list[index]
                current_best_key = key_list[index]
            
            # Track history
            values_list.append(current_value)
            circuit_list.append(current_parent.copy())
            depth_list.append(current_parent.depth())
            best_keys_list.append(current_best_key)
            
            # Print progress
            if verbose:
                print(f"Step: {step + 1}/{generations} | Depth: {current_parent.depth()} | Value: {current_value:.7f} | Key: {current_best_key}", end="\r", flush=True)
            
            # Check for early stopping based on target value
            if current_value - ref_value <= tol:
                if verbose:
                    print(f"\nReached target tolerance at generation {step + 1}")
                break
                
    except Exception as e:
        print(f"Error during optimization: {str(e)}")
        # Return best results so far even if there's an error
    
    end = time.time()
    
    if verbose:
        print(f"\nOptimization completed in {end - start:.2f} seconds")
        print(f"Best value: {current_value:.7f}")
        print(f"Best circuit depth: {current_parent.depth()}")
        print(f"Best Key: {current_best_key}")
    
    # Prepare results dictionary
    results = {
        "generations": len(values_list),
        "values": values_list,
        # "circuits": circuit_list,
        "best_value": current_value,
        "best_circuit": current_parent,
        "best_key": current_best_key,
        "best_keys_history": best_keys_list,
        "depths": depth_list,
        "time": end - start,
        "ration": ref_value / current_value
    }
    
    return results