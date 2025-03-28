from qiskit.quantum_info import SparsePauliOp
import gurobipy as gb
from gurobipy import GRB
import itertools
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
import numpy as np

def build_hamiltonian(parameters):
    """
    Combines the calculation of Hamiltonian coefficients and the creation of a SparsePauliOp
    in a single function for improved efficiency.
    """
    w, c, A = parameters[0], parameters[1], parameters[2]
    m = len(A)
    n = len(w)
    
    coeff = {tuple([i]): 0 for i in range(n)}
    coeff[()] = 0
    for i in range(n):
        for j in range(n):
            if i != j:
                coeff[(i, j)] = 0
    for i in range(n):
        coeff[()] += w[i] / 2
        coeff[tuple([i])] -= w[i] / 2
    
    for i in range(m):
        temp = 0
        for k in range(n):
            for j in range(n):
                a = c[i] * A[i][j] * A[i][k] / 4
                coeff[()] += a
                coeff[tuple([j])] -= a
                coeff[tuple([k])] -= a
                if j == k:
                    coeff[()] += a
                else:
                    coeff[tuple([j, k])] += a
            coeff[()] -= A[i][k] * c[i]
            coeff[tuple([k])] += A[i][k] * c[i]
        coeff[()] += c[i]
    
    # Convert coefficients into SparsePauliOp
    pauli_terms = []
    for var, value in coeff.items():
        if value != 0.0:
            pauli_string = ['I'] * n
            for idx in var:
                pauli_string[idx] = 'Z'
            pauli_terms.append((''.join(pauli_string), value))
    
    operator_prob = SparsePauliOp.from_list(pauli_terms)
    return operator_prob

def read_instance(fname):
    f = open(fname, "r")
    lines = f.readlines()
    size = int(lines[0])
    w = [int(x) for x in lines[1].split(",")]
    c = [int(x) for x in lines[2].split(",")]
    A = []
    for i in range(3,len(lines)):
        temp = lines[i][:-1].split(" ")
        temp = [int(x) for x in temp]
        A.append(temp)
    return w,c,A

def sp_objective(w,x):
    """
    Objective function of the set partitioning problem
    :param x: binary vector of length n
    :param c: vector of length n
    :return: value of the objective function
    """
    obj = 0
    for i in range(len(w)):
        obj += w[i]*x[i]
    return obj

def sp_constraint(A,x):
    """
    Constraint function of the set partitioning problem
    :param A: matrix of size nxn
    :param x: binary vector of length n
    :return: value of the constraint function
    """
    cons = []
    for i in range(len(A)):
        temp = 0
        for j in range(len(x)):
            temp += A[i][j]*x[j]
        cons.append(temp)
    return cons

def sp_gurobi(w,A):
    """
    Classical solution using guroby for the set partitioning problem
    :param c: vector of length n
    :param A: matrix of size nxn
    :return: binary vector of length n
    """
    with gb.Env(empty=True) as env:
        env.setParam('OutputFlag', 0)
        env.start()
        with gb.Model(env=env) as model:
            model.Params.LogToConsole = 0
            n = len(w)
            
            x = model.addVars(n,vtype=gb.GRB.BINARY,name='x')
            
            model.setObjective(sp_objective(w,x),GRB.MINIMIZE)
            model.addConstrs(sp_constraint(A,x)[i] == 1 for i in range(len(A)))
            model.optimize()
            return model.x

def ansatz_adiabatic(n_qubits):
    
    p = 1  # Number of layers

    # Parameters for a QAOA layer
    gamma = 0
    beta = 0.5

    # Manually constructing a QAOA layer for `p=1`
    qaoa_layer = QuantumCircuit(n_qubits)

    # Applying a rotation around Z for each qubit (Problem Hamiltonian)
    for qubit in range(n_qubits):
        qaoa_layer.h(qubit)
        qaoa_layer.rz(2 * gamma, qubit)

    # Applying CNOTs between adjacent qubits for the entanglement term
    for qubit in range(n_qubits - 1):
        qaoa_layer.cx(qubit, qubit + 1)
        qaoa_layer.rz(2 * gamma, qubit + 1)
        qaoa_layer.cx(qubit, qubit + 1)

    # Applying a rotation around X for each qubit (Mixer Hamiltonian)
    for qubit in range(n_qubits):
        qaoa_layer.rx(2 * beta, qubit)
    
    return qaoa_layer

def to_qubo(hamiltoniano: SparsePauliOp):
    """
    Converts a Hamiltonian of type SparsePauliOp (assuming it is diagonal, i.e., containing only 'I' and 'Z')
    into a QUBO formulation, where the cost function is defined by:
        f(x) = x^T Q x + constant,
    with x binary variables (0,1).

    Parameters:
        hamiltoniano (SparsePauliOp): Hamiltonian to be converted.

    Returns:
        Q (np.ndarray): Q matrix of the QUBO formulation.
        constante (float): Constant term resulting from the conversion.
    """
    n = hamiltoniano.num_qubits
    Q = np.zeros((n, n))
    constante = 0.0

    # Helper function that generates all subsets (including the empty one) of a list
    def obter_subconjuntos(lista):
        for r in range(len(lista) + 1):
            for subset in itertools.combinations(lista, r):
                yield subset

    # Iterate through each term of the Hamiltonian
    for pauli, coef in zip(hamiltoniano.paulis, hamiltoniano.coeffs):
        label = pauli.to_label()
        # Check if the term is diagonal (only I and Z are allowed)
        if any(ch not in ['I', 'Z'] for ch in label):
            raise ValueError("The Hamiltonian contains non-diagonal terms; conversion to QUBO is not straightforward.")
        
        # Indices where the operator is 'Z'
        indices_Z = [i for i, ch in enumerate(label) if ch == 'Z']
        
        # Expand the product ∏ (1 - 2x_j) for qubits where there is 'Z'
        for subset in obter_subconjuntos(indices_Z):
            termo_coef = coef * ((-2) ** len(subset))
            if len(subset) == 0:
                # Constant contribution
                constante += termo_coef
            elif len(subset) == 1:
                # Linear term: sum on the diagonal of Q
                i = subset[0]
                Q[i, i] += termo_coef
            elif len(subset) == 2:
                # Quadratic term: off-diagonal position
                i, j = subset
                Q[i, j] += termo_coef
            else:
                # Terms of order higher than 2 are not directly representable in QUBO
                raise NotImplementedError("Terms of order higher than 2 detected; conversion to QUBO requires additional techniques.")
    
    return Q, constante

def qubo_evaluation(Q, constante, assignment):
    """
    Evaluates the cost function of the QUBO problem for a given variable assignment.
    
    The cost function is defined by:
        f(x) = constante + sum_{i} Q[i,i]*x_i + sum_{i<j} Q[i,j]*x_i*x_j
    where 'assignment' is a string representing the binary values of the variables,
    for example, '11000' corresponds to x0=1, x1=1, x2=0, x3=0, x4=0.
    
    Parameters:
        Q (np.ndarray): Q matrix of the QUBO formulation.
        constante (float): Constant term.
        assignment (str): String with the assignment of binary variables.
        
    Returns:
        custo (float): Numerical value of the cost function evaluated at the assignment.
    """
    # Convert the string to a list of integers (0 or 1)
    x = [int(bit) for bit in assignment]
    n = Q.shape[0]
    
    if len(x) != n:
        raise ValueError("The size of the assignment must be equal to the number of variables.")
    
    custo = constante
    
    # Sum the linear terms
    for i in range(n):
        custo += Q[i, i] * x[i]
    
    # Sum the quadratic terms (i < j)
    for i in range(n):
        for j in range(i+1, n):
            custo += Q[i, j] * x[i] * x[j]
    
    return custo.real

def check_keys(A, counts_dict):
    """
    Receives:
    - A: list of lists representing the matrix (each row is a list of integers);
    - counts_dict: dictionary in the format {'000101': 1024, '000111': 104, ...}.
    
    For each key in the dictionary:
      1. Converts the key (binary string) into a column vector.
      2. Calculates the matrix product: A * column_vector.
      3. If any element of the resulting vector is different from 1, adds the key to the output list.
    
    Returns:
    - A list with the keys for which some element of the result is not 1.
    
    Note:
    The integer value associated with each key is not used in the multiplication.
    """
    matriz_A = np.array(A)
    chaves_invalidadas = []
    
    for key in counts_dict:
        # Convert the binary string to a column vector
        vetor = np.array([int(c) for c in key]).reshape(-1, 1)
        resultado = np.dot(matriz_A, vetor)
        # If any element of the result is different from 1, add the key
        if not np.all(resultado == 1):
            chaves_invalidadas.append(key)
    
    return chaves_invalidadas