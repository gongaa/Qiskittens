#!/usr/bin/env python3
# coding: utf-8
import numpy as np
import copy

# Problem modelling imports
from docplex.mp.model import Model

# Qiskit imports
from qiskit import BasicAer
# from qiskit.utils.algorithm_globals import algorithm_globals
from qiskit_optimization.problems.variable import VarType
from qiskit_optimization.converters.quadratic_program_to_qubo import QuadraticProgramToQubo
from qiskit_optimization.translators import from_docplex_mp

from qiskit.utils import algorithm_globals, QuantumInstance
from qiskit.algorithms import QAOA, NumPyMinimumEigensolver
from qiskit_optimization.algorithms import (
    CplexOptimizer,
    MinimumEigenOptimizer,
    RecursiveMinimumEigenOptimizer,
    SolutionSample,
    OptimizationResultStatus,
    GroverOptimizer
)
from qiskit_optimization import QuadraticProgram
from qiskit.visualization import plot_histogram
from typing import List, Tuple
import numpy as np
from qiskit.quantum_info import Statevector
from qiskit.opflow import PauliExpectation, StateFn, CircuitSampler



def create_problem(mu: np.array, sigma: np.array, total: int = 3) -> QuadraticProgram:
    """Solve the quadratic program using docplex."""

    mdl = Model()
    x = [mdl.binary_var("x%s" % i) for i in range(len(sigma))]

    objective = mdl.sum([mu[i] * x[i] for i in range(len(mu))])
    objective += mdl.sum(
        [sigma[i, j] * x[i] * x[j] for i in range(len(mu)) for j in range(len(mu))]
    )
    mdl.minimize(objective)
#     cost = mdl.sum(x)
#     mdl.add_constraint(cost == total)

    qp = from_docplex_mp(mdl)
    return qp


def relax_problem(problem) -> QuadraticProgram:
    """Change all variables to continuous."""
    relaxed_problem = copy.deepcopy(problem)
    for variable in relaxed_problem.variables:
        variable.vartype = VarType.CONTINUOUS

    return relaxed_problem


# Easy
#h = np.array([396, 752, 1204, 1177])
#d = np.array([1040, 2480, 7960, 3230])
#T = 5
#M = 1000.0

# Hard
h = np.array([870, 2501, 4478, 1777, 2970])
d = np.array([2700, 6500, 15000, 100, 10000])
T = 13.5
M = 1000

t = h/(M/2.0) + d/(5.0*M)
print(t)
Q = np.outer(t, t)
print(Q)
b = -2.0 * T * t
# print(b)
# Q = Q + np.diag(b)
# print(Q)


import itertools

values = []
print("x, cost")
for x in itertools.product([0, 1], repeat=t.size):
    values.append((x, (T-np.asarray(x).dot(t))**2))
    print(*values[-1])

optimal = min(values, key=lambda value: value[1])
print("Optimal solution:", *optimal)


qubo = create_problem(b, Q)
qubo


H, offset = qubo.to_ising()
print("offset: {}".format(offset))
print("operator:")
print(H)

H.to_matrix()


exact = NumPyMinimumEigensolver().compute_minimum_eigenvalue(H)
exact.eigenstate

statevector = exact.eigenstate.primitive
print("Optimal state from Hamiltonian:")
print(statevector.probabilities_dict())
print("Value:")
print(-exact.eigenvalue)

print("\nCompare to orignal solution:")
print(optimal)


# ## B. Finding the minimum

# ### Evaluating the cost function
# 
# First, construct the QAOA ansatz and then write a function to evaluate the energy. Given a quantum circuit, you can evaluate the expectation value as


from qiskit.circuit.library import QAOAAnsatz, GroverOperator, RealAmplitudes


ra = RealAmplitudes(5, reps=1, entanglement='linear', insert_barriers=False)

from qiskit.test.mock import FakeQuito
backend=FakeQuito()
backend.configuration().basis_gates


from qiskit import transpile
from qiskit.algorithms.optimizers import SLSQP

def exact_energy(theta):
    bound = ra.bind_parameters(theta)
    return np.real(Statevector(bound).expectation_value(H.primitive))



# Turn off qiskit parallelization
import os
os.environ['QISKET_IN_PARALLEL'] = 'True'

# Set up parallelization
import multiprocessing as mp
from tqdm import tqdm 
import itertools

THREADS = 8
MINIMZATIONS = 128  # total minimzations to try
LAYOUTS = 128  # per permutation



################################################################################
# Energy optimization
################################################################################

# Pick the optimizer and set random initial parameters from the circuit
optimizer = SLSQP()
initial_parameters = np.random.random(ra.num_parameters)

# Parallel minimizations
def find_opt(unused):
    minfev = 1e6
    return optimizer.minimize(exact_energy, initial_parameters)

# Define the thread pool
pool_obj = mp.Pool(processes=THREADS)
    
# Dispatch a bunch of minimzations in parallel
min_result = list(tqdm(pool_obj.imap(find_opt, list(range(MINIMZATIONS))), total=128))

# Get the minimum fev count
minfev = 1e6
for r in min_result:
    fev = r.nfev
    if fev < minfev:
        minfev = fev
        result = r
print('nfev:', minfev)

# Bind the parameters to our circuit
circuit = ra.decompose().decompose().decompose().bind_parameters(result.x)



################################################################################
# Transpile
################################################################################

# Parallel transpilations
def find_min(initial_layout):
    mindepth = 1024
    for _ in range(LAYOUTS):
        tp = transpile(circuit, backend, initial_layout=initial_layout, optimization_level=3)
        depth = tp.count_ops()['cx']
        if depth < mindepth:
            mindepth = depth
            transpiled = tp
    return transpiled

# Define the thread pool
pool_obj = mp.Pool(processes=THREADS)

# Dispatch a bunch of transpilations for each possible inital layout, in parallel
layouts = list(itertools.permutations([0, 1, 2, 3, 4]))
min_tp = list(tqdm(pool_obj.imap(find_min, layouts), total=len(layouts)))

# Get the minimum cnot count
mindepth = 1024
for tp in min_tp:
    depth = tp.count_ops()['cx']
    if depth < mindepth:
        mindepth = depth
        transpiled = tp
print('Depth:', transpiled.depth())
print('Gate count:', transpiled.count_ops())


# Save the circuit
from qiskit.circuit import qpy_serialization
with open('hard_slsqp.qpy', 'wb') as fd:
    qpy_serialization.dump(transpiled, fd)

# Save the result
with open('hard_slsqp_result.txt', 'w') as f:
    f.write(str(result))
