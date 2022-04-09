from qiskit import IBMQ
from qiskit import transpile
from qiskit.circuit import qpy_serialization

IBMQ.load_account()
provider = IBMQ.get_provider(group="open")
backend = provider.get_backend("ibmq_quito")

file = 'hard_slsqp_ra_rep1_linear_nobarriers.qpy'

with open(file, 'rb') as fd:
    circuit = qpy_serialization.load(fd)[0]

transpiled = transpile(circuit, backend, optimization_level=3)
circuit.measure_all()
job = backend.run(circuit, shots=16)
print(file, job.job_id())
