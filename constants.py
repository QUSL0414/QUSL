from qiskit.test.mock import FakeVigo
from tools import projectq_of_string

NUMBER_OF_QUBITS = 14

NUMBER_OF_GENERATIONS = 16
POPULATION_SIZE = 15
SAVE_RESULT = True
NUM_TRIPLETS = 5000

# Backend configurations for noise simulation
FAKE_MACHINE = FakeVigo()
BASIS_GATES = FAKE_MACHINE.configuration().basis_gates

ALLOWED_GATES = [projectq_of_string(gate)
                 for gate in BASIS_GATES if gate != 'id']

CONNECTIVITY = FAKE_MACHINE.configuration().coupling_map
MAX_CIRCUIT_LENGTH = 10

FITNESS_WEIGHTS = (-1.0, -0.5)
