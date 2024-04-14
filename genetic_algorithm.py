import math
import random
import individual
import numpy as np
import numpy.random
from math import pi
from tqdm import tqdm
from qiskit import QuantumCircuit
from projectq.ops import (H, X, Y, Z, T, Tdagger, S, Sdagger, CNOT, CX, Rx,
                          Ry, Rz, SqrtX, get_inverse, Swap, SwapGate)
from concurrent.futures import ThreadPoolExecutor
from deap.tools.emo import sortNondominated as sort_nondominated
from deap import creator, base, tools
import pennylane as qml
from pennylane import numpy as np
from pennylane.templates import AmplitudeEmbedding

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)


def triplet_loss(z_out):
    return np.abs(z_out[0] - z_out[2]) + np.abs(z_out[1] - z_out[3])


def from_qiskit_circuit(qiskit_circuit):
    produced_circuit = []
    for instr, qargs, cargs in qiskit_circuit.data:
        if instr.name in ['cx', 'cnot']:
            control, target = qargs
            produced_circuit.append(("TFG", CX, qiskit_circuit.qubits.index(control), qiskit_circuit.qubits.index(target)))

        elif instr.name == 'swap':
            control, target = qargs
            produced_circuit.append(("TFG", Swap, qiskit_circuit.qubits.index(control), qiskit_circuit.qubits.index(target)))
        elif instr.name in ['h', 'x', 'y', 'z', 't', 'tdg', 's', 'sdg', 'sx']:
            target = qiskit_circuit.qubits.index(qargs[0])

            gate_dict = {'h': H, 'x': X, 'y': Y, 'z': Z, 't': T, 'tdg': Tdagger, 's': S, 'sdg': Sdagger,
                         'sx': SqrtX}
            produced_circuit.append(("SFG", gate_dict[instr.name], target))
        elif instr.name in ['rx', 'ry', 'rz']:
            target = qiskit_circuit.qubits.index(qargs[0])
            parameter = instr.params[0]
            gate_dict = {'rx': Rx, 'ry': Ry, 'rz': Rz}
            produced_circuit.append(("SG", gate_dict[instr.name], target, parameter))
        else:
            pass

    return produced_circuit


def mutate_ind(individual):
    if individual.optimizedx:
        individual.parameter_mutation()
        return individual
    mutation_choice_fn = random.choice([
        individual.discrete_uniform_mutation,
        individual.continuous_uniform_mutation,
        individual.sequence_insertion,
        individual.sequence_and_inverse_insertion,
        individual.insert_mutate_invert,
        individual.sequence_deletion,
        individual.sequence_replacement,
        individual.sequence_swap,
        individual.sequence_scramble,
        individual.permutation_mutation,
        individual.clean,
        individual.move_gate
    ])
    mutation_choice_fn()
    return individual


@qml.qnode(qml.device(name='default.qubit', wires=14))
def runcircuit(ind, flattened, number_of_qubits):
    my_circuit = qml.load(ind, format='qiskit')
    AmplitudeEmbedding(features=flattened, wires=range(number_of_qubits), normalize=True,
                       pad_with=0)
    my_circuit(wires=range(number_of_qubits))
    expectation = [qml.expval(qml.PauliZ(i)) for i in range(4)]
    return expectation


class Evolution:
    def __init__(self, triplets, connectivity="ALL", EMC=5.0, ESL=5.0):
        self.EMC = EMC
        self.ESL = ESL
        self.CMW = 0.2
        self.optimized = True
        self.population_size = 2
        self.number_of_generations = 1
        self.number_of_qubits = 14
        self.batch = 10
        self.triplets = triplets
        self.connectivity = connectivity
        self.allowed_gates = [Rx, Ry, Rz, CNOT]
        self.base_path = "D:\pycharm_projects\SLIQ-PENNYLANE_mnist/"
        self.crossover_rate = 0.3
        self.dataset = 'landscape'

    def generate_random_circuit(self, initialize=True):
        if initialize:
            p = 1 / 30
        else:
            p = 1 / self.ESL
        cir_length = 60 + numpy.random.geometric(p)
        produced_circuit = []
        quantum_circuit = QuantumCircuit(self.number_of_qubits)
        for i in range(cir_length):
            gate = random.choice(self.allowed_gates)
            if gate in [CNOT, CX, Swap]:
                if self.connectivity == "2D":
                    if self.number_of_qubits == 4:
                        control, target = random.choice([(0, 1), (1, 0), (1, 2), (2, 1), (2, 3), (3, 2)])
                        produced_circuit.append(('TFG', gate, control, target))
                        quantum_circuit.append(gate, [control, target])
                    if self.number_of_qubits == 8:
                        control, target = random.choice(
                            [(0, 1), (1, 0), (1, 2), (2, 1), (2, 3), (3, 2), (4, 5), (5, 4), (5, 6), (6, 5), (6, 7),
                             (7, 6)])
                        produced_circuit.append(('TFG', gate, control, target))
                        quantum_circuit.append(gate, [control, target])
                    if self.number_of_qubits == 14:
                        control, target = random.choice(
                            [(0, 1), (1, 0), (1, 2), (2, 1), (2, 3), (3, 2), (4, 5), (5, 4), (5, 6), (6, 5), (6, 7),
                             (7, 6), (8, 9), (9, 8), (9, 10), (10, 9), (10, 11), (11, 10), (12, 13), (13, 12)])
                        produced_circuit.append(('TFG', gate, control, target))
                        quantum_circuit.append(gate, [control, target])
                else:
                    control, target = sorted(numpy.random.choice(self.number_of_qubits, 2, replace=False))
                    produced_circuit.append(('TFG', gate, control, target))
                    quantum_circuit.append(gate, [control, target])
            else:
                qubit = random.choice(range(self.number_of_qubits))
                if gate in [Rx, Ry, Rz]:
                    parameter = random.uniform(0, 2 * pi)
                    produced_circuit.append(("SG", gate, qubit, parameter))
                    quantum_circuit.append(gate(parameter), [qubit])
        produced_circuit = [(f, g, *t) for f, g, *t in produced_circuit]
        return produced_circuit, quantum_circuit

    def generate_population(self, size, initialize=True):
        population = []
        for _ in range(size):
            ind = individual.Individual(*self.generate_random_circuit(initialize))
            population.append(ind)
        return population

    def evaluate_individuals(self, individuals):
        batch = min(self.batch, len(individuals))
        num_batches = math.ceil(len(individuals) / batch)
        batches = [individuals[i * batch:(i + 1) * batch] for i in range(num_batches)]
        results = []

        for b in batches:
            results += self.run_batch(b)

        return results

    def run_batch(self, individuals):
        with ThreadPoolExecutor() as executor:
            future_to_individual = {
                executor.submit(self.run_individual, ind): ind for ind in individuals
            }
            results = []
            for future in tqdm(concurrent.futures.as_completed(future_to_individual)):
                ind = future_to_individual[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as exc:
                    print(f'Generated an exception: {exc}')
        return results

    def run_individual(self, ind):
        flattened = self.triplets
        result = runcircuit(ind, flattened, self.number_of_qubits)
        return result

    def crossover(self, individual1, individual2):
        offspring1 = individual1.deepcopy()
        offspring2 = individual2.deepcopy()
        for gate1 in offspring1:
            if gate1[0] == "SG" or gate1[0] == "TFG":
                continue
            for gate2 in offspring2:
                if gate2[0] == "SG" or gate2[0] == "TFG":
                    continue
                if gate1[1] == gate2[1] and gate1[0] == "SFG" and gate2[0] == "SFG":
                    temp = gate1[2]
                    gate1[2] = gate2[2]
                    gate2[2] = temp
                    break
        return offspring1, offspring2

    def mutate(self, individual):
        return mutate_ind(individual)


