import os
import csv
import bisect
import math
import time
import random
import individual
import numpy as np
import numpy.random
import multiprocessing
from math import pi
from qiskit import *
from tqdm import tqdm
# import pennylane_qiskit
import datetime
from qiskit import transpile, assemble
from functools import partial
from tools import string_of_projectq
from qiskit import QuantumCircuit, Aer
from qiskit.visualization import circuit_drawer
from projectq.ops import (H, X, Y, Z, T, Tdagger, S, Sdagger, CNOT, CX, Rx,
                          Ry, Rz, SqrtX, get_inverse, Swap, SwapGate)
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from deap.tools.emo import sortNondominated as sort_nondominated
from deap import creator, base, tools
import pennylane as qml
from pennylane import numpy as np
from pennylane.templates import AmplitudeEmbedding
import matplotlib.pyplot as plt

creator.create("FitnessMax", base.Fitness, weights=(1.0,))

creator.create("Individual", list, fitness=creator.FitnessMax)


def triplet_loss(z_out):
    return np.abs(z_out[0] - z_out[2]) + np.abs(z_out[1] - z_out[3])


def from_qiskit_circuit(qiskit_circuit):
    produced_circuit = []
    for instr, qargs, cargs in qiskit_circuit.data:
        if instr.name in ['cx', 'cnot']:
            control, target = qargs
            produced_circuit.append(
                ("TFG", CX, qiskit_circuit.qubits.index(control), qiskit_circuit.qubits.index(target)))

        elif instr.name == 'swap':
            control, target = qargs
            produced_circuit.append(
                ("TFG", Swap, qiskit_circuit.qubits.index(control), qiskit_circuit.qubits.index(target)))
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
            print(f"Unsupported gate: {instr.name}")

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
        # self.base_path = "/home/ubuntu/ylh/"
        self.base_path = "D:\pycharm_projects\SLIQ-PENNYLANE_mnist/"
        self.crossover_rate = 0.3
        self.dataset = 'landscape'

    def generate_random_circuit(self, initialize=True):

        if initialize:
            p = 1 / 30
        else:
            p = 1 / self.ESL
        cir_length = 60 + numpy.random.geometric(p)
        print('cir_length', cir_length)
        produced_circuit = []
        quantum_circuit = QuantumCircuit(self.number_of_qubits)
        for i in range(cir_length):
            gate = random.choice(self.allowed_gates)
            if gate in [CNOT, CX, Swap]:
                if self.connectivity == "ALL":
                    control, target = random.sample(
                        range(self.number_of_qubits), 2)
                else:
                    control, target = random.choice(self.connectivity)
                    print("control, target:", control, target)
                # TFG stands for Two Qubit Fixed Gate
                produced_circuit.append(("TFG", gate, control, target))

            elif gate in [H, X, Y, Z, ]:
                # choose the index to apply
                target = random.choice(range(self.number_of_qubits))
                # SFG stands for Single Qubit Fixed Gate
                produced_circuit.append(("SFG", gate, target))
            elif gate in [Rx, Ry, Rz]:
                target = random.choice(range(self.number_of_qubits))
                significant_figure = 2 
                parameter = round(pi * random.uniform(0, 2),
                                  significant_figure)
                produced_circuit.append(("SG", gate, target, parameter))
                quantum_circuit.rx(parameter, target)
            else:
                print("Unexpected gate:", gate)
        return produced_circuit

    def to_qiskit_circuit(self, produced_circuit):
        """
        Returns: qiskit.QuantumCircuit object of the circuit of the Candidate
        """
        qr = QuantumRegister(self.number_of_qubits)
        cr = ClassicalRegister(self.number_of_qubits)
        qc = QuantumCircuit(qr, cr)
        for op in produced_circuit:
            if op[0] == "TFG":
                # can be CNOT,CX,Swap,SwapGate
                if op[1] in [CX, CNOT]:
                    qc.cx(op[2], op[3])
                elif op[1] in [Swap, SwapGate]:
                    qc.swap(op[2], op[3])
                else:
                    print("Problem in to_qiskit_circuit:", op[1])

            elif op[0] == "SFG":
                # can be H,X,Y,Z,T,T^d,S,S^d,sqrtX,sqrtXdagger
                if op[1] == H:
                    qc.h(op[2])
                elif op[1] == X:
                    qc.x(op[2])
                elif op[1] == Y:
                    qc.y(op[2])
                elif op[1] == Z:
                    qc.z(op[2])
                elif op[1] == T:
                    qc.t(op[2])
                elif op[1] == Tdagger:
                    qc.tdg(op[2])
                elif op[1] == S:
                    qc.s(op[2])
                elif op[1] == Sdagger:
                    qc.sdg(op[2])
                elif op[1] == SqrtX:
                    qc.sx(op[2])
                elif op[1] == get_inverse(SqrtX):
                    qc.sxdg(op[2])
                else:
                    print("Problem in to_qiskit_circuit:", op[1])

            elif op[0] == "SG":
                # can be Rx,Ry,Rz
                if op[1] == Rx:
                    qc.rx(op[3], op[2])
                elif op[1] == Ry:
                    qc.ry(op[3], op[2])
                elif op[1] == Rz:
                    qc.rz(op[3], op[2])
                else:
                    print("Problem in to_qiskit_circuit:", op[1])
        return qc

    def optimize(self, cir, optimization_level=2):
        basis = [string_of_projectq(gate) for gate in self.allowed_gates]
        if self.connectivity == "ALL":
            qc = transpile(
                cir,
                optimization_level=optimization_level,
                basis_gates=basis,
                layout_method="trivial",
            )
        else:
            qc = transpile(
                cir,
                optimization_level=optimization_level,
                basis_gates=basis,
                layout_method="trivial",
            )
        self.optimized = True
        return qc

    def new_individual(self):
        qr = QuantumRegister(self.number_of_qubits)
        cr = ClassicalRegister(self.number_of_qubits, 'creg')
        circuit = QuantumCircuit(qr, cr, name="genetic")

        for qubit in range(self.number_of_qubits):
            circuit.h(qubit)
        circuit.compose(
            self.optimize(self.to_qiskit_circuit(self.generate_random_circuit(initialize=True)), optimization_level=2),
            list(range(self.number_of_qubits)), inplace=True)
        return circuit  # quantum_circuit

    def new_pop(self, toolbox):
        first_pop = []
        x_train_batch = random.sample(self.triplets, self.batch)
        for i in range(self.population_size + 5):
            circuit = self.new_individual()  # quantum_circuit
            circuit_drawer(circuit, output='mpl').show()
            ind = creator.Individual(circuit, self.number_of_qubits)
            ind.fitness.values = toolbox.evaluate(self, ind=circuit, features=x_train_batch,
                                                  number_of_qubits=self.number_of_qubits)
            first_pop.append(ind)
            qasm_string = circuit.qasm()
            file_name = f'{self.base_path}/result/{self.dataset}/qasm/{self.dataset}_individual{i}.qasm'

            with open(file_name, "w") as file:
                file.write(qasm_string)
            qc_ind = QuantumCircuit.from_qasm_file(file_name)
            circuit_drawer(qc_ind, output='mpl',
                           filename=f'{self.base_path}/result/{self.dataset}/png/{self.dataset}_individual{i}_{ind.fitness.values[0]}.png')

        fitness_pop = [ind.fitness.values[0] for ind in first_pop]
        return first_pop, fitness_pop  # class and fitness

    def continued_pop(self, dataset, toolbox):
        pop = []
        x_train_batch = random.sample(self.triplets, self.batch)
        j = 6
        for i in range(1, 21):
            individual_path = f'{self.base_path}/result/{dataset}/qasm/{dataset}_next_generation_{j}_{i}.qasm'
            if not os.path.exists(individual_path):
                print('11')
                break
            circuit = QuantumCircuit.from_qasm_file(individual_path)
            circuit_drawer(circuit, output='mpl').show()
            ind = creator.Individual(circuit, self.number_of_qubits)
            ind.fitness.values = toolbox.evaluate(self, ind=circuit, features=x_train_batch,
                                                  number_of_qubits=self.number_of_qubits)
            pop.append(ind)
        fitness_pop = [ind.fitness.values[0] for ind in pop]
        return pop, fitness_pop  # class and fitness

    def mate(self, parent1, parent2, toolbox):
        return parent1.cross_over(parent2, toolbox), parent2.cross_over(parent1, toolbox)

    def select_parents(self, pop, num_parents, toolbox):
        parents = []
        x_train_batch = random.sample(self.triplets, self.batch)
        if len(pop) < num_parents:
            num_new_ind = self.population_size-len(pop)
            for i in range(num_new_ind+2):
                circuit = self.new_individual()  # quantum_circuit
                ind = creator.Individual(circuit, self.number_of_qubits)
                pop.append(ind)
                ind.fitness.values = toolbox.evaluate(self, ind=circuit, features=x_train_batch,
                                                      number_of_qubits=self.number_of_qubits)

        fitness_values = [ind.fitness.values[0] for ind in pop]
        sorted_indices = sorted(range(len(fitness_values)), key=lambda k: fitness_values[k],
                                reverse=True)
        for i in range(num_parents):
            parents.append(pop[sorted_indices[i]])
        return parents
    def select(self, pop, to_carry, fitness_of_pop):

        normalized_fitness_values = [fitness / sum(fitness_of_pop) for fitness in fitness_of_pop]
        cumulative_fitness_values = [sum(normalized_fitness_values[:i + 1]) for i in
                                     range(len(normalized_fitness_values))]
        selected_individuals = []
        selected_ind_fit = []
        for i in range(to_carry):
            random_number = random.random()
            index = bisect.bisect_left(cumulative_fitness_values, random_number)
            selected_individuals.append(pop[index])
            selected_ind_fit.append(fitness_of_pop[index])
        return selected_individuals, selected_ind_fit

    def tournament_selection(self, pop, fitness_of_pop, tournament_size):
        selected_individuals = []
        selected_ind_fit = []
        remaining_pop = list(zip(pop, fitness_of_pop))

        for _ in range(4):
            tournament = random.sample(remaining_pop, tournament_size)
            tournament.sort(key=lambda x: x[1], reverse=True) 
            winner = tournament[0][0]
            winner_fitness = tournament[0][1]
            selected_individuals.append(winner)
            selected_ind_fit.append(winner_fitness)
            remaining_pop.remove((winner, winner_fitness))
        return selected_individuals, selected_ind_fit

    def elitism_selection(self, pop, to_carry, fitness_of_pop):
        best_index = np.argmax(fitness_of_pop)
        selected_individuals = [pop[best_index]] * to_carry
        selected_ind_fit = [fitness_of_pop[best_index]] * to_carry
        return selected_individuals, selected_ind_fit

    def crossover(self, pop, crossover, toolbox):
        mate_individual = []
        num_cross = int(crossover)
        num_parents = 6
        parents = self.select_parents(pop, num_parents, toolbox)
        for _ in range(num_cross):
            parent1, parent2 = random.sample(parents, 2)
            circuit1 = from_qiskit_circuit(parent1.circuit) 
            individuals1 = individual.Individual(circuit1, self.number_of_qubits)
            circuit2 = from_qiskit_circuit(parent1.circuit)
            individuals2 = individual.Individual(circuit2, self.number_of_qubits)
            child1, child2 = self.mate(individuals1, individuals2, toolbox) 
            mate_individual.append(creator.Individual(self.to_qiskit_circuit(child1.circuit), self.number_of_qubits))
            mate_individual.append(creator.Individual(self.to_qiskit_circuit(child2.circuit), self.number_of_qubits))
        return mate_individual  # creator class

    def mutate_individuals(self, ranks, N, toolbox, current_rank=1):
        L = len(ranks)
        T = 0
        for i in range(L):
            T += math.exp(-current_rank - i)
            T += math.exp(-current_rank - i)

        cps = []

        for _ in range(N):
            random_number = random.uniform(0, T)

            list_index = -1
            right_border = 0
            for i in range(L):
                right_border += math.exp(-current_rank - i)
                if random_number <= right_border:
                    list_index = i
                    break
            if list_index == -1:
                list_index = L - 1
            left_border = right_border - math.exp(-current_rank - list_index)
            element_index = math.floor(
                len(ranks[list_index]) * (random_number - left_border) / (right_border - left_border))

            while len(ranks[list_index]) == 0:
                list_index += 1
                if len(ranks[list_index]) != 0:
                    element_index = random.choice(range(len(ranks[list_index])))

            if element_index >= len(ranks[list_index]):
                element_index = -1

            cp = deepcopy(ranks[list_index][element_index])
            circuit = from_qiskit_circuit(cp.circuit)
            individuals = individual.Individual(circuit, self.number_of_qubits)
            cp_mutat = toolbox.mutate_ind(individuals)
            new_circuit = cp_mutat.circuit 
            circuit = self.to_qiskit_circuit(new_circuit)
            cp_class = creator.Individual(circuit, self.number_of_qubits)
            cps.append(cp_class)
        print('len-cps:', len(cps))
        return cps

    def mutate_individuals_self(self, pop, toolbox):
        mutate_individual = self.select_parents(pop, 4, toolbox)
        fitness_individual = [ind.fitness.values[0] for ind in mutate_individual]
        return mutate_individual, fitness_individual

    def select_and_evolve(self, pop, fitness_of_pop, toolbox):
        """
        Apply nondominated sorting to rank individuals and select individuals for the next generation,
        then mutate and perform crossover to generate the next generation.
        """
        ranks = sort_nondominated(pop, len(pop)) 
        to_carry = len(ranks[0])
        individuals, individuals_fit = self.tournament_selection(pop, fitness_of_pop,
                                                                 tournament_size=5)

        print(f'after: select fitness_of_pop\n {individuals_fit}')
        next_generation = individuals
        crossover = int(len(pop) * self.crossover_rate)
        N = len(pop) - to_carry - crossover
        N = N if N > 0 else 1
        mutated_individuals = self.mutate_individuals(ranks, N, toolbox, current_rank=1)
        next_generation.extend(mutated_individuals)
        mate_individuals = self.crossover(pop, crossover, toolbox)
        next_generation.extend(mate_individuals)
        x_train_batch = random.sample(self.triplets, self.batch)
        for ind in next_generation:
            ind.fitness.values = toolbox.evaluate(self, ind=ind.circuit, features=x_train_batch,
                                                  number_of_qubits=self.number_of_qubits)

        fitness_dict = {}
        for ind in next_generation:
            fitness = ind.fitness.values[0]
            if fitness in fitness_dict:
                fitness_dict[fitness].append(ind)
            else:
                fitness_dict[fitness] = [ind]

        new_next_generation = []
        for fitness, individuals in fitness_dict.items():
            if len(individuals) > 1:
                new_next_generation.append(individuals[0])
            else:
                new_next_generation.extend(individuals)

        next_generation = new_next_generation

        x_train_batch = random.sample(self.triplets, self.batch)
        if len(next_generation) < self.population_size:
            num_new_ind = self.population_size - len(next_generation)
            for i in range(num_new_ind):
                circuit = self.new_individual()  # quantum_circuit
                ind = creator.Individual(circuit, self.number_of_qubits)
                ind.fitness.values = toolbox.evaluate(self, ind=circuit, features=x_train_batch,
                                                      number_of_qubits=self.number_of_qubits)
                next_generation.append(ind)

        for ind in next_generation:
            circuit_drawer(ind.circuit, output='mpl').show()
        fitness_pop = [ind.fitness.values[0] for ind in next_generation]
        print(f'next_generations{fitness_pop}\nlen{len(fitness_pop)}')
        return next_generation  # creator_class

    def evolution(self, dataset, toolbox):
        start = time.perf_counter()
        initial_pop, fitness_pop = self.new_pop(toolbox)
        runtime = round(time.perf_counter() - start, 2)
        print("runtime new_pop", runtime)
        combined_data = list(zip(initial_pop, fitness_pop))
        sorted_data = sorted(combined_data, key=lambda x: x[1], reverse=True)
        sorted_initial_pop, sorted_fitness_pop = zip(*sorted_data) 
        sorted_next_generation = sorted_initial_pop[:self.population_size]
        sorted_fitness_values = sorted_fitness_pop[:self.population_size]

        for i in tqdm(range(self.number_of_generations)):
            print('number_of_generations:', i)
            timestamp = int(time.time())
            start = time.perf_counter()
            next_generation = self.select_and_evolve(sorted_next_generation[:self.population_size], sorted_fitness_values[:self.population_size],
                                                     toolbox)  # creator class
            runtime = round(time.perf_counter() - start, 2)
            print("runtime select_and_evolve", runtime)
            print('len_next_generation', len(next_generation))
            fitness_values = [ind.fitness.values[0] for ind in next_generation]
            print(f'next_generation_fitness_values{fitness_values}')

            combined_data = list(zip(next_generation, fitness_values))
            sorted_data = sorted(combined_data, key=lambda x: x[1], reverse=True)
            sorted_next_generations, sorted_fitness_valuess = zip(*sorted_data)
            sorted_next_generation = sorted_next_generations[:self.population_size]
            sorted_fitness_values = sorted_fitness_valuess[:self.population_size]
            print(f'sorted_next_generation sorted_fitness_values{sorted_fitness_values}')

            filename = f'{self.base_path}/result/{self.dataset}/csv/{self.dataset}_fitness_values_ranks_{i + 1}.csv'
            with open(filename, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['Index', 'Fitness Value'])
                for j, fitness in enumerate(sorted_fitness_values):
                    writer.writerow([j + 1, fitness])
            count = 1

            for ind in sorted_next_generation:
                qasm_string = ind.circuit.qasm()
                file_name = f'{self.base_path}/result/{self.dataset}/qasm/{self.dataset}_next_generation_{i + 1}_{count}.qasm'
                with open(file_name, "w") as file:
                    file.write(qasm_string)
                qc_ind = QuantumCircuit.from_qasm_file(file_name)
                circuit_drawer(qc_ind, output='mpl',
                               filename=f'{self.base_path}/result/{self.dataset}/png/{self.dataset}_next_generation_{i+1}_{count}_{ind.fitness.values[0]}.png')
                count += 1
            print('len-next-generation:', len(sorted_next_generation))
        print(f'last sorted_next_generation{sorted_next_generation},\n sorted_fitness_values{sorted_fitness_values}')
        return sorted_next_generation[:self.population_size], sorted_fitness_values[:self.population_size]

    def process_feature(self, ind, number_of_qubits, im):
        print('x_train_batch:', np.array(im[0]).shape)
        flattened = []
        for i, j, k in zip(im[0], im[1], im[2]):
            flattened.append(i)
            flattened.append(j)
        z_out1 = runcircuit(ind, flattened, number_of_qubits)
        z1_loss = triplet_loss(z_out1)

        flattened = []
        for i, j, k in zip(im[0], im[1], im[2]):
            flattened.append(k)
            flattened.append(i)
        z_out2 = runcircuit(ind, flattened, number_of_qubits)
        z2_loss = triplet_loss(z_out2)
        siam_loss = (z1_loss - z2_loss)
        consistancy_loss = (np.abs((z_out1[0] - z_out2[2])) + np.abs((z_out1[1] - z_out2[3])))
        loss = 0.9 * siam_loss + 0.1 * consistancy_loss
        return loss

    def evaluate(self, ind, features, number_of_qubits):
        flattened = np.random.randint(2, size=2 ** self.number_of_qubits)
        flattened = np.reshape(flattened, (2 ** self.number_of_qubits,))
        qml.drawer.use_style('pennylane')
        fig, ax = qml.draw_mpl(runcircuit)(ind, flattened=flattened, number_of_qubits=number_of_qubits)
        plt.show()
        start_time = time.time()
        num_processes = multiprocessing.cpu_count()
        with multiprocessing.Pool(processes=num_processes) as pool:
            partial_process_feature = partial(self.process_feature, ind, number_of_qubits)
            loss_list = pool.map(partial_process_feature, features)
        loss = sum(loss_list)
        E = loss / len(features)
        fitness = 1 / (1 + E)
        print(f'loss: {E}')
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Total fitness calculation time: {elapsed_time} seconds\nAverage fitness: {fitness}")
        return [fitness]
