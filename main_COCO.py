import concurrent.futures
import pandas as pd
import csv
import data_processing
import data_load_ladscape
from functools import partial
from multiprocessing import Pool
from qiskit import QuantumCircuit
from qiskit.visualization import circuit_drawer
import os
from os import listdir
import genetic_algorithm
# from datetime import datetime
from os.path import isfile, join
import argparse
import time
from constants import NUMBER_OF_GENERATIONS, NUMBER_OF_QUBITS, POPULATION_SIZE, NUM_TRIPLETS
from toolbox import initialize_toolbox  # also initializes creator


def main():
    dataset = 'DISC21'
    device = 'window'  # Assume the device is either windows or linux
    if device == 'window':
        base_path = fr"D:\pycharm_projects\SLIQ-PENNYLANE_mnist/"
    elif device == 'linux':
        base_path = "/home/ubuntu/ylh/"
        # base_path = "/home/ubuntu/ylh/SLIQ-PENNYLANE_mnist/"
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    else:
        raise ValueError("Unknown device type")

    start = time.perf_counter()

    """Runs the genetic algorithm based on the global constants
    """
    # Initialize parser
    parser = argparse.ArgumentParser()

    # Adding optional argument
    parser.add_argument("-p", "--POPSIZE", help="Size of the population")
    parser.add_argument("-g", "--NGEN", help="The number of generations")
    parser.add_argument("-q", "--NQUBIT", help="The number of qubits")
    parser.add_argument("-i", "--INDEX", help="Index of desired state")
    # FIXME -id is illegal (it means -i -d)
    parser.add_argument("-id", "--ID", help="ID of the saved file")

    # Read arguments from command line
    args = parser.parse_args()
    # population_size = int(args.POPSIZE) if args.POPSIZE else POPULATION_SIZE
    # number_of_generations = int(args.NGEN) if args.NGEN else NUMBER_OF_GENERATIONS
    number_of_qubits = int(args.NQUBIT) if args.NQUBIT else NUMBER_OF_QUBITS
    triplets, image_indices = data_load_ladscape.generate_landscape_triplets(base_path, dataset, num_triplets=5000, testing=False)
    toolbox = initialize_toolbox(number_of_qubits)
    EVO = genetic_algorithm.Evolution(triplets)
    pop, fitness_ranks = EVO.evolution(dataset, toolbox)  # Evolved population
    final_individual = pop
    print('last population', final_individual)
    print('fitness of last population', fitness_ranks)
    count = 1

    for ind in final_individual:
        qasm_string = ind.circuit.qasm()
        file_name = f'{base_path}/result/{dataset}/qasm/{dataset}_best_Candidate_{count}.qasm'
        with open(file_name, "w") as file:
            file.write(qasm_string)
        qc_ind = QuantumCircuit.from_qasm_file(file_name)
        circuit_drawer(qc_ind, output='mpl',
                       filename=f'{base_path}/result/{dataset}/png/{dataset}_best_Candidate_{count}_{ind.fitness.values[0]}.png')
        count += 1
    runtime = round(time.perf_counter() - start, 2)
    print("runtime train", runtime)

    triplets, image_indices = data_load_ladscape.generate_landscape_triplets(base_path, dataset, num_triplets=100, testing=True)
    for j in range(1, 21):
        relatative = []
        num_cnot = []
        for i in range(1, 21):
            individual_path = f'{base_path}result/{dataset}/qasm/{dataset}_next_generation_{j}_{i}.qasm'
            if not os.path.exists(individual_path):
                break
            data_load_ladscape.find_closest(base_path, dataset, number_of_qubits, i, j, triplets, image_indices)
            relatative.append(data_load_ladscape.Spearman(base_path, dataset, i, j))
            num_of_cnot = data_load_ladscape.cnot(base_path, dataset, i, j)
            num_cnot.append(num_of_cnot)
            print(f'Individual {j}_{i}, cnot count: {num_of_cnot}')
        # Save to CSV file
        with open(f'{base_path}/result/{dataset}/csv/combined_relatative_num_cnot_j{j}.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['relatative', 'num_cnot'])
            writer.writerows(zip(relatative, num_cnot))

    combined_df = pd.DataFrame()
    for j in range(1, 21):
        file_name = fr'{base_path}/result/{dataset}/csv/combined_relatative_num_cnot_j{j}.csv'
        df = pd.read_csv(file_name)

        # Extract fitness data from another file
        file_name2 = fr'{base_path}/result/{dataset}/csv/{dataset}_fitness_values_ranks_{j}.csv'
        fitness_df = pd.read_csv(file_name2)

        # Place fitness data in front of 'relatative' column
        new_df = pd.DataFrame()
        new_df[f'generation{j}'] = list(range(1, 21))
        new_df[f'fitness{j}'] = fitness_df['Fitness Value']
        new_df[f'relatative'] = df['relatative']
        new_df[f'num_cnot'] = df['num_cnot']
        new_df[''] = ''  # Insert blank column

        combined_df = pd.concat([combined_df, new_df], axis=1)

    # Save processed data to a new file
    combined_df.to_excel(f'{base_path}/result/{dataset}/csv/combined_Fitness_relatative_cnot_combined.xlsx',
                       index=False)


    combined_df = pd.DataFrame()
    for j in range(1, 21):
        # Extract fitness data from another file
        file_name2 = f'{base_path}/result/{dataset}/csv/{dataset}_fitness_values_ranks_{j}.csv'
        fitness_df = pd.read_csv(file_name2)

        # Place fitness data in front of 'relatative' column
        new_df = pd.DataFrame()
        new_df[f'fitness{j}'] = fitness_df['Fitness Value']

        combined_df = pd.concat([combined_df, new_df], axis=1)

    # Save processed data to a new file
    combined_df.to_excel(f'{base_path}/result/{dataset}/csv/combined_Fitness.xlsx',
                       index=False)

    runtime5 = round(time.perf_counter() - start, 2)
    print("runtime test", runtime5)


if __name__ == '__main__':
    main()
