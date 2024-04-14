import concurrent.futures
import pandas as pd
import csv
import data_load_ladscape
import os
from qiskit import QuantumCircuit
from qiskit.visualization import circuit_drawer
from os import listdir
import genetic_algorithm
from datetime import datetime
from os.path import isfile, join
import argparse
import time
from constants import NUMBER_OF_GENERATIONS, NUMBER_OF_QUBITS, POPULATION_SIZE, NUM_TRIPLETS
from toolbox import initialize_toolbox


def main():
    dataset = 'landscape'
    device = 'linux'
    if device == 'window':
        base_path = "D:\pycharm_projects\SLIQ-PENNYLANE_mnist/"
    elif device == 'linux':
        base_path = "/home/ubuntu/ylh/"
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    else:
        raise ValueError("Unknown device type")

    start = time.perf_counter()
    parser = argparse.ArgumentParser()

    parser.add_argument("-p", "--POPSIZE", help="Size of the population")
    parser.add_argument("-g", "--NGEN", help="The number of generations")
    parser.add_argument("-q", "--NQUBIT", help="The number of qubits")
    parser.add_argument("-i", "--INDEX", help="Index of desired state")
    parser.add_argument("-id", "--ID", help="ID of the saved file")

    args = parser.parse_args()
    number_of_qubits = int(args.NQUBIT) if args.NQUBIT else NUMBER_OF_QUBITS
    triplets, image_indices = data_load_ladscape.generate_landscape_triplets(base_path, dataset, num_triplets=5000, testing=False)

    runtime1 = round(time.perf_counter() - start, 2)
    print("runtime first", runtime1)

    toolbox = initialize_toolbox(number_of_qubits)
    EVO = genetic_algorithm.Evolution(triplets)
    pop, fitness_ranks = EVO.evolution(dataset, toolbox)
    runtime2 = round(time.perf_counter() - runtime1, 2)
    print("runtime evolution", runtime2)
    final_individual = pop
    print('last population', final_individual)
    print('fitness of last population', fitness_ranks)
    count = 1

    for ind in final_individual:
        qasm_string = ind.circuit.qasm()
        file_name = f'{base_path}/result/landscape/qasm/land_best_Candidate_{count}.qasm'
        with open(file_name, "w") as file:
            file.write(qasm_string)
        qc_ind = QuantumCircuit.from_qasm_file(file_name)
        circuit_drawer(qc_ind, output='mpl',
                       filename=f'{base_path}/result/landscape/png/land_best_Candidate_{count}_{ind.fitness.values[0]}.png')
        count += 1
    runtime4 = round(time.perf_counter() - runtime1, 2)
    print("runtime train", runtime4)

    for j in range(16):
        relatative = []
        num_cnot = []
        for i in range(20):
            data_load_ladscape.find_closest(number_of_qubits, i+1, j+1)
            relatative.append(data_load_ladscape.Spearman(i+1, j+1))
            num_of_cnot = data_load_ladscape.cnot(i+1, j+1)
            num_cnot.append(num_of_cnot)
            print(f'Individual {j+1}_{i+1}, cnot count: {num_of_cnot}')
        # Save to CSV file
        with open(f'{base_path}/result/landscape/csv/combined_relatative_num_cnot_j{j + 1}.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['relatative', 'num_cnot'])
            writer.writerows(zip(relatative, num_cnot))

    combined_df = pd.DataFrame()
    for j in range(1, 21):
        file_name = f'{base_path}/result/landscape/csv/combined_relatative_num_cnot_j{j}.csv'
        df = pd.read_csv(file_name)

        # Extract fitness data from another file
        file_name2 = f'{base_path}/result/landscape/csv/landscape_fitness_values_ranks_{j}.csv'
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
    combined_df.to_excel(f'{base_path}/result/landscape/csv/combined_Fitness_relatative_cnot_combined.xlsx',
                       index=False)


    combined_df = pd.DataFrame()
    for j in range(1, 21):
        # Extract fitness data from another file
        file_name2 = f'{base_path}/result/landscape/csv/landscape_fitness_values_ranks_{j}.csv'
        fitness_df = pd.read_csv(file_name2)

        # Place fitness data in front of 'relatative' column
        new_df = pd.DataFrame()
        new_df[f'fitness{j}'] = fitness_df['Fitness Value']

        combined_df = pd.concat([combined_df, new_df], axis=1)

    # Save processed data to a new file
    combined_df.to_excel(f'{base_path}/result/landscape/csv/combined_Fitness.xlsx',
                       index=False)

    runtime5 = round(time.perf_counter() - start, 2)
    print("runtime test", runtime5)


if __name__ == '__main__':
    main()
