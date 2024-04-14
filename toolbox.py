from deap import creator, base, tools
from functools import partial
from genetic_algorithm import triplet_loss, runcircuit, Evolution, mutate_ind
from individual import Individual

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", Individual, fitness=creator.FitnessMax)


def initialize_toolbox(number_of_qubits):
    toolbox = base.Toolbox()

    toolbox.register("individual", Evolution.new_individual)
    toolbox.register("population", Evolution.new_pop)

    toolbox.register("mate", Evolution.mate, toolbox=toolbox)
    toolbox.register("mutate_individuals", Evolution.mutate_individuals)
    toolbox.register("mutate_ind", mutate_ind)
    toolbox.register("select", tools.selNSGA2)
    toolbox.register("select_and_evolve", Evolution.select_and_evolve)

    toolbox.register("evaluate", Evolution.evaluate)

    toolbox.register("evolution", Evolution.evolution)

    toolbox.register("process_feature", Evolution.process_feature)
    toolbox.register("triplet_loss", triplet_loss)
    toolbox.register("runcircuit", runcircuit, number_of_qubits=number_of_qubits)

    return toolbox

