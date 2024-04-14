from os import path
import pickle
import copy
from math import pi 
import numpy as np
from projectq.ops import H, X, Y, Z, T, Tdagger, S, Sdagger, CNOT, CX, Rx, Ry, Rz, SqrtX, Swap, get_inverse
from qiskit import transpile
import qclib
from qclib.state_preparation.baa_schmidt import initialize


def save(pop, logbook, basedir, problem_name):
    """Save a population object and a logbook
    """
    with open(basedir+problem_name+".pop", 'wb') as file:
        pickle.dump(pop, file)
    with open(basedir+problem_name+".logbook", 'wb') as file:
        pickle.dump(logbook, file)
    print('Saved!')


def load(filepath):
    """Load a population object and corresponding logbook.

    Path contains the name of pop/logbook file
    WITHOUT .pop/.logbook extension
    """
    with open(filepath+".pop", 'rb') as file:
        pop = pickle.load(file)
    with open(filepath+".logbook", 'rb') as file:
        logbook = pickle.load(file)
    return pop, logbook


def load_state(number_of_qubits, index):
    state_name = str(number_of_qubits)+"QB_state"+str(index)
    with open(f'states/{number_of_qubits}_qubits/{state_name}', 'rb') as f:
        return pickle.load(f)


def load_qiskit_circuit(file):
    with open(path.join("qiskit_circuits", file), 'rb') as file:
        return pickle.load(file)


def get_permutation(circ):
    """Returns the permutation done by the transpile function by looking at the
    order of the measurement operators

    Requires measurement operators at the end of the circuit
    """
    # TODO remove function
    perm = [0, 1, 2, 3, 4]
    for operator, qubits, clbits in circ.data:
        if operator.name == 'measure':
            a = perm.index(clbits[0].index)
            temp = perm[qubits[0].index]
            perm[qubits[0].index] = clbits[0].index
            perm[a] = temp
    circ.remove_final_measurements()
    return perm


# TODO rename get_permutation_new_and_improved

def qubit(triple):
    return triple[1]


def clbit(triple):
    return triple[2]


def filter_measure_operator(operator_triple):
    return operator_triple[0].name == 'measure'


def map_operator_indices(operator_triple):
    return [operator_triple[0], qubit(operator_triple)[0].index, clbit(operator_triple)[0].index]


def sort_by_qubit_index(operator_indices):
    return sorted(operator_indices, key=qubit)


def get_permutation_new_and_improved(circ):
    """Returns the permutation done by the transpile function by looking at the
    order of the measurement operators

    Requires measurement operators at the end of the circuit
    """

    operator_indices = map(map_operator_indices, filter(
        filter_measure_operator, circ.data))

    perm = list(map(clbit, sort_by_qubit_index(operator_indices)))
    circ.remove_final_measurements()
    return perm


def lrsp_circs(state, toolbox, backend):
    # define list of fidelity loss values to try out
    losses = list(np.linspace(0.0, 1.0, 10))
    pop = toolbox.population(n=1)
    # find the exact circuit
    circuit = initialize(state, max_fidelity_loss=0.0,
                         strategy="brute_force", use_low_rank=True)
    circuit.measure_all()
    transpiled_circuit = transpile(
        circuit, backend, optimization_level=3)

    # create a list of circuits with increasing fidelity loss
    circuits = [transpiled_circuit]

    for loss in losses:
        # find approximate initialization circuit with fidelity loss
        circuit = initialize(state, max_fidelity_loss=loss,
                             strategy="brute_force", use_low_rank=True)
        circuit.measure_all()
        transpiled_circuit = transpile(
            circuit, backend, optimization_level=3)

        # if transpiled_circuit.depth() < circuits[-1].depth():
        circuits.append(transpiled_circuit)
    # ---------------------------------

    unaltered = []
    for circ in circuits:
        perm = get_permutation(circ)
        circ = qasm2ls(circ.qasm())
        pop[0].circuit = circ
        pop[0].permutation = perm
        pop[0].fitness.values = toolbox.evaluate(pop[0])
        unaltered.append(copy.deepcopy(pop[0]))
    return unaltered


def total_cnots(circ):
    cnots = 0
    for gate in circ:
        if gate[1] == CNOT:
            cnots += 1
    return cnots


def evaluate_cost(individual):
    individual.setCMW(error)
    cost = individual.evaluate_cost()
    return (error, cost)


def qasm2ls(qasmstr):
    """This helper function takes in qasmstr representing a quantum circuit
    and then transforms it into our list representation of the circuit.

    Returns: list
    """
    operations = qasmstr.split("\n")[3:]
    oplist = []
    for op in operations:
        opl = op.split(" ")
        if opl[0] in ["x", "h", "y", "z", "t", "tdg", "s", "sdg", "sx", "sxdg"]:
            t = int(opl[1][2:-2])
            if opl[0] == "x":
                oplist.append(("SFG", X, t))
            elif opl[0] == "h":
                oplist.append(("SFG", H, t))
            elif opl[0] == "y":
                oplist.append(("SFG", Y, t))
            elif opl[0] == "z":
                oplist.append(("SFG", Z, t))
            elif opl[0] == "t":
                oplist.append(("SFG", T, t))
            elif opl[0] == "tdg":
                oplist.append(("SFG", Tdagger, t))
            elif opl[0] == "s":
                oplist.append(("SFG", S, t))
            elif opl[0] == "sdg":
                oplist.append(("SFG", Sdagger, t))
            elif opl[0] == "sx":
                oplist.append(("SFG", SqrtX, t))
            elif opl[0] == "sxdg":
                oplist.append(("SFG", get_inverse(SqrtX), t))

        elif opl[0] in ["cx", "swap"]:
            co, ta = opl[1].split(",")
            co = int(co[2:-1])
            ta = int(ta[2:-2])
            if opl[0] == "cx":
                oplist.append(("TFG", CX, co, ta))
            else:
                oplist.append(("TFG", Swap, co, ta))

        elif opl[0][:2] in ["rx", "ry", "rz"]:
            t = int(opl[1][2:-2])
            p = eval(compile(opl[0][3:-1], "<string>", "eval"))  # TODO why!?

            if opl[0][:2] == "rx":
                oplist.append(("SG", Rx, t, p))
            elif opl[0][:2] == "ry":
                oplist.append(("SG", Ry, t, p))
            elif opl[0][:2] == "rz":
                oplist.append(("SG", Rz, t, p))

        elif opl[0] == "":
            continue
        else:
            print("error in qasm2ls:", opl)

    return oplist


projectq_of_string_map = {
    "h": H,
    "x": X,
    "y": Y,
    "z": Z,
    "t": T,
    "tdg": Tdagger,
    "s": S,
    "sdg": Sdagger,
    "sx": SqrtX,
    "sxdg": get_inverse(SqrtX),
    "cx": CNOT,  # TODO CNOT or CX?
    "swap": Swap,  # TODO Swap or SwapGate?
    "rx": Rx,
    "ry": Ry,
    "rz": Rz
}

string_of_projectq_items = [(v, k) for k, v in projectq_of_string_map.items()]


def projectq_of_string(string):
    return projectq_of_string_map[string]


def string_of_projectq(gate):
    maybe_string = [s for g, s in string_of_projectq_items if g == gate]
    return maybe_string[0]

