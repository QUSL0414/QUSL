from deap import creator, base, tools
import random
import numpy.random
import numpy as np
from projectq.ops import H, X, Y, Z, T, Tdagger, S, Sdagger, CNOT, CX, Rx, Ry, Rz, SqrtX, Measure, All, get_inverse, \
    Swap, SwapGate
from math import pi
from qiskit import QuantumCircuit


class Individual:
    def __init__(self, circuit, number_of_qubits, connectivity="ALL", EMC=5.0, ESL=5.0):
        self.number_of_qubits = number_of_qubits
        self.allowed_gates = self.allowed_gates = [Rx, Ry, Rz, CNOT]  # 共10种量子门
        self.connectivity = connectivity
        self.permutation = random.sample(
            range(number_of_qubits), number_of_qubits)

        self.EMC = EMC
        self.ESL = ESL
        self.circuit = circuit
        self.CMW = 0.3
        self.optimizedx = False
        self.num_clbits = None

    def generate_random_circuit(self, initialize=True):

        if initialize:
            p = 1 / 30
        else:
            p = 1 / self.ESL
        cir_length = numpy.random.geometric(p)
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
                produced_circuit.append(("TFG", gate, control, target))

            elif gate in [H, X, Y, Z, ]:
                target = random.choice(range(self.number_of_qubits))
                produced_circuit.append(("SFG", gate, target))
            elif gate in [Rx, Ry, Rz]:
                target = random.choice(range(self.number_of_qubits))
                significant_figure = 2
                parameter = round(pi * random.uniform(0, 2),
                                  significant_figure)
                produced_circuit.append(("SG", gate, target, parameter))
                quantum_circuit.rx(parameter, target)
            else:
                print("WHAT ARE YOU DOING HERE!!")
                print("GATE IS:", gate)

        return produced_circuit

    def get_permutation_matrix(self):
        """
        Args:
            perm: a list representing where each qubit is mapped to.
                perm[i] represents the physical qubit and i represents the virtual qubit.
                So [1,0,2] will be interpreted as (virtual->physical) 0->1, 1->0, 2->2
        Returns:
            2^N x 2^N numpy matrix representing the action of the permutation where
            N is the number of qubits.
        """
        Nexp = 2 ** self.number_of_qubits
        M = np.zeros((Nexp, Nexp))
        for frm in range(Nexp):
            to_bin = list(bin(0)[2:].zfill(self.number_of_qubits))
            frm_bin = list(bin(frm)[2:].zfill(self.number_of_qubits))
            to_bin.reverse()
            frm_bin.reverse()
            for p in range(self.number_of_qubits):
                to_bin[p] = frm_bin[self.permutation.index(p)]
            to_bin.reverse()
            to = int("".join(to_bin), 2)
            M[to][frm] = 1.0
        return M

    def __str__(self):
        output = "number_of_qubits: " + str(self.number_of_qubits)
        output += "\nConnectivity = " + str(self.connectivity)
        output += "\nQubit Mapping = " + str(self.permutation)
        output += "\nallowedGates: ["
        for i in range(len(self.allowed_gates)):
            if self.allowed_gates[i] == Rx:
                output += "Rx, "
            elif self.allowed_gates[i] == Ry:
                output += "Ry, "
            elif self.allowed_gates[i] == Rz:
                output += "Rz, "
            elif self.allowed_gates[i] in [SwapGate, Swap]:
                output += "Swap, "
            elif self.allowed_gates[i] in [SqrtX]:
                output += "SqrtX, "
            elif self.allowed_gates[i] in [CNOT, CX]:
                output += "CX, "
            else:
                output += str(self.allowed_gates[i]) + ", "
        output = output[:-2]
        output += "]\nEMC: " + str(self.EMC) + ", ESL: " + str(self.ESL) + "\n"
        output += self.print_circuit()
        output += "\ncircuitLength: " + str(len(self.circuit))
        return output

    def print_circuit(self):
        output = "Qubit Mapping:" + str(self.permutation) + "\n"
        output += "Circuit: ["
        for operator in self.circuit:
            if operator[0] == "SFG":
                output += (
                        "("
                        + str(operator[1])
                        + ","
                        + str(operator[2])
                        + "), "
                )
            elif operator[0] == "TFG":
                output += (
                        "("
                        + str(operator[1])
                        + ","
                        + str(operator[2])
                        + ","
                        + str(operator[3])
                        + "), "
                )
            elif operator[0] == "SG":
                output += (
                        "("
                        + str(operator[1](round(operator[3], 3)))
                        + ","
                        + str(operator[2])
                        + "), "
                )
        output = output[:-2]
        output += "]"
        return output

    def clean(self):
        """
        Optimizes self.circuit by removing redundant gates
        """
        finished = False
        while not finished:
            finished = True
            i = 0
            while i < len(self.circuit) - 1:
                gate = self.circuit[i]

                if gate[1] == SqrtX:
                    j = i + 1
                    while j < len(self.circuit):
                        if self.circuit[j][1] == gate[1] and self.circuit[j][2] == gate[2]:
                            self.circuit.pop(j)
                            self.circuit[i] = ("SFG", X, gate[2])
                            finished = False
                            break
                        elif self.circuit[j][1] == get_inverse(SqrtX) and self.circuit[j][2] == gate[2]:
                            self.circuit.pop(j)
                            self.circuit.pop(i)
                            finished = False
                            break
                        elif self.circuit[j][0] == "TFG":
                            if self.circuit[j][2] == gate[2] or self.circuit[j][3] == gate[2]:
                                break
                        elif self.circuit[j][2] == gate[2]:
                            break
                        j += 1
                # elif gate[1] == Rz:
                elif gate[1] == Rz or gate[1] == Rx or gate[1] == Ry:
                    j = i + 1
                    while j < len(self.circuit):
                        if self.circuit[j][1] == gate[1] and self.circuit[j][2] == gate[2]:
                            parameter = (self.circuit[j][3] + gate[3]) % (pi * 2)
                            self.circuit.pop(j)
                            self.circuit[i] = ("SG", Rz, gate[2], parameter)
                            finished = False
                            break
                        elif self.circuit[j][0] == "TFG":
                            if self.circuit[j][2] == gate[2] or self.circuit[j][3] == gate[2]:
                                break
                        elif self.circuit[j][2] == gate[2]:
                            break
                        j += 1
                elif gate[1] == X:
                    j = i + 1
                    while j < len(self.circuit):
                        if self.circuit[j][1] == gate[1] and self.circuit[j][2] == gate[2]:
                            self.circuit.pop(j)
                            self.circuit.pop(i)
                            finished = False
                            break
                        elif self.circuit[j][0] == "TFG":
                            if self.circuit[j][2] == gate[2] or self.circuit[j][3] == gate[2]:
                                break
                        elif self.circuit[j][2] == gate[2]:
                            break
                        j += 1
                elif gate[1] == CX:
                    j = i + 1
                    while j < len(self.circuit):
                        if self.circuit[j][1] == gate[1] and self.circuit[j][2] == gate[2] and self.circuit[j][3] == \
                                gate[3]:
                            self.circuit.pop(j)
                            self.circuit.pop(i)
                            finished = False
                            break
                        elif self.circuit[j][2] == gate[2]:
                            break
                        elif self.circuit[j][2] == gate[3]:
                            break
                        elif self.circuit[j][0] == "TFG":
                            if self.circuit[j][3] == gate[2] or self.circuit[j][3] == gate[3]:
                                break
                        j += 1
                elif gate[1] == Swap or gate[1] == CNOT:
                    j = i + 1
                    while j < len(self.circuit):
                        if self.circuit[j][1] == gate[1] and self.circuit[j][2] == gate[2] and self.circuit[j][3] == \
                                gate[3]:
                            self.circuit.pop(j)
                            self.circuit.pop(i)
                            finished = False
                            break
                        if self.circuit[j][1] == gate[1] and self.circuit[j][2] == gate[3] and self.circuit[j][3] == \
                                gate[2]:
                            self.circuit.pop(j)
                            self.circuit.pop(i)
                            finished = False
                            break
                        elif self.circuit[j][2] == gate[2]:
                            break
                        elif self.circuit[j][2] == gate[3]:
                            break
                        elif self.circuit[j][0] == "TFG":
                            if self.circuit[j][3] == gate[2] or self.circuit[j][3] == gate[3]:
                                break
                        j += 1

                i += 1

    def permutation_mutation(self):
        self.permutation = random.sample(
            range(self.number_of_qubits), self.number_of_qubits)

    def discrete_uniform_mutation(self):
        """
        Args:
          None
        Returns:
          None -> should I return sth ? maybe self?
        """
        circuit_length = len(self.circuit)
        if circuit_length == 0:
            mutation_prob = 0
        else:
            mutation_prob = self.EMC / circuit_length
        # I don't know if we really need this part
        if mutation_prob >= 1.0:
            mutation_prob = 0.5

        # We will loop over all the gates
        for i in range(circuit_length):
            if random.random() < mutation_prob:
                self.discrete_mutation(i)

    def sequence_insertion(self):
        circuit_to_insert = self.generate_random_circuit(initialize=False)
        old_circuit_length = len(self.circuit)
        if old_circuit_length == 0:
            insertion_index = 0
        else:
            insertion_index = random.choice(range(old_circuit_length))
        self.circuit[insertion_index:] = circuit_to_insert + \
                                         self.circuit[insertion_index:]
        return self.circuit

    def sequence_and_inverse_insertion(self):
        """
        This function generates a random circuit with circuit length given by choosing
        a value from a geometric distribution with mean value ESL, it is inserted to a
        random point in self.circuit and its inverse is inserted to another point.
        """
        circuit_to_insert = self.generate_random_circuit(initialize=False)
        # MAYBE CONNECTIVITY IS NOT REFLECTIVE ?
        inverse_circuit = get_inverse_circuit(circuit_to_insert)
        old_circuit_length = len(self.circuit)
        if old_circuit_length >= 2:
            index1, index2 = random.sample(range(old_circuit_length), 2)
            if index1 > index2:
                index2, index1 = index1, index2
        else:
            index1, index2 = 0, 1
        new_circuit = (
                self.circuit[:index1]
                + circuit_to_insert
                + self.circuit[index1:index2]
                + inverse_circuit
                + self.circuit[index2:]
        )
        self.circuit = new_circuit

    def discrete_mutation(self, index):
        """
        This function applies a discrete mutation to the circuit element at index.
        Discrete mutation means that the control and/or target qubits are randomly changed.
        """
        if len(self.circuit) == 0:
            return
        while index >= len(self.circuit):
            index -= 1
        if self.circuit[index][0] == "SFG":
            # This means we have a single qubit fixed gate
            new_target = random.choice(range(self.number_of_qubits))
            self.circuit[index] = ("SFG", self.circuit[index][1], new_target)
        elif self.circuit[index][0] == "TFG":
            # This means we have two qubit fixed gate
            if self.connectivity == "ALL":
                new_control, new_target = random.sample(
                    range(self.number_of_qubits), 2)
            else:
                new_control, new_target = random.choice(self.connectivity)
            self.circuit[index] = (
                "TFG", self.circuit[index][1], new_control, new_target)
        elif self.circuit[index][0] == "SG":
            # This means we have a single rotation gate
            new_target = random.choice(range(self.number_of_qubits))
            self.circuit[index] = (
                "SG",
                self.circuit[index][1],
                new_target,
                self.circuit[index][3],
            )
        else:
            print("WRONG BRANCH IN discrete_mutation")

    def continuous_mutation(self, index):
        """
        This function applies a continuous mutation to the circuit element at index.
        Continuous mutation means that if the gate has a parameter, its parameter its
        changed randomly, if not a discrete_mutation is applied.
        """
        if len(self.circuit) == 0:
            return
        while index >= len(self.circuit):
            index -= 1

        if self.circuit[index][0] == "SG":
            # This means we have a single rotation gate
            newParameter = float(
                self.circuit[index][-1]) + numpy.random.normal(scale=self.CMW)
            self.circuit[index] = (
                "SG", self.circuit[index][1], self.circuit[index][2], newParameter)
        elif self.circuit[index][0] == "SFG":
            # This means we have a single qubit/two qubit fixed gate and we need to
            # apply a discrete_mutation.
            new_target = random.choice(range(self.number_of_qubits))
            self.circuit[index] = ("SFG", self.circuit[index][1], new_target)
        elif self.circuit[index][0] == "TFG":
            # This means we have two qubit fixed gate
            if self.connectivity == "ALL":
                new_control, new_target = random.sample(
                    range(self.number_of_qubits), 2)
            else:
                new_control, new_target = random.choice(self.connectivity)
            self.circuit[index] = (
                "TFG", self.circuit[index][1], new_control, new_target)
        else:
            print("WRONG BRANCH IN continuous_mutation")

    def parameter_mutation(self):
        if len(self.circuit) == 0:
            return

        mutation_prob = self.EMC / len(self.circuit)
        for index in range(len(self.circuit)):
            if random.random() < mutation_prob:
                if self.circuit[index][0] == "SG":
                    # This means we have a single rotation gate
                    newParameter = float(
                        self.circuit[index][-1]) + numpy.random.normal(scale=self.CMW)
                    newParameter = newParameter % (2 * pi)
                    self.circuit[index] = (
                        "SG", self.circuit[index][1], self.circuit[index][2], newParameter)

    def continuous_uniform_mutation(self):
        """
        This function iterates over all the gates defined in the circuit and
        randomly changes the parameter if possible, if not target and/or control qubits
        with probability EMC / circuit_length.
        Args:
          None
        Returns:
          None -> should I return sth ? maybe self?
        """
        circuit_length = len(self.circuit)
        if circuit_length == 0:
            mutation_prob = 0
        else:
            mutation_prob = self.EMC / circuit_length
        # I don't know if we really need this part
        if mutation_prob >= 1.0:
            mutation_prob = 0.5

        # We will loop over all the gates
        for i in range(circuit_length):
            if random.random() < mutation_prob:
                self.continuous_mutation(i)

    def insert_mutate_invert(self):
        """
        This function performs a discrete mutation on a single gate, then places a
        randomly selected gate immediately before it and its inverse immediately
        after it.
        """
        # index to apply discrete mutation
        if len(self.circuit) == 0:
            index = 0
        else:
            index = random.choice(range(len(self.circuit)))

        # Discrete Mutation
        self.discrete_mutation(index)

        # Generate the circuit to insert and its inverse
        circuit_to_insert = self.generate_random_circuit(initialize=False)
        while len(circuit_to_insert) == 0:
            circuit_to_insert = self.generate_random_circuit(initialize=False)
        circuit_to_insert = [circuit_to_insert[0]]
        inverse_circuit = get_inverse_circuit(circuit_to_insert)
        if index >= len(self.circuit):
            # This probably happens only when index = 0 and length of the circuit = 0
            if index == 0:
                new_circuit = circuit_to_insert + inverse_circuit
            else:
                print("\n\nIT SHOULD NEVER ENTER HEREE!!!\n\n")
        else:
            new_circuit = (
                    self.circuit[:index]
                    + circuit_to_insert
                    + [self.circuit[index]]
                    + inverse_circuit
                    + self.circuit[(index + 1):]
            )
        self.circuit = new_circuit

    def swap_qubits(self):
        """
        This function swaps two randomly selected qubits.
        """
        qubit1, qubit2 = random.sample(range(self.number_of_qubits), 2)

        for operator in self.circuit:
            if operator[0] == "SFG":
                if operator[2] == qubit1:
                    operator = operator[0:2] + (qubit2,)
                elif operator[2] == qubit2:
                    operator = operator[0:2] + (qubit1,)

            elif operator[0] == "TFG":
                if operator[2] == qubit1 and operator[3] == qubit2:
                    operator = operator[0:2] + (qubit2, qubit1)

                elif operator[2] == qubit2 and operator[3] == qubit1:
                    operator = operator[0:2] + (qubit1, qubit2)

                elif operator[2] == qubit1:
                    operator = (
                            operator[0:2] + (qubit2,) + operator[3:]
                    )

                elif operator[2] == qubit2:
                    operator = (
                            operator[0:2] + (qubit1,) + operator[3:]
                    )

                elif operator[3] == qubit1:
                    operator = operator[0:3] + (qubit2,)

                elif operator[3] == qubit2:
                    operator = operator[0:3] + (qubit1,)

            elif operator[0] == "SG":
                if operator[2] == qubit1:
                    operator = (
                            operator[0:2] +
                            (qubit2,) + (operator[3],)
                    )
                elif operator[2] == qubit2:
                    operator = (
                            operator[0:2] +
                            (qubit1,) + (operator[3],)
                    )

    def sequence_deletion(self):
        """
        This function deletes a randomly selected interval of the circuit.
        """
        if len(self.circuit) < 2:
            return

        circuit_length = len(self.circuit)
        index = random.choice(range(circuit_length))
        # If this is the case, we'll simply remove the last element
        if index == (circuit_length - 1):
            self.circuit = self.circuit[:-1]
        else:
            sequence_length = numpy.random.geometric(p=(1 / self.ESL))
            if (index + sequence_length) >= circuit_length:
                self.circuit = self.circuit[: (-circuit_length + index)]
            else:
                self.circuit = (
                        self.circuit[:index] +
                        self.circuit[(index + sequence_length):]
                )

    def sequence_replacement(self):
        """
        This function first applies sequence_deletion, then applies a sequence_insertion.
        """
        self.sequence_deletion()
        self.sequence_insertion()

    def sequence_swap(self):
        """
        This function randomly chooses two parts of the circuit and swaps them.
        """
        if len(self.circuit) < 4:
            return

        indices = random.sample(range(len(self.circuit)), 4)
        indices.sort()
        i1, i2, i3, i4 = indices[0], indices[1], indices[2], indices[3]

        self.circuit = (
                self.circuit[0:i1]
                + self.circuit[i3:i4]
                + self.circuit[i2:i3]
                + self.circuit[i1:i2]
                + self.circuit[i4:]
        )

    def sequence_scramble(self):
        circuit_length = len(self.circuit)
        if circuit_length < 2:
            index1 = 0
        else:
            index1 = random.choice(range(circuit_length - 1))

        sequence_length = numpy.random.geometric(p=(1 / self.ESL))
        if (index1 + sequence_length) >= circuit_length:
            index2 = circuit_length - 1
        else:
            index2 = index1 + sequence_length

        toShuffle = self.circuit[index1:index2]
        random.shuffle(toShuffle)

        self.circuit = self.circuit[:index1] + \
                       toShuffle + self.circuit[index2:]

    def move_gate(self):
        """
        This function randomly moves a gate from one point to another point.
        """
        circuit_length = len(self.circuit)
        if circuit_length < 2:
            return
        old_index, new_index = random.sample(range(circuit_length), 2)

        temp = self.circuit.pop(old_index)
        self.circuit.insert(new_index, temp)

    def cross_over(self, parent2, toolbox):
        """This function gets two parent solutions, creates an empty child, randomly
        picks the number of gates to be selected from each parent and selects that
        number of gates from the first parent, and discards that many from the
        second parent. Repeats this until parent solutions are exhausted.
        """
        self_circuit = self.circuit[:]
        parent2_circuit = parent2.circuit[:]
        p1 = p2 = 1.0

        if len(self_circuit) != 0:
            p1 = self.EMC / len(self.circuit)
        if (p1 <= 0) or (p1 > 1):
            p1 = 1.0

        if len(parent2_circuit) != 0:
            p2 = parent2.EMC / len(parent2.circuit)
        if (p2 <= 0) or (p2 > 1):
            p2 = 1.0

        # child = Individual()
        # circuit = None
        circuit = []
        child = creator.Individual(circuit, self.number_of_qubits)
        child.circuit = []
        turn = 1
        while len(self_circuit) or len(parent2_circuit):
            if turn == 1:
                number_of_gates_to_select = numpy.random.geometric(p1)
                child.circuit += self_circuit[:number_of_gates_to_select]
                turn = 2
            else:
                number_of_gates_to_select = numpy.random.geometric(p2)
                child.circuit += parent2_circuit[:number_of_gates_to_select]
                turn = 1
            self_circuit = self_circuit[number_of_gates_to_select:]
            parent2_circuit = parent2_circuit[number_of_gates_to_select:]
        return child  # individual class


def print_circuit(circuit):
    output = "Circuit: ["
    for i in range(len(circuit)):
        if circuit[i][0] == "SFG":
            output += "(" + str(circuit[i][1]) + \
                      "," + str(circuit[i][2]) + "), "
        elif circuit[i][0] == "TFG":
            output += (
                    "("
                    + str(circuit[i][1])
                    + ","
                    + str(circuit[i][2])
                    + ","
                    + str(circuit[i][3])
                    + "), "
            )
        elif circuit[i][0] == "SG":
            output += (
                    "("
                    + str(circuit[i][1](round(circuit[i][3], 3)))
                    + ","
                    + str(circuit[i][2])
                    + "), "
            )
    output = output[:-2]
    output += "]"
    return output


def get_inverse_circuit(circuit):
    """
    This function takes a circuit and returns a circuit which is the inverse circuit.
    """
    if len(circuit) == 0:
        return []

    reversed_circuit = circuit[::-1]
    for gate in reversed_circuit:
        if gate[1] in [H, X, Y, Z, CX, Swap, SwapGate]:
            continue
        elif gate[1] == S:
            gate = ("SFG", Sdagger, gate[2])
        elif gate[1] == Sdagger:
            gate = ("SFG", S, gate[2])
        elif gate[1] == T:
            gate = ("SFG", Tdagger, gate[2])
        elif gate[1] == Tdagger:
            gate = ("SFG", T, gate[2])
        elif gate[1] in [Rx, Ry, Rz]:
            gate = (
                "SG",
                gate[1],
                gate[2],
                round(2 * pi - gate[3], 3),
            )
        elif gate[1] in [SqrtX]:
            gate = ("SFG", get_inverse(
                SqrtX), gate[2])
        elif gate[1] in [get_inverse(SqrtX)]:
            gate = ("SFG", SqrtX, gate[2])
        else:
            print("\nWRONG BRANCH IN get_inverse_circuit\n")

    return reversed_circuit
