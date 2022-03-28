#! /usr/bin/python3

from pennylane import numpy as np
import pennylane as qml


# Number of wires in the circuit
num_wires = 2

dev = qml.device('default.qubit', wires=num_wires)
dev2 = qml.device('default.qubit', wires=['anc'] + list(range(num_wires)))


def main():
    """
    main method to run program
    """

    # construct the Hamiltonian we wish to use
    obs = [qml.PauliZ(0) @ qml.PauliX(1),
           qml.PauliX(0) @ qml.PauliZ(1),
           qml.PauliZ(0) @ qml.PauliZ(1)]
    coeff = [1, 1, 3]
    H = qml.Hamiltonian(coeff, obs)

    print('-----------------------------------------')
    print('The Hamiltonian we are approximating the ground state of')
    print('H =', H)

    print('-----------------------------------------')

    # initial values of the trial state parameters theta
    theta = np.array([0, np.pi, 0, 0, np.pi/2, 0, 0, np.pi/2],
                     requires_grad=True)

    # List of operations used in the Ansatz
    ansatz_ops = [
            qml.RY(theta[0], wires=0),
            qml.RY(theta[1], wires=1),
            qml.RZ(theta[2], wires=0),
            qml.RZ(theta[3], wires=1),
            qml.CNOT(wires=[0, 1]),
            qml.RY(theta[4], wires=0),
            qml.RY(theta[5], wires=1),
            qml.RZ(theta[6], wires=0),
            qml.RZ(theta[7], wires=1)
            ]

    # Ansatz tape instead of qnode
    with qml.tape.QuantumTape() as ansatz_tape:
        for op in ansatz_ops:
            op.queue()

    # Update the values of theta
    theta = np.array([np.pi/2, np.pi, np.pi/3, 0,
                      np.pi/2, 0, np.pi/4, np.pi/2], requires_grad=True)
    ansatz_tape.set_parameters(theta)

    print('The ansatz being used to prepare the trial state |phi(theta)>')
    print(ansatz_tape.draw())

    print('-----------------------------------------')

    print('Here we construct the matrix ' +
          'A_ij = Re( (d <phi(theta)|/d theta_i) (d |phi(theta)>/d theta_j) )')
    A = create_A(ansatz_tape, theta)
    print('A =', A)

    print('-----------------------------------------')

    print('Here we construct the vector ' +
          'C_i = -Re( (d <phi(theta)|/d theta_i) (H |phi(theta)>) )')
    C = create_C(ansatz_tape, theta, H)
    print('C =', C)

    print(qml.draw(Ci)(np.pi/2, theta, ansatz_tape, H.ops[1], 5))

    # TODO properly calculate the phase angle

    # TODO add ODE solver to solve A theta_dot = C

    # TODO add way to update theta

    # TODO graph the results
# end of main()


@qml.qnode(dev)
def ansatz(ansatz_ops, thetas):
    """
    Implement the ansatz we are using to create |phi(theta)>

    Parameters:
        ansatz_ops - the operations for the ansatz
        thetas - the trainable parameters

    Return:
        ansatz_tape - the ansatz in tape form
    """

    with qml.tape.QuantumTape() as ansatz_tape:
        for op in ansatz_ops:
            op.queue()

    return ansatz_tape
# end of ansatz()


@qml.qnode(dev2)
def Aij(alpha, thetas, ansatz_tape, i, j):
    """
      Circuit to implement the calculation of Aij

    Parameters:
        alpha - needed phase angle
        thetas - the trainable angles used to create |phi(theta)>
        ansatz_tape - the ansatz we are using to create the trial state
        i - the index of matrix A we are calculating
        j - the index of matrix A we are calculating

    Return:
        A_i - the expectation value
    """

    theta_index = -1  # index used for parameter checking
    I_index = min(i, j)  # the smaller index
    J_index = max(i, j)  # the larger index

    # initial state
    qml.Hadamard(wires='anc')
    qml.RZ(alpha, wires='anc')
    qml.Barrier(wires=['anc'] + list(range(num_wires)))

    # the circuit

    for op in ansatz_tape.operations:
        # if the current operation is non-parameterised, just queue it
        if op.num_params == 0:
            op.queue()

        # if the operation is parameterised
        else:
            theta_index += 1

            # if the operation is the first one we want the derivative of
            if theta_index == I_index:
                qml.Barrier(wires=['anc'] + list(range(num_wires)))
                # apply the first X to the ancilla qubit
                qml.PauliX(wires='anc')

                # now check to see which controlled operation to apply
                if op.basis == 'X':
                    qml.CNOT(wires=['anc', op.wires[0]])
                if op.basis == 'Y':
                    qml.CY(wires=['anc', op.wires[0]])
                if op.basis == 'Z':
                    qml.CZ(wires=['anc', op.wires[0]])

                # apply second X to the ancilla qubit
                qml.PauliX(wires='anc')

                # in the special case that i = j
                if I_index == J_index:
                    qml.Hadamard(wires='anc')
                    break

                # now apply the operation
                op.queue()
                qml.Barrier(wires=['anc'] + list(range(num_wires)))

            # if the operation is the last one we want the derivative of
            elif theta_index == J_index:
                qml.Barrier(wires=['anc'] + list(range(num_wires)))
                # now check to see which controlled operation to apply
                if op.basis == 'X':
                    qml.CNOT(wires=['anc', op.wires[0]])
                if op.basis == 'Y':
                    qml.CY(wires=['anc', op.wires[0]])
                if op.basis == 'Z':
                    qml.CZ(wires=['anc', op.wires[0]])

                # do not apply the operation in this case
                # apply final Hadamard to ancilla qubit
                qml.Hadamard(wires='anc')
                qml.Barrier(wires=['anc'] + list(range(num_wires)))

                # end the for loop
                break

            # if an intermediate operation
            else:
                op.queue()

    return qml.expval(qml.PauliZ('anc'))
# end of Aij()


@qml.qnode(dev2)
def Ci(alpha, thetas, ansatz_tape, H, i):
    """
    Circuit to implement the calculation of Ci (for single term of H)

    Parameters:
        alpha - needed phase angle
        thetas - the trainable angles used to create |phi(theta)>
        ansatz_tape - the ansatz we are using to create the trial state
        H - the Hamiltonian we are approximating the ground state of
        i - the index of vector C we are calculating

    Return:
        C_i - the expectation value
    """

    theta_index = -1  # index used for parameter checking

    # initial state
    qml.Hadamard(wires='anc')
    qml.RZ(alpha, wires='anc')
    qml.Barrier(wires=['anc'] + list(range(num_wires)))

    # the circuit

    for op in ansatz_tape.operations:
        # if the current operation is non-parameterised, just queue it
        if op.num_params == 0:
            op.queue()

        # if the operation is parameterised
        else:
            theta_index += 1

            # if the operation is the first one we want the derivative of
            if theta_index == i:
                qml.Barrier(wires=['anc'] + list(range(num_wires)))
                # apply the first X to the ancilla qubit
                qml.PauliX(wires='anc')

                # now check to see which controlled operation to apply
                if op.basis == 'X':
                    qml.CNOT(wires=['anc', op.wires[0]])
                if op.basis == 'Y':
                    qml.CY(wires=['anc', op.wires[0]])
                if op.basis == 'Z':
                    qml.CZ(wires=['anc', op.wires[0]])

                # apply second X to the ancilla qubit
                qml.PauliX(wires='anc')

                # now apply the operation
                op.queue()
                qml.Barrier(wires=['anc'] + list(range(num_wires)))

            # if an intermediate operation
            else:
                op.queue()

    # apply the controlled-Hamiltonian operation
    qml.Barrier(wires=['anc'] + list(range(num_wires)))
    qml.ControlledQubitUnitary(H.matrix, control_wires='anc', wires=H.wires)
    qml.Barrier(wires=['anc'] + list(range(num_wires)))

    # apply final Hadamard to ancilla qubit
    qml.Hadamard(wires='anc')

    return qml.expval(qml.PauliZ(wires='anc'))
# end of Ci()


def create_A(ansatz_tape, theta):
    '''
    Create the matrix A

    Parameters:
        ansatz_tape - the ansatz we are using to create the trial state
        theta - the trainable angles used to create |phi(theta)>

    Return:
        A - the matrix A
    '''

    A = np.zeros((len(theta), len(theta)))
    for i in range(len(theta)):
        for j in range(len(theta)):
            A[i, j] = Aij(0, theta, ansatz_tape, i, j)

    return A
# end of create_A()


def create_C(ansatz_tape, theta, H):
    '''
    Create the vector C

    Parameters:
        ansatz_tape - the ansatz we are using to create the trial state
        theta - the trainable angles used to create |phi(theta)>
        H - the Hamiltonian we are approximating the ground state of

    Return:
        C - the vector C
    '''

    C = np.zeros(len(theta))
    for i in range(len(theta)):
        for h in H.ops:
            C[i] += Ci(np.pi/2, theta, ansatz_tape, h, i)

    return C
# end of create_C()


if __name__ == "__main__":
    main()
