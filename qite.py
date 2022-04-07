#! /usr/bin/python3

# import modules {{{
from pennylane import numpy as np
import pennylane as qml
import scipy.linalg as linalg
from scipy.linalg import solve, inv, eigvals, lstsq
from scipy.integrate import RK45, solve_ivp
from scipy.optimize import minimize, least_squares
# }}}


# set global variables {{{
# Number of wires in the circuit
num_wires = 2

dev = qml.device('default.qubit', wires=num_wires)
dev2 = qml.device('default.qubit', wires=['anc'] + list(range(num_wires)))

# construct the Hamiltonian we wish to use
obs = [qml.PauliZ(0) @ qml.PauliX(1),
       qml.PauliX(0) @ qml.PauliZ(1),
       qml.PauliZ(0) @ qml.PauliZ(1)]
coeff = [1, 1, 3]
H = qml.Hamiltonian(coeff, obs)

# initial values of the trial state parameters theta
theta = np.array([0, 0, 0, 0, np.pi/2, np.pi/2, 0, 0],
                 requires_grad=True)
theta_dot = np.zeros_like(theta)

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
# }}}


# main() {{{
def main():
    """
    main method to run program
    """

    global theta

    print('-----------------------------------------')
    print('The Hamiltonian we are approximating the ground state of')
    print('H =', H)

    print('-----------------------------------------')

    # Update the values of theta
    # TODO look at ansatz update in Aij and Ci
    #theta = np.array([np.pi/2, np.pi, np.pi/3, 0,
            #np.pi/2, 0, np.pi/4, np.pi/2], requires_grad=True)
    #ansatz_tape.set_parameters(theta)

    print('The ansatz being used to prepare the trial state |phi(theta)>')
    print(ansatz_tape.draw())
    print('variance of Hamiltonian w.r.t. |phi(theta)>', compute_variance())

    print('-----------------------------------------')

    print('Here we construct the matrix ' +
          'A_ij = Re( (d <phi(theta)|/d theta_i) (d |phi(theta)>/d theta_j) )')
    A = create_A(theta)
    print('A =', A)

    print('-----------------------------------------')

    print('Here we construct the vector ' +
          'C_i = -Re( (d <phi(theta)|/d theta_i) (H |phi(theta)>) )')
    C = create_C(theta)
    print('C =', C)

    print('-----------------------------------------')
    print(qml.draw(Aij)(np.pi/2, theta, 1, 5))

    # TODO properly calculate the phase angle

    # TODO add ODE solver to solve A theta_dot = C

    print('-----------------------------------------')

    print('theta before =', theta)

    def std_ode(t, theta):
        # Tikhonov Regularization for ill posed matrix A and vector C
        lamb = 1
        A2 = np.concatenate((A, np.sqrt(lamb) * np.eye(len(theta))))
        C2 = np.concatenate((C, np.zeros(len(theta))))

        return linalg.lstsq(A2, C2)[0]

    theta_dot = std_ode(0, theta)
    print('theta_dot =', theta_dot)

    def e_t(thetas_dot, thetas):
        # do A/C need to be create_A/C(thetas)?
        # if so, code takes significantly longer
        return (compute_variance()
                + np.sum([[thetas_dot[i] * thetas_dot[j] * A[i][j]
                            for i in range(len(thetas))]
                            for j in range(len(thetas))])
                - 2 * np.sum(thetas_dot[i] * C[i] for i in range(len(thetas))))

    print('e_t =', e_t(theta_dot, theta))

    def min_ode(t, theta):
        # return the theta_dot = argmin || |e_t> ||^2
        return minimize(e_t, args=theta, x0=theta_dot, method="COBYLA").x

    print('-----------------------------------------')
    print('-----------------------------------------')
    # DO NOT RUN THIS CODE, MAKES COMPUTER SAD
    #theta_dott = min_ode(0, theta)
    #print('theta_dot from min_ode =', theta_dott)

    # TODO add way to update theta
    # DO NOT RUN THIS CODE, MAKES COMPUTER SAD
    theta_std = solve_ivp(std_ode, (0, 1), theta, method='RK45').y[:,-1]
    #theta_min = solve_ivp(min_ode, (0, 1), theta, method='RK45').y[:,-1]
    #print('theta from std_ode =', theta_std)
    print('-----------------------------------------')
    #print('theta from min_ode =', theta_min)

    print('-----------------------------------------')
    print('-----------------------------------------')
    print('state_init =', ansatz_with_state())
    print(ansatz_tape.draw())
    print('-----------------------------------------')
    ansatz_tape.set_parameters(theta_std)
    print('state_std =', ansatz_with_state())
    print(ansatz_tape.draw())
    print('-----------------------------------------')
    #ansatz_tape.set_parameters(theta_min)
    #print('state_min =', ansatz_with_state())
    #print(ansatz_tape.draw())

    # TODO graph the results
# end of main() }}}


# ansatz(ansatz_ops, thetas) {{{
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
# end of ansatz() }}}


# compute state, expected value, and variance {{{
@qml.qnode(dev)
def ansatz_with_state():
    # take the ansatz tape and then measure the expected value

    #for op in ansatz_tape.operations:
        #op.queue()
    for op in ansatz_tape.operations:
        op.queue()

    return qml.state()


@qml.qnode(dev)
def ansatz_with_expval():
    # take the ansatz tape and then measure the expected value

    for op in ansatz_tape.operations:
        op.queue()

    return qml.expval(H)


@qml.qnode(dev)
def ansatz_with_sq_expval():
    # take the ansatz tape and then measure the variance

    for op in ansatz_tape.operations:
        op.queue()

    # why is there no better way to compute the variance of the Hamiltonian
    return qml.expval(qml.Hermitian(np.dot(qml.matrix(H), qml.matrix(H)),
                                    wires=H.wires))


def compute_variance():
    return ansatz_with_sq_expval() - (ansatz_with_expval())**2
# }}}


# Aij(alpha, thetas, i, j) {{{
@qml.qnode(dev2)
def Aij(alpha, thetas, i, j):
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
# end of Aij() }}}


# Ci(alpha, thetas, H, i) {{{
@qml.qnode(dev2)
def Ci(alpha, thetas, H, i):
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
    qml.ControlledQubitUnitary(qml.matrix(H),
                               control_wires='anc', wires=H.wires)
    qml.Barrier(wires=['anc'] + list(range(num_wires)))

    # apply final Hadamard to ancilla qubit
    qml.Hadamard(wires='anc')

    return qml.expval(qml.PauliZ(wires='anc'))
# end of Ci() }}}


# create_A(thetas) {{{
def create_A(thetas):
    '''
    Create the matrix A

    Parameters:
        ansatz_tape - the ansatz we are using to create the trial state
        theta - the trainable angles used to create |phi(theta)>

    Return:
        A - the matrix A
    '''

    A = np.zeros((len(thetas), len(thetas)))
    for i in range(len(thetas)):
        for j in range(len(thetas)):
            A[i, j] = Aij(0, thetas, i, j)

    return A
# end of create_A() }}}


# create_C(thetas) {{{
def create_C(thetas):
    '''
    Create the vector C

    Parameters:
        ansatz_tape - the ansatz we are using to create the trial state
        thetas - the trainable angles used to create |phi(theta)>
        H - the Hamiltonian we are approximating the ground state of

    Return:
        C - the vector C
    '''

    C = np.zeros(len(thetas))
    for i in range(len(thetas)):
        for h in H.ops:
            C[i] += Ci(np.pi/2, thetas, h, i)

    return C
# end of create_C() }}}


if __name__ == "__main__":
    main()
