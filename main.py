import collections
import matplotlib.pyplot as plt
import numpy as np
import qiskit as qk
import scipy.optimize as sciopt
import math

statevector_backend = qk.Aer.get_backend('statevector_simulator')
backend = qk.Aer.get_backend('qasm_simulator')


def build_t2_circ(circ: qk.QuantumCircuit,
                  qr: qk.QuantumRegister,
                  cr: qk.ClassicalRegister,
                  n_qubit_state: int,
                  u_params: collections.abc.Collection):

    # Setting of the random states & U3-gates for 'fitting' of states
    for i in range(n_qubit_state):
        circ.u(np.random.rand()*2*np.pi, np.random.rand()*2*np.pi, np.random.rand()*2*np.pi, qr[i+1])

        circ.u(u_params[3*i],
               u_params[3*i+1],
               u_params[3*i+2],
               qr[i + n_qubit_state + 1])

    # SWAP test implementation
    circ.h(qr[0])

    # Apply Fredkin gate in a pair-wise manner
    for i in range(1, n_qubit_state + 1):
        circ.cswap(qr[0], qr[i], qr[i + n_qubit_state])
    circ.h(qr[0])
    circ.measure(qr[0], cr[0])

    circ.draw('mpl')
    plt.show()


def build_t3_circ(circ: qk.QuantumCircuit,
                  qr: qk.QuantumRegister,
                  n_qubit_state: int):

    # Random initialization of the 'pattern' states
    for i in range(n_qubit_state):
        if np.random.choice((0, 1)):
            circ.x(qr[i+1])

    # SWAP test implementation
    circ.h(qr[0])


def swap_test(circ: qk.QuantumCircuit) -> float:
    n_shots = 1024
    job = qk.execute(circ, backend, shots=n_shots)
    job_result = job.result()
    counts = job_result.get_counts(circ)

    return counts['0']/n_shots


def err(params: collections.abc.Collection, circ: qk.QuantumCircuit, u_params: collections.abc.Collection) -> float:
    tmp_circ_loc = circ.bind_parameters({u_params[i]: params[i] for i in range(len(u_params))})
    res = swap_test(tmp_circ_loc)
    return (1 - res)**2


def get_spherical_coords(lst: collections.abc.Collection) -> list:
    alpha = np.real(2*np.arccos(lst[0]))
    beta = np.real(1/1j*np.log(lst[1] / np.sin(alpha/2)))
    return list((1, alpha, beta))


def get_u_params(n_qubits: int):
    return list(sum([(qk.circuit.Parameter('{}θ'.format(i)),
                      qk.circuit.Parameter('{}ϕ'.format(i)),
                      qk.circuit.Parameter('{}λ'.format(i))) for i in range(int(n_qubits / 2 + 1), n_qubits)], ()))


def task_1_1():
    """
    Different states are created by application of U3-gate to |0> state.
    """

    qreg_q = qk.QuantumRegister(1, 'q')
    creg_c = qk.ClassicalRegister(1, 'c')
    circuit = qk.QuantumCircuit(qreg_q, creg_c)

    for i in range(5):
        circuit.reset(qreg_q[0])
        circuit.u(np.random.rand()*2*np.pi, np.random.rand()*2*np.pi, np.random.rand()*2*np.pi, qreg_q[0])
        tmp_res = qk.execute(circuit, statevector_backend).result()
        out_state = tmp_res.get_statevector()
        qk.visualization.plot_bloch_vector(get_spherical_coords(out_state), coord_type='spherical')
        plt.show()


def task_1_2():
    """
    Fitting of q2 to the random state in q1.
    """

    n_qubits = 5

    # Number of qubits per one state
    n_qubit_state = int((n_qubits - 1) / 2)

    # p1 = θ, p2 = ϕ, p3 = λ, p4 = θ, ...
    u_params = get_u_params(n_qubits)

    qreg_q = qk.QuantumRegister(n_qubits, 'q')
    creg_c = qk.ClassicalRegister(1, 'c')
    circuit = qk.QuantumCircuit(qreg_q, creg_c)

    build_t2_circ(circuit, qreg_q, creg_c, n_qubit_state, u_params)

    result = sciopt.differential_evolution(err,
                                           bounds=[(0, 2*np.pi)]*3*int(n_qubits/2),
                                           args=(circuit, u_params),
                                           maxiter=15)

    tmp_circ = circuit.bind_parameters({u_params[i]: result.x[i] for i in range(3*int(n_qubits/2))})
    print('SWAP test after optimization: P(q0 = 0) = {}'.format(swap_test(tmp_circ)))
    print('Parameters of U3-gates:')
    for i in range(n_qubit_state):
        print('q{}: θ = {}, ϕ = {}, λ = {}'.format(i + n_qubit_state + 1,
                                                   result.x[3*i],
                                                   result.x[3*i+1],
                                                   result.x[3*i+2]))


def task_1_3(n_qubits: int):
    n_qubit_state = int((n_qubits - 1) / 2)

    qreg_q = qk.QuantumRegister(n_qubits, 'q')
    creg_c = qk.ClassicalRegister(1, 'c')
    circuit = qk.QuantumCircuit(qreg_q, creg_c)

    # Build randomly initialized circuit
    build_t3_circ(circuit, qreg_q, n_qubit_state)

    # List of detected pattern states
    swap_results = []

    # Put Fredkin gate and measure for different pairs of qubits
    for i in range(n_qubit_state):
        tmp_circ = circuit.copy()
        # Apply Fredkin gate to the specific qubit pair
        tmp_circ.cswap(qreg_q[0], qreg_q[i + 1], qreg_q[i + 1 + n_qubit_state])
        tmp_circ.h(qreg_q[0])
        tmp_circ.measure(qreg_q[0], creg_c[0])
        swap_results.append(swap_test(tmp_circ))

    # Invert 'fitted' qubits accordingly
    for i in range(n_qubit_state):
        if math.isclose(round(swap_results[i], 1), 0.5):
            circuit.x(i + n_qubit_state + 1)

    circuit.draw('mpl')
    plt.show()


if __name__ == '__main__':
    print('Task 1.1')
    task_1_1()
    print('Task 1.2')
    task_1_2()
    print('Task 1.3')
    task_1_3(7)
