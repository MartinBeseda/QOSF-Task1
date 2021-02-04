import collections

import numpy as np
import qiskit as qk
import scipy.optimize as sciopt

# backend = qk.Aer.get_backend('statevector_simulator')
backend = qk.Aer.get_backend('qasm_simulator')

# noinspection SpellCheckingInspection
n_qubits = 7

# Number of qubits per one state
n_qubit_state = int((n_qubits - 1)/2)


# p1 = θ, p2 = ϕ, p3 = λ, p4 = θ, ...
# u_params = (qk.circuit.Parameter('θ'), qk.circuit.Parameter('ϕ'), qk.circuit.Parameter('λ'))
u_params = list(sum([(qk.circuit.Parameter('{}θ'.format(i)),
                      qk.circuit.Parameter('{}ϕ'.format(i)),
                      qk.circuit.Parameter('{}λ'.format(i))) for i in range(int(n_qubits/2 + 1), n_qubits)], ()))


# noinspection SpellCheckingInspection
def build_circ(circ: qk.QuantumCircuit, qr: qk.QuantumRegister, cr: qk.ClassicalRegister):
    # Setting of the 1st state
    circ.u(np.pi, np.pi, np.pi, qr[1])

    # U3-gate for 'fitting' of q2 state
    for i in range(n_qubit_state):
        circ.u(u_params[3*i],
               u_params[3*i+1],
               u_params[3*i+2],
               qr[i + n_qubit_state])

    # SWAP test implementation
    circ.h(qr[0])
    # Apply Fredkin gate in a pair-wise manner
    for i in range(1, n_qubit_state + 1):
        circ.cswap(qr[0], qr[i], qr[i + n_qubit_state])
    circ.h(qr[0])
    circ.measure(qr[0], cr[0])

    # circuit.draw('mpl')
    # plt.show()


# noinspection SpellCheckingInspection
def swap_test(circ: qk.QuantumCircuit) -> float:
    n_shots = 1024
    job = qk.execute(circ, backend, shots=n_shots)
    job_result = job.result()
    counts = job_result.get_counts(circ)

    return counts['0']/n_shots


# noinspection SpellCheckingInspection
def err(params: collections.abc.Collection, circ: qk.QuantumCircuit) -> float:
    tmp_circ_loc = circ.bind_parameters({u_params[i]: params[i] for i in range(len(u_params))})
    res = swap_test(tmp_circ_loc)
    return (1 - res)**2


if __name__ == '__main__':
    # noinspection SpellCheckingInspection
    qreg_q = qk.QuantumRegister(n_qubits, 'q')
    # noinspection SpellCheckingInspection
    creg_c = qk.ClassicalRegister(1, 'c')
    circuit = qk.QuantumCircuit(qreg_q, creg_c)

    build_circ(circuit, qreg_q, creg_c)

    result = sciopt.differential_evolution(err,
                                           bounds=[(0, 2*np.pi)]*3*int(n_qubits/2),
                                           args=(circuit,),
                                           maxiter=15)

    tmp_circ = circuit.bind_parameters({u_params[i]: result.x[i] for i in range(3*int(n_qubits/2))})
    print('SWAP test after optimization: P(q0 = 0) = {}'.format(swap_test(tmp_circ)))
    print('Parameters of U3-gates:')
    for i in range(n_qubit_state):
        print('q{}: θ = {}, ϕ = {}, λ = {}'.format(i + n_qubit_state + 1, result.x[3*i], result.x[3*i+1], result.x[3*i+2]))
