import qiskit as qk
from numpy import pi
import matplotlib.pyplot as plt  # For circuit visualization
import scipy.optimize as sciopt
import collections

# backend = Aer.get_backend('statevector_simulator')
backend = qk.Aer.get_backend('qasm_simulator')

u_params = (qk.circuit.Parameter('θ'), qk.circuit.Parameter('ϕ'), qk.circuit.Parameter('λ'))


def build_circ(circ: qk.QuantumCircuit, qr: qk.QuantumRegister, cr: qk.ClassicalRegister):
    circuit.u(u_params[0], u_params[1], u_params[2], qr[2])
    circ.h(qr[0])
    # circuit.barrier(qr[0], qr[1], qr[2])

    # Fredkin gate
    circ.cx(qr[2], qr[1])
    circ.h(qr[2])
    circ.cx(qr[1], qr[2])
    circ.tdg(qr[2])
    circ.cx(qr[0], qr[2])
    circ.t(qr[2])
    circ.cx(qr[1], qr[2])
    circ.tdg(qr[1])
    circ.tdg(qr[2])
    circ.cx(qr[0], qr[2])
    circ.cx(qr[0], qr[1])
    circ.t(qr[2])
    circ.t(qr[0])
    circ.tdg(qr[1])
    circ.h(qr[2])
    circ.cx(qr[0], qr[1])
    circ.s(qr[1])
    circ.cx(qr[2], qr[1])
    # circ.barrier(qr[0], qr[1], qr[2])
    # End of Fredkin gate

    circ.h(qr[0])
    # circ.barrier(qr[0], qr[1], qr[2])
    circ.measure(qr[0], cr[0])

    # circuit.draw('mpl')
    # plt.show()


# noinspection SpellCheckingInspection
def swap_test(circ: qk.QuantumCircuit) -> float:

    # Launch the job - QASM simulator
    n_shots = 1024
    job = qk.execute(circ, backend, shots=n_shots)
    job_result = job.result()
    counts = job_result.get_counts(circ)
    # print('counts {}:'.format(counts))
    # print(counts)
    # print('Measured P(q0 = 0) = {}'.format(counts['0']/(counts['1']+counts['0'])))
    return counts['0']/n_shots


def err(params: collections.abc.Collection, circ: qk.QuantumCircuit) -> float:
    # print('err')
    # print({u_params[i]: params[i] for i in range(len(u_params))})
    tmp_circ = circ.bind_parameters({u_params[i]: params[i] for i in range(len(u_params))})
    # circuit.draw('mpl')
    # plt.show()
    result = swap_test(tmp_circ)
    return (1 - result)**2


if __name__ == '__main__':

    qreg_q = qk.QuantumRegister(3, 'q')
    creg_c = qk.ClassicalRegister(1, 'c')
    circuit = qk.QuantumCircuit(qreg_q, creg_c)

    init_u_params = (0.7*pi/2, pi/2, pi/3)

    build_circ(circuit, qreg_q, creg_c)

    tmp_circ1 = circuit.bind_parameters({u_params[i]: init_u_params[i] for i in range(3)})
    print('Before optimization: {}'.format(swap_test(tmp_circ1)))

    result = sciopt.differential_evolution(err,
                                           bounds=[(0, 2*pi)]*3,
                                           args=(circuit,),
                                           maxiter=5)

    tmp_circ2 = circuit.bind_parameters({u_params[i]: result.x[i] for i in range(3)})
    print(result.x)
    print('After optimization: {}'.format(swap_test(tmp_circ2)))
    # tmp_circ2.measure_all()
    # job_result = qk.execute(tmp_circ2, backend, shots=1024).result()
    # counts = job_result.get_counts(tmp_circ2)



    # # Visualize the circuit
    # circuit.draw('mpl')
    # plt.show()

    # # Launch the job - statevector simulator
    # job = qiskit.execute(circuit, backend)
    # job_result = job.result()
    # out_state = job_result.get_statevector(circuit)


