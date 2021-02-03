from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from numpy import pi
from qiskit import Aer  # For backends
import matplotlib.pyplot as plt  # For circuit visualization
import qiskit
# backend = Aer.get_backend('statevector_simulator')
backend = Aer.get_backend('qasm_simulator')


# noinspection SpellCheckingInspection
def swap_test(circ, qr, cr):
    circ.h(qr[0])
    # circuit.barrier(qr[0], qr[1], qr[2])
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
    circ.h(qr[0])
    # circ.barrier(qr[0], qr[1], qr[2])

    circ.measure(qreg_q[0], cr[0])

    # Launch the job - QASM simulator
    job = qiskit.execute(circ, backend, shots=1024)
    job_result = job.result()
    counts = job_result.get_counts(circ)
    # print(counts)
    # print('Measured P(q0 = 0) = {}'.format(counts['0']/(counts['1']+counts['0'])))
    return counts['0']/(counts['1']+counts['0'])


def err(circ: qiskit.QuantumCircuit, qr: qiskit.QuantumRegister, u_params: list) -> float:
    circuit.u(0.7*pi/2, pi/2, pi/3, qreg_q[2])
    result = swap_test(circuit, qreg_q, creg_c)
    
    return (0.5 - result)**2


if __name__ == '__main__':

    qreg_q = QuantumRegister(3, 'q')
    creg_c = ClassicalRegister(1, 'c')
    circuit = QuantumCircuit(qreg_q, creg_c)

    # Prepare the states
    circuit.u(0.7*pi/2, pi/2, pi/3, qreg_q[2])
    # circuit.barrier(qreg_q[0], qreg_q[1], qreg_q[2])

    # P(q0 = 0)
    # result = swap_test(circuit, qreg_q, creg_c)

    # # Visualize the circuit
    # circuit.draw('mpl')
    # plt.show()

    # # Launch the job - statevector simulator
    # job = qiskit.execute(circuit, backend)
    # job_result = job.result()
    # out_state = job_result.get_statevector(circuit)


