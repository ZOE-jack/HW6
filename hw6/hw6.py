# Circuit building and running
from qiskit.circuit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit import transpile, qpy
from qiskit_aer import AerSimulator
# trotter evolution   
from qiskit.synthesis import SuzukiTrotter
from qiskit.circuit.library import PauliEvolutionGate
# hamiltonian building
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer import AerSimulator
# other tools
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.linalg import expm

print("Please forgive me about late submission of cleaner code.\n"
      + "I had been busy of proposal last few weeks.\n"
      + "I only take the qiskit documentations as references.")

## 1. Spin precession
pauli_z = np.array([[1, 0], [0, -1]])
pauli_x = np.array([[0, 1], [1, 0]])
pauli_y = np.array([[0, -1j], [1j, 0]])
H = pauli_z
psi0 = np.array([[np.cos(0.3*np.pi/2)], [np.sin(0.3*np.pi/2)]])
eigu, eigv = np.linalg.eigh(H)

time = np.arange(0, 50, 0.1)
result = np.empty((0, 3))
for t in time:
    sumv = np.zeros((2,1), dtype = "complex128")
    for index in range(len(eigu)):
        egv = eigv[:, index].reshape(2, 1)
        phi = np.exp(-1j*0.5*eigu[index]*t) * np.dot(egv.T, psi0) * egv
        sumv += phi
    recordx_expt = np.dot(np.conj(sumv.T), np.matmul(pauli_x, sumv)).real.item()
    recordy_expt = np.dot(np.conj(sumv.T), np.matmul(pauli_y, sumv)).real.item()
    recordz_expt = np.dot(np.conj(sumv.T), np.matmul(pauli_z, sumv)).real.item()
    result = np.append(result, np.array([[recordx_expt, recordy_expt, recordz_expt]]), axis=0)

for index, pauli in enumerate(["<X>", "<Y>", "<Z>"]):
    plt.plot(time, result[:, index], label=pauli)
plt.ylabel("expectation value")
plt.xlabel("time")
plt.title("Spin precession")
plt.legend()
plt.show()

## 2.1a Inplement Suzuki by qiskit

# A. build the hamiltonian of Ising and the obersable Z, which represents M in question.
Z = SparsePauliOp(["ZIII", "IZII", "IIZI", "IIIZ"])
H = SparsePauliOp(["ZZII", "IZZI", "IIZZ", "ZIIZ", "XIII", "IXII", "IIXI", "IIIX"]
                  , coeffs=[1]*4+[-1]*4)

# B. build the circuits of different time, here we just load the qiskit circuit data I built before.
# circs = []
# # Take about 50 seconds
# del_t = 0.1
# time = np.arange(0, 10, del_t)
# for times, t in tqdm(enumerate(time)):
#     trotter = PauliEvolutionGate(operator=H, time=t)
#     circ = SuzukiTrotter(order=2, reps=times).synthesize(trotter)
#     circ.measure_all()
#     circs.append(transpile(circ, basis_gates=['rx','ry','rz','cx']))
print("Please wait for loading the data...")
with open('circs.qpy', 'rb') as data:
    circs = qpy.load(data)
print("It just done! Don't be scared!")

# C. Simulate the circuit and count the expectation value of <Zi>
def single_marginal(data, indice):
    """count the "0" and "1" state of the ith qubit."""
    count = {"0":0, "1":0}
    for key, value in data.items():
        if key[indice] == "1":
            count["1"] += value
        else:
            count["0"] += value
    return count

simulator = AerSimulator() # noise-free simulator
job = simulator.run(circs, shots=1000)
result = job.result()
counts = result.get_counts()

record_qiskit = []
for num in range(len(counts)):# count the expectation value
    val = 0
    for index in range(4):
        data = single_marginal(counts[num], index)
        val += (data["0"] - data["1"])/1000
    record_qiskit.append(val)

## 2.1b Inplement Suzuki decomposition by expm function of scipy
ZZ = SparsePauliOp(["ZZII", "IZZI", "IIZZ", "ZIIZ"], coeffs=[1]*4).to_matrix() # ZZ interaction
X = SparsePauliOp(["XIII", "IXII", "IIXI", "IIIX"], coeffs=[-1]*4).to_matrix() # X field
matrix_Z = SparsePauliOp(["ZIII", "IZII", "IIZI", "IIIZ"]).to_matrix() # M oberservable
psi = np.zeros((16,1))
psi[0] += 1

del_t = 0.1
time = np.arange(0, 10, del_t)
U = np.matmul(expm(-1j*ZZ*del_t), expm(-1j*X*del_t)) # An evolution operator U.
record = [np.dot(np.conj(psi.T), np.matmul(matrix_Z, psi)).real.item()] # the record containing the expval of psi0.

for _ in time[1:]:
    psi_n = np.matmul(U, psi) # U gate apply to psi_t to get psi_t+1
    exval = np.dot(np.conj(psi_n.T), np.matmul(matrix_Z, psi_n)).real.item() # calculate expval of psi_t+1 
    record.append(exval)
    psi = psi_n # put the psi_t+1 as the next psi to calculate psi_t+2

## 2.2 By normal calculation in class 
matrix_H = H.to_matrix()
matrix_Z = Z.to_matrix()
psi0 = np.zeros((16,1))
psi0[0] += 1
eigu, eigv = np.linalg.eig(matrix_H)
time = np.arange(0, 10, 0.1)
result = np.array([])
dim = len(eigu)
for t in time:
    sumv = np.zeros((dim,1), dtype = "complex128")
    for index in range(dim):
        egv = eigv[:, index].reshape(dim, 1)
        phi = np.exp(-1j*eigu[index]*t) * np.dot(egv.T, psi0) * egv
        sumv += phi
    record_z = np.dot(np.conj(sumv.T), np.matmul(matrix_Z, sumv)).real.item()
    result = np.append(result, record_z)

plt.plot(time, record_qiskit, label="qiskit-simulate")
plt.plot(time, record, label="suzuki with expm func")
plt.plot(time, result, label="exact solution")
plt.title("Result Comparing in $\Delta$t = 0.1")
plt.xlabel("time")
plt.ylabel("<M>", rotation=0)
plt.grid()
plt.legend()
plt.show()
