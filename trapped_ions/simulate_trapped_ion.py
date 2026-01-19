import numpy as np
import matplotlib.pyplot as plt
from qutip import *

# Define system parameters
Delta = 1.0  # Detuning
nu = 1.0     # Motional frequency
Omega = 1.0  # Rabi frequency
eta = 0.1    # Lamb-Dicke parameter
n_thermal = 5  # Average thermal phonon number

# Define operators
N = 20  # Truncation for the Fock space

# QHO operators
a = destroy(N)  # Annihilation operator
a_dag = a.dag()  # Creation operator

# Spin operators
sz = sigmaz()
s_plus = sigmap()
s_minus = sigmam()

# Displacement operator
D_eta = displace(N, 1j * eta)

# Hamiltonian
H = (Delta / 2) * tensor(sz, qeye(N)) + nu * tensor(qeye(2), a_dag * a) + \
    (Omega / 2) * (tensor(s_plus, D_eta) + tensor(s_minus, D_eta.dag()))

# Initial state: thermal state for motion and ground state for spin
thermal_state = thermal_dm(N, n_thermal)
ground_state = basis(2, 1)  # |g>
initial_state = tensor(ground_state * ground_state.dag(), thermal_state)

# Time evolution
T = 10  # Total time
num_steps = 500
times = np.linspace(0, T, num_steps)

# Solve the master equation
result = mesolve(H, initial_state, times, [], [])

# Extract probabilities
p_g = []  # Probability of spin in |g>
p_e = []  # Probability of spin in |e>

for state in result.states:
    p_g.append(expect(tensor(basis(2, 1) * basis(2, 1).dag(), qeye(N)), state))
    p_e.append(expect(tensor(basis(2, 0) * basis(2, 0).dag(), qeye(N)), state))

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(times, p_g, label='P(|g>)')
plt.plot(times, p_e, label='P(|e>)')
plt.xlabel('Time')
plt.ylabel('Probability')
plt.title('Occupancy Probabilities for Spin States')
plt.legend()
plt.grid()
plt.show()