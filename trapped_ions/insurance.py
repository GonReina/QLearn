# %% [markdown]
# # Replicate QST article sims
# 
# ### 1. RSB Hamiltonian Figure 1b
# 
# In the Fock mixtures paper, Figure 1b shows the dynamics under the RSB Hamiltonian with dissipation. The full ion trap QHO hamiltonian is:
# $$
# H = \frac{\Delta}{2}\sigma_{z} + \nu aa^{\dagger} + \frac{\Omega}{2}\big[\sigma^{+}D(i\eta) +\sigma^{-}D^{\dagger}(i\eta) \big]
# $$
# 
# Where $D(i\eta) = e^{i\eta(a + a^{\dagger})}$ is the displacement operator. In the Lamb-Dicke regime, we can approximate this Hamiltonian to the first red sideband (RSB) interaction:
# $$
# H' = \frac{i\Omega\eta}{2}\big[\sigma_x(a^{\dagger} + a )\big]
# $$
# 

# %%
import numpy as np
import matplotlib.pyplot as plt
from qutip import *

# Define system parameters
Delta = 1.0  # Detuning
nu = 1.0     # Motional frequency
Omega=.02  # Rabi frequency
eta = 0.02    # Lamb-Dicke parameter
n_thermal = 3  # Average thermal phonon number
n0 = 1
gamma = 1000  # Spin relaxation
beta = 0.01    # Inverse temperature
kT = 1.0      # Boltzmann constant times temperature
tau = 2 * np.pi / (eta * Omega * np.sqrt(n0))  # Pulse duration

# Define operators
N = 20  # Truncation for the Fock space

# QHO operators
a = destroy(N)  # Annihilation operator
a_dag = a.dag()  # Creation operator
xop = a + a_dag # Position operator

# Spin operators
sz = sigmaz()
s_plus = sigmap()
s_minus = sigmam()

# Full size operators
s_z_full = tensor(sz, qeye(N))
s_plus_full = tensor(s_plus, qeye(N))
a_full = tensor(qeye(2), a)
a_dag_full = tensor(qeye(2), a_dag)
s_minus_full = tensor(s_minus, qeye(N))

# Hamiltonian (Red Sideband for Pulse)
# H_pulse = 1j*eta*Omega/2 * (a_full* s_plus_full - a_dag_full * s_minus_full)
# Displacement operator D(i*eta)
# Note: In the paper, the argument is i*eta.
D_op = tensor(qeye(2), displace(N, 1j * eta))

# RSB Hamiltonian (Figure 1b)
H_pulse = -1j*eta*Omega/2 * (a_full* s_plus_full - a_dag_full * s_minus_full)

# Hamiltonian (Free evolution for Dissipation) - Assuming H0 during dissipation phase? 
# For now, let's assume H_diss = 0 for the dissipative step relative to the rotating frame.
H_diss = Delta*s_z_full/2 + nu*a_dag_full*a_full

# --- Collapse Operators for Dissipation with recoil ---
cosmax = 100
cosal = np.arange(-cosmax, cosmax + 1) / cosmax
W = 3 * (cosal**2 + 1) / 4 * 1 / (2 * cosmax)
W = W / np.sum(W)
G = gamma / 2  # This is Γ/2 in the master equation
tg = 2 / G  # Dissipation duration adjusted for rates
c_ops = []
for i, c in enumerate(cosal):
    U_mot = (1j * c * eta * xop).expm()
    U_full = tensor(qeye(2), U_mot)
    c_ops.append(np.sqrt(G * W[i]) * (s_minus_full * U_full))

# Add sigma_minus decay without recoil (optional, for comparison)
c_ops.append(np.sqrt(gamma) * s_minus_full)

# --- Construct Liouvillians ---
# 1. Pulse Step: Coherent RSB drive, NO dissipation (or minimal?)
# Typically only Unitary part active during pulse
L_pulse = liouvillian(H_pulse, []) 

# 2. Dissipation Step: No Drive, ONLY Dissipation (Optical Pumping)
# H = 0 (or H0), c_ops active
L_diss = liouvillian(H_diss, c_ops)

# --- Propagators ---
# Propagator for Pulse step (duration tau)
prop_pulse = (L_pulse * tau).expm()

# Propagator for Dissipation step (duration tg)
prop_diss = (L_diss * tg).expm()

# Combined Propagator for one cycle (Pulse -> Dissipation)
prop_cycle = prop_diss * prop_pulse 


# --- Simulation ---
num_pulses = 30 # Number of cycles
times_sim = np.arange(num_pulses + 1) # Just integer steps for cycles

# Initial state
rho_thermal = thermal_dm(N, n_thermal)
p_thermal = rho_thermal.diag()
ground_state = basis(2, 1) 
initial_state = tensor(ground_state * ground_state.dag(), rho_thermal)

# Storage
occupancies = np.zeros((num_pulses + 1, N))
rho_vec = operator_to_vector(initial_state)

# Initial occupancy
rho_mat = vector_to_operator(rho_vec)
occupancies[0, :] = rho_mat.ptrace(1).diag().real

# Evolve step-by-step
current_rho_vec = rho_vec
for k in range(1, num_pulses + 1):
    current_rho_vec = prop_cycle * current_rho_vec
    
    # Store
    rho_mat = vector_to_operator(current_rho_vec)
    occupancies[k, :] = rho_mat.ptrace(1).diag().real


# Plot the occupancy of all motional states over cycles
plt.figure()
for n in range(N):
    plt.plot(times_sim, occupancies[:, n], '.-', label=f'n={n}')
plt.xlabel('Pulse Cycle')
plt.ylabel('Occupancy')
plt.title('Occupancy vs Cooling Cycles')
plt.legend()
plt.show()

# Bar plot
p_motional_initial = occupancies[0, :]
p_motional_final = occupancies[-1, :]
plt.figure()
plt.plot(np.arange(N)-0.2, p_motional_initial, 'o-', label='Initial', color='red')
plt.bar(np.arange(N)+0.2, p_motional_final, width=0.4, label='Final', alpha=0.7)
plt.xlabel('Motional State n')
plt.ylabel('Population')
plt.title('State Populations After Cooling Cycles')
plt.legend()
plt.show()

# %% [markdown]
# ### 2. Full Hamiltonian (Figures 5 and 6)

# %%
import numpy as np
import matplotlib.pyplot as plt
from qutip import *

# Define system parameters
Delta = 1.0  # Detuning
nu = 1.0     # Motional frequency
Omega=1  # Rabi frequency
eta = 0.05    # Lamb-Dicke parameter
n_thermal = 1  # Average thermal phonon number
n0 = 1
gamma = 1000  # Spin relaxation
beta = 0.01    # Inverse temperature
kT = 1.0      # Boltzmann constant times temperature
tau = 2 * np.pi / (eta * Omega * np.sqrt(n0))  # Pulse duration

# Define operators
N = 14  # Truncation for the Fock space

# QHO operators
a = destroy(N)  # Annihilation operator
a_dag = a.dag()  # Creation operator
xop = a + a_dag # Position operator

# Spin operators
sz = sigmaz()
s_plus = sigmap()
s_minus = sigmam()

# Full size operators
s_z_full = tensor(sz, qeye(N))
s_plus_full = tensor(s_plus, qeye(N))
a_full = tensor(qeye(2), a)
a_dag_full = tensor(qeye(2), a_dag)
s_minus_full = tensor(s_minus, qeye(N))

# Displaced hamiltonian operator
D_op = tensor(qeye(2), displace(N, 1j * eta))

# Full Hamiltonian (Eq. 1)
# We set detuning Delta = nu to drive the Red Sideband
H_pulse = 0.5 * Delta * s_z_full + nu * a_dag_full * a_full + \
          0.5 * Omega * (s_plus_full * D_op + s_minus_full * D_op.dag())

# Hamiltonian (Free evolution for Dissipation)
H_diss = Delta*s_z_full/2 + nu*a_dag_full*a_full

# The density matrix in the master equation is displaced due to recoil from spontaneous emission.
# We can include this effect in the collapse operators by integrating over the angular distribution of emission
# in a dipolar transition.


# --- Collapse Operators for Dissipation with recoil ---
cosmax = 100
cosal = np.arange(-cosmax, cosmax + 1) / cosmax
W = 3 * (cosal**2 + 1) / 4 * 1 / (2 * cosmax)
W = W / np.sum(W)
G = gamma / 2  # This is Γ/2 in the master equation
tg = 2 / G  # Dissipation duration adjusted for rates
c_ops = []
for i, c in enumerate(cosal):
    U_mot = (1j * c * eta * xop).expm()
    U_full = tensor(qeye(2), U_mot)
    c_ops.append(np.sqrt(G * W[i]) * (s_minus_full * U_full))

# Add sigma_minus decay without recoil (optional, for comparison)
c_ops.append(np.sqrt(gamma) * s_minus_full)
# --- Construct Liouvillians ---
# 1. Pulse Step: Coherent RSB drive, NO dissipation (or minimal?)
# Typically only Unitary part active during pulse
L_pulse = liouvillian(H_pulse, []) 

# 2. Dissipation Step: No Drive, ONLY Dissipation (Optical Pumping)
# H = 0 (or H0), c_ops active
L_diss = liouvillian(H_diss, c_ops)

# --- Propagators ---
# Propagator for Pulse step (duration tau)
prop_pulse = (L_pulse * tau).expm()

# Propagator for Dissipation step (duration tg)
prop_diss = (L_diss * tg).expm()

# Combined Propagator for one cycle (Pulse -> Dissipation)
prop_cycle = prop_diss * prop_pulse 


# --- Simulation ---
num_pulses = 30 # Number of cycles
times_sim = np.arange(num_pulses + 1) # Just integer steps for cycles

# Initial state
thermal_state = (-beta * nu * a_dag * a / kT).expm()
thermal_state = thermal_state / thermal_state.tr() 
ground_state = basis(2, 1) 
initial_state = tensor(ground_state * ground_state.dag(), thermal_state)

# Storage
occupancies = np.zeros((num_pulses + 1, N))
rho_vec = operator_to_vector(initial_state)

# Initial occupancy
rho_mat = vector_to_operator(rho_vec)
occupancies[0, :] = rho_mat.ptrace(1).diag().real

# Evolve step-by-step
current_rho_vec = rho_vec
for k in range(1, num_pulses + 1):
    current_rho_vec = prop_cycle * current_rho_vec
    
    # Store
    rho_mat = vector_to_operator(current_rho_vec)
    occupancies[k, :] = rho_mat.ptrace(1).diag().real


# Plot the occupancy of all motional states over cycles
plt.figure()
for n in range(N):
    plt.plot(times_sim, occupancies[:, n], '.-', label=f'n={n}')
plt.xlabel('Pulse Cycle')
plt.ylabel('Occupancy')
plt.title('Occupancy vs Cooling Cycles')
plt.legend()
plt.show()

# Bar plot
p_motional_initial = occupancies[0, :]
p_motional_final = occupancies[-1, :]
plt.figure()
plt.plot(np.arange(N)-0.2, p_motional_initial, 'o-', label='Initial')
plt.bar(np.arange(N)+0.2, p_motional_final, width=0.4, label='Final', alpha=0.7)
plt.xlabel('Motional State n')
plt.ylabel('Population')
plt.title('State Populations After Cooling Cycles')
plt.legend()
plt.show()

# %% [markdown]
# ### 3. New pulse sequence
# 
# 
# * RF dipolar pulse to the trapped ion
# * Y pulse of pi/2
# * x pulse of length $\tau$
# * undo y pulse
# * Relacation to ground spin state
# * undo displacement RF pulse
# 
# In this frame, $\Omega = \nu$ and $\Delta = 0$

# %% [markdown]
# #### 3.1 Testing each pulse separately
# - RF dipolar pulse:
# $$
# H_{RF} = i\frac{RF\_strength}{2} (a - a^{\dagger})
# $$
# where $RF\_strength$ is the strength of the RF dipolar coupling, set to $\eta$

# %%
import numpy as np
import matplotlib.pyplot as plt
from qutip import *

# Define system parameters
Delta = 0  # Detuning
RF_strength = 1  # RF dipolar coupling strength
nu = 1.0     # Motional frequency
Omega = nu  # Rabi frequency
Omega_y = 1*Omega  # Rabi frequency for Y pulse
eta = 0.2    # Lamb-Dicke parameter
n_thermal = 1  # Average thermal phonon number
n0 = 1
gamma = 1000  # Spin relaxation
beta = 0.01    # Inverse temperature
kT = 1.0      # Boltzmann constant times temperature
tau = 2 * np.pi / (eta * Omega * np.sqrt(n0))  # Pulse duration
print(f"Pulse duration tau: {tau}")
# Define operators
N = 20  # Truncation for the Fock space

# QHO operators
a = destroy(N)  # Annihilation operator
a_dag = a.dag()  # Creation operator
xop = a + a_dag  # Position operator

# Spin operators
sz = sigmaz()
s_plus = sigmap()
s_minus = sigmam()
s_y = sigmay()
s_x = sigmax()

# Full size operators
s_z_full = tensor(sz, qeye(N))
s_plus_full = tensor(s_plus, qeye(N))
s_minus_full = tensor(s_minus, qeye(N))
s_y_full = tensor(s_y, qeye(N))
a_full = tensor(qeye(2), a)
a_dag_full = tensor(qeye(2), a_dag)
s_minus_full = tensor(s_minus, qeye(N))
s_x_full = tensor(s_x, qeye(N))

# Displaced Hamiltonian operator
D_op = tensor(qeye(2), displace(N, 1j * eta))

"""Hamiltonians for pulse and dissipation steps defined above."""

# Hamiltonian (Free evolution for Dissipation)
H_diss = Delta * s_z_full / 2 + nu * a_dag_full * a_full

# --- Pulses for the new sequence ---

# 1) RF dipolar pulse of i*eta/2
H_rf = (RF_strength / 2) * (a_full - a_dag_full)
rf_duration = -1j * eta / RF_strength
L_rf = liouvillian(H_rf, [])
prop_rf = (L_rf * rf_duration).expm()
# undo RF
prop_rf_undo = (-L_rf * rf_duration).expm()

# Test the displacement of this pulse on a vacuum motional state
ground_state = basis(2, 1)
initial_state = tensor(ground_state * ground_state.dag(), basis(N, 0) * basis(N, 0).dag())

print("Initial motional state populations:", initial_state.ptrace(1).diag().real)
print("Expected displacement amplitude:", eta / 2)

initial_state_vec = operator_to_vector(initial_state)
final_state_vec = prop_rf * initial_state_vec

final_state = vector_to_operator(final_state_vec)
print("Final motional state populations after RF pulse:", final_state.ptrace(1).diag().real)
# Calculate the expected displaced state populations for comparison
displaced_state = displace(N, -1j * eta / 2) * basis(N, 0)
expected_populations = np.abs(displaced_state.full().flatten())**2
print("Expected motional state populations after displacement:", expected_populations)


# %% [markdown]
# - Y(π/2) pulse on the spin
# $$
# H_{Y} = \frac{\Delta}{2}\sigma_{z} + \nu aa^{\dagger} + \frac{i\Omega_{y}}{2}\sigma_{y} [\sigma^{+}D(i\eta) - \sigma^{-}D^{\dagger}(i\eta)]
# $$

# %%
import numpy as np
import matplotlib.pyplot as plt
from qutip import *

# Define system parameters
Delta = 0  # Detuning
RF_strength = 1  # RF dipolar coupling strength
nu = 1.0     # Motional frequency
Omega = nu  # Rabi frequency
Omega_y = 1*Omega  # Rabi frequency for Y pulse
eta = 0.2    # Lamb-Dicke parameter
n_thermal = 1  # Average thermal phonon number
n0 = 1
gamma = 1000  # Spin relaxation
beta = 0.01    # Inverse temperature
kT = 1.0      # Boltzmann constant times temperature
tau = 2 * np.pi / (eta * Omega * np.sqrt(n0))  # Pulse duration
print(f"Pulse duration tau: {tau}")
# Define operators
N = 20  # Truncation for the Fock space

# QHO operators
a = destroy(N)  # Annihilation operator
a_dag = a.dag()  # Creation operator
xop = a + a_dag  # Position operator

# Spin operators
sz = sigmaz()
s_plus = sigmap()
s_minus = sigmam()
s_y = sigmay()
s_x = sigmax()

# Full size operators
s_z_full = tensor(sz, qeye(N))
s_plus_full = tensor(s_plus, qeye(N))
s_minus_full = tensor(s_minus, qeye(N))
s_y_full = tensor(s_y, qeye(N))
a_full = tensor(qeye(2), a)
a_dag_full = tensor(qeye(2), a_dag)
s_minus_full = tensor(s_minus, qeye(N))
s_x_full = tensor(s_x, qeye(N))

# Displaced Hamiltonian operator
D_op = tensor(qeye(2), displace(N, 1j * eta))

"""Hamiltonians for pulse and dissipation steps defined above."""

# Hamiltonian (Free evolution for Dissipation)
H_diss = Delta * s_z_full / 2 + nu * a_dag_full * a_full

 
# 2) Y(π/2) pulse on the spin, same as H_pulse but with phase shift on the sigmas
H_y = 0.5 * Delta * s_z_full + nu * a_dag_full * a_full + \
          0.5 * Omega_y * (1j * s_plus_full * D_op - 1j * s_minus_full * D_op.dag())
y_duration = np.pi / (2 *Omega_y)
L_y = liouvillian(H_y, [])
prop_y = (L_y * y_duration).expm()


# Test Y(π/2) pulse on spin-up state
ground_state = basis(2, 0)
initial_state = tensor(ground_state * ground_state.dag(), basis(N, 0) * basis(N, 0).dag())
initial_state_vec = operator_to_vector(initial_state)

print("Initial <s_z>:", expect(s_z_full, initial_state))
print("Initial <s_y>:", expect(s_y_full, initial_state))
print("Initial <s_x>:", expect(s_x_full, initial_state))

final_state_vec = prop_y * initial_state_vec
final_state = vector_to_operator(final_state_vec)

print("Final <s_z> after Y(π/2):", expect(s_z_full, final_state))
print("Final <s_y> after Y(π/2):", expect(s_y_full, final_state))
print("Final <s_x> after Y(π/2):", expect(s_x_full, final_state))

# %% [markdown]
# - X $\tau$ pulse on the spin:
# $$
# H_{X} = \frac{\Delta}{2}\sigma_{z} + \nu aa^{\dagger} + \frac{i\Omega}{2}[\sigma^{+}D(i\eta) - \sigma^{-}D^{\dagger}(i\eta)]
# $$
# which is applied for a time $\tau = \frac{\pi}{\eta \Omega}$, which should ideally produce a displacement of $-i\eta$ in the motional state.

# %%
import numpy as np
import matplotlib.pyplot as plt
from qutip import *

# Define system parameters
Delta = 0  # Detuning
RF_strength = 1  # RF dipolar coupling strength
nu = 1.0     # Motional frequency
Omega = nu  # Rabi frequency
Omega_y = 1*Omega  # Rabi frequency for Y pulse
eta = 0.01    # Lamb-Dicke parameter
n_thermal = 1  # Average thermal phonon number
n0 = 1
gamma = 1000  # Spin relaxation
beta = 0.01    # Inverse temperature
kT = 1.0      # Boltzmann constant times temperature
tau = 2 * np.pi / (eta * Omega * np.sqrt(n0))  # Pulse duration
print(f"Pulse duration tau: {tau}")
# Define operators
N = 20  # Truncation for the Fock space

# QHO operators
a = destroy(N)  # Annihilation operator
a_dag = a.dag()  # Creation operator
xop = a + a_dag  # Position operator

# Spin operators
sz = sigmaz()
s_plus = sigmap()
s_minus = sigmam()
s_y = sigmay()
s_x = sigmax()

# Full size operators
s_z_full = tensor(sz, qeye(N))
s_plus_full = tensor(s_plus, qeye(N))
s_minus_full = tensor(s_minus, qeye(N))
s_y_full = tensor(s_y, qeye(N))
a_full = tensor(qeye(2), a)
a_dag_full = tensor(qeye(2), a_dag)
s_minus_full = tensor(s_minus, qeye(N))
s_x_full = tensor(s_x, qeye(N))

# Displaced Hamiltonian operator
D_op = tensor(qeye(2), displace(N, 1j * eta))

"""Hamiltonians for pulse and dissipation steps defined above."""

# Hamiltonian (Free evolution for Dissipation)
H_diss = Delta * s_z_full / 2 + nu * a_dag_full * a_full
 
 
# 3) X(τ) pulse on the spin, same as H_pulse but with different phase
H_x = 0.5 * Delta * s_z_full + nu * a_dag_full * a_full + \
          0.5 * Omega * (s_plus_full * D_op + s_minus_full * D_op.dag())
L_x = liouvillian(H_x, [])
prop_x = (L_x * tau).expm()

# Test the X pulse on spin-up state
ground_state = basis(2, 0)
initial_state = tensor(ground_state * ground_state.dag(), basis(N, 0) * basis(N, 0).dag())
initial_state_vec = operator_to_vector(initial_state)

print("Initial <s_z>:", expect(s_z_full, initial_state))
print("Initial <s_y>:", expect(s_y_full, initial_state))
print("Initial <s_x>:", expect(s_x_full, initial_state))
final_state_vec = prop_x * initial_state_vec
final_state = vector_to_operator(final_state_vec)
print("Final <s_z> after X(τ):", expect(s_z_full, final_state))
print("Final <s_y> after X(τ):", expect(s_y_full, final_state))
print("Final <s_x> after X(τ):", expect(s_x_full, final_state))

# %% [markdown]
# #### 3.2 Full sequence simulation

# %%
import numpy as np
import matplotlib.pyplot as plt
from qutip import *

# Define system parameters
Delta = 0  # Detuning
RF_strength = 1  # RF dipolar coupling strength
nu = 1.0     # Motional frequency
Omega = nu  # Rabi frequency
Omega_y = 1000  # Rabi frequency for Y pulse
eta = 0.2   # Lamb-Dicke parameter
n_thermal = 1  # Average thermal phonon number
n0 = 1
gamma = 1000  # Spin relaxation
kT = 100      # Boltzmann constant times temperature
tau =  2*np.pi / (eta * Omega * np.sqrt(n0))  # Pulse duration
print(f"Pulse duration tau: {tau}")
# Define operators
N = 20  # Truncation for the Fock space

# QHO operators
a = destroy(N)  # Annihilation operator
a_dag = a.dag()  # Creation operator
xop = a + a_dag  # Position operator

# Spin operators
sz = sigmaz()
s_plus = sigmap()
s_minus = sigmam()
s_y = sigmay()
s_x = sigmax()

# Full size operators
s_z_full = tensor(sz, qeye(N))
s_plus_full = tensor(s_plus, qeye(N))
s_minus_full = tensor(s_minus, qeye(N))
s_y_full = tensor(s_y, qeye(N))
a_full = tensor(qeye(2), a)
a_dag_full = tensor(qeye(2), a_dag)
s_minus_full = tensor(s_minus, qeye(N))
s_x_full = tensor(s_x, qeye(N))

# Displaced Hamiltonian operator
D_op = tensor(qeye(2), displace(N,1j*eta))

"""Hamiltonians for pulse and dissipation steps defined above."""

# Full Hamiltonian (Eq. 1) – not directly used in the new sequence,
## but kept for reference
# H_pulse = 0.5 * Delta * s_z_full + nu * a_dag_full * a_full + \
#           0.5 * Omega * (s_plus_full * D_op + s_minus_full * D_op.dag())

# Keep for reference
# H_pulse = nu * a_dag_full * a_full + 1j * 0.5 * eta * Omega * s_x_full * (a_full - a_dag_full)  + Omega/2 * s_z_full

# Hamiltonian (Free evolution for Dissipation)
H_diss = Delta * s_z_full / 2 + nu * a_dag_full * a_full

# --- Pulses for the new sequence ---

# 1) RF dipolar pulse of i*eta/2
H_rf = (RF_strength / 2) * (a_full + a_dag_full)
rf_duration = eta / RF_strength
L_rf = liouvillian(H_rf, [])
prop_rf = (L_rf * rf_duration).expm()
# undo RF
prop_rf_undo = (-L_rf * rf_duration).expm()
 
# 2) Y(π/2) pulse on the spin, same as H_pulse but with phase shift on the sigmas
#H_y = 0.5 * Delta * s_z_full + nu * a_dag_full * a_full + \
#          0.5 * Omega_y *(-1j * s_plus_full * D_op + 1j * s_minus_full * D_op.dag())
H_y = 0.5 * Omega_y *s_y_full
y_duration = np.pi / (2 *Omega_y)
L_y = liouvillian(H_y, [])
prop_y = (L_y * y_duration).expm()
# Undo Y(π/2)
L_y_undo = liouvillian(-H_y, [])
prop_y_undo = (L_y_undo * y_duration).expm()
 
 
# 3) X(τ) pulse on the spin, same as H_pulse but with different phase
#H_x = 0.5 * Delta * s_z_full + nu * a_dag_full * a_full + \
#          0.5 * Omega * (s_plus_full * D_op + s_minus_full * D_op.dag())
H_x = 0.5 * nu * s_x_full + nu * a_dag_full * a_full + \
          0.5 * 1j * Omega * eta * s_z_full * (a_dag_full - a_full)
L_x = liouvillian(H_x, [])
prop_x = (L_x * tau).expm()

# --- Collapse Operators for Dissipation with recoil ---
cosmax = 100
cosal = np.arange(-cosmax, cosmax + 1) / cosmax
W = 3 * (cosal**2 + 1) / 4 * 1 / (2 * cosmax)
W = W / np.sum(W)
G = gamma / 2  # This is Γ/2 in the master equation
tg = 2 / G  # Dissipation duration adjusted for rates
c_ops = []
for i, c in enumerate(cosal):
    U_mot = (1j * c * eta * xop).expm()
    U_full = tensor(qeye(2), U_mot)
    c_ops.append(np.sqrt(G * W[i]) * (s_minus_full * U_full))

# Add sigma_minus decay without recoil (optional, for comparison)
c_ops.append(np.sqrt(gamma) * s_minus_full)

# --- Liouvillian and propagator for dissipation step ---
L_diss = liouvillian(H_diss, c_ops)
prop_diss = (L_diss * tg).expm()

# Combined Propagator for one cooling cycle:
## RF dipolar pulse -> Y(π/2) -> X(τ) -> undo Y -> relaxation -> undo RF
prop_cycle = prop_diss * prop_rf_undo * prop_y_undo * prop_x * prop_y * prop_rf
#prop_cycle = prop_rf_undo * prop_diss * prop_y_undo * prop_x * prop_y * prop_rf
# prop_cycle = prop_diss * prop_y_undo * prop_x * prop_y 

# Test whether D_op.dag is equivalent to H_rf
# L_D_op = liouvillian(D_op.dag(), [])
# prop_D_op_full = (L_D_op ).expm()
# prop_undo_D_op_full = ( L_D_op ).expm()
# prop_cycle =  prop_diss * prop_y_undo * prop_x * prop_y * prop_D_op_full.dag()

# # Now test with a ground state initial condition
# mot_vac = basis(N, 0)
# spin_g = basis(2, 0)
# psi_test = tensor(spin_g, mot_vac)

# # Convert state to vector form for Liouvillian propagator
# psi_test_vec = operator_to_vector(psi_test * psi_test.dag())

# psi_D_vec  = prop_cycle_test * psi_test_vec
# psi_normal_vec = prop_cycle * psi_test_vec

# # Convert back to operator form
# psi_D = vector_to_operator(psi_D_vec)
# psi_normal = vector_to_operator(psi_normal_vec)

# # Verify equivalence
# diff_norm = (prop_cycle_test - prop_cycle).norm()
# print(f"Norm difference between both implementations of the cycle propagator: {diff_norm}")

# overlap = (psi_D * psi_normal).tr()
# print(f"Overlap between both implementations on ground state: {overlap}")


# --- Simulation ---
num_pulses = 60  # Number of cycles
times_sim = np.arange(num_pulses + 1)  # Integer steps for cycles

# Initial state: spin in ground, motion thermal
thermal_state = (- nu * a_dag * a / kT).expm()
thermal_state = thermal_state / thermal_state.tr()
ground_state = basis(2, 1)
initial_state = tensor(ground_state * ground_state.dag(), thermal_state)

# Storage
occupancies = np.zeros((num_pulses + 1, N))
rho_vec = operator_to_vector(initial_state)

# Initial occupancy
rho_mat = vector_to_operator(rho_vec)
occupancies[0, :] = rho_mat.ptrace(1).diag().real

# Evolve step-by-step under the new pulse sequence
current_rho_vec = rho_vec
for k in range(1, num_pulses + 1):
    current_rho_vec = prop_cycle * current_rho_vec

    # Store
    rho_mat = vector_to_operator(current_rho_vec)
    occupancies[k, :] = rho_mat.ptrace(1).diag().real


# Plot the occupancy of all motional states over cycles
plt.figure()
for n in range(N):
    plt.plot(times_sim, occupancies[:, n], '.-', label=f'n={n}')
plt.xlabel('Pulse Cycle')
plt.ylabel('Occupancy')
plt.title('Occupancy vs Cooling Cycles (New Pulse Sequence)')
plt.legend()
plt.show()

# Bar plot
p_motional_initial = occupancies[0, :]
p_motional_final = occupancies[-1, :]
plt.figure()
plt.plot(np.arange(N), p_motional_initial, 'o-', label='Initial' , color='red')
plt.bar(np.arange(N), p_motional_final, width=0.4, label='Final', alpha=0.7)
plt.xlabel('Motional State n')
plt.ylabel('Population')
plt.title('State Populations After Cooling Cycles (New Pulse Sequence)')
plt.legend()
plt.show()

# %% [markdown]
# ### Paper figure 1

# %%
import numpy as np
import matplotlib.pyplot as plt
from qutip import *

# Define system parameters
Delta = 0
RF_strength = 1
nu = 1.0
Omega = nu
Omega_y = 1000
n_thermal = 1
n0 = 1
gamma = 1000
kT = 100
N = 14
num_pulses = 60

eta_values = [0.02, 0.05, 0.08, 0.1]

fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharey=True, sharex=True)
axes = axes.flatten()

for idx, eta in enumerate(eta_values):
    tau = 2 * np.pi / (eta * Omega * np.sqrt(n0))

    # QHO operators
    a = destroy(N)
    a_dag = a.dag()
    xop = a + a_dag

    # Spin operators
    sz = sigmaz()
    s_plus = sigmap()
    s_minus = sigmam()
    s_y = sigmay()
    s_x = sigmax()

    # Full size operators
    s_z_full = tensor(sz, qeye(N))
    s_plus_full = tensor(s_plus, qeye(N))
    s_minus_full = tensor(s_minus, qeye(N))
    s_y_full = tensor(s_y, qeye(N))
    a_full = tensor(qeye(2), a)
    a_dag_full = tensor(qeye(2), a_dag)
    s_x_full = tensor(s_x, qeye(N))

    D_op = tensor(qeye(2), displace(N, 1j * eta))

    H_diss = Delta * s_z_full / 2 + nu * a_dag_full * a_full

    # 1) RF dipolar pulse
    H_rf = (RF_strength / 2) * (a_full + a_dag_full)
    rf_duration = eta / RF_strength
    L_rf = liouvillian(H_rf, [])
    prop_rf = (L_rf * rf_duration).expm()
    prop_rf_undo = (-L_rf * rf_duration).expm()

    # 2) Y(π/2) pulse
    # H_y = 0.5 * Omega_y * s_y_full
    H_y = 0.5 * Delta * s_z_full + nu * a_dag_full * a_full + \
          0.5 * Omega_y *(-1j * s_plus_full * D_op + 1j * s_minus_full * D_op.dag())
    y_duration = np.pi / (2 * Omega_y)
    L_y = liouvillian(H_y, [])
    prop_y = (L_y * y_duration).expm()
    L_y_undo = liouvillian(-H_y, [])
    prop_y_undo = (L_y_undo * y_duration).expm()

    # 3) X(τ) pulse
    H_x = 0.5 * nu * s_x_full + nu * a_dag_full * a_full + \
              0.5 * 1j * Omega * eta * s_z_full * (a_dag_full - a_full)
    L_x = liouvillian(H_x, [])
    prop_x = (L_x * tau).expm()

    # Collapse operators with recoil
    cosmax = 100
    cosal = np.arange(-cosmax, cosmax + 1) / cosmax
    W = 3 * (cosal**2 + 1) / 4 * 1 / (2 * cosmax)
    W = W / np.sum(W)
    G = gamma / 2
    tg = 2 / G
    c_ops = []
    for i, c in enumerate(cosal):
        U_mot = (1j * c * eta * xop).expm()
        U_full = tensor(qeye(2), U_mot)
        c_ops.append(np.sqrt(G * W[i]) * (s_minus_full * U_full))
    c_ops.append(np.sqrt(gamma) * s_minus_full)

    # Propagators
    L_diss = liouvillian(H_diss, c_ops)
    prop_diss = (L_diss * tg).expm()
    prop_cycle = prop_diss * prop_rf_undo * prop_y_undo * prop_x * prop_y * prop_rf

    # Initial state
    thermal_state = (-nu * a_dag * a / kT).expm()
    thermal_state = thermal_state / thermal_state.tr()
    ground_state = basis(2, 1)
    initial_state = tensor(ground_state * ground_state.dag(), thermal_state)

    occupancies = np.zeros((num_pulses + 1, N))
    rho_vec = operator_to_vector(initial_state)
    rho_mat = vector_to_operator(rho_vec)
    occupancies[0, :] = rho_mat.ptrace(1).diag().real

    current_rho_vec = rho_vec
    for k in range(1, num_pulses + 1):
        current_rho_vec = prop_cycle * current_rho_vec
        rho_mat = vector_to_operator(current_rho_vec)
        occupancies[k, :] = rho_mat.ptrace(1).diag().real

    # Plot
    ax = axes[idx]
    p_initial = occupancies[0, :]
    p_final = occupancies[-1, :]
    n_states = np.arange(N)
    ax.plot(n_states, p_initial, 'o-', color='tab:red', markersize=4, label='Initial')
    ax.bar(n_states, p_final, width=0.5, alpha=0.7, color='tab:blue', label='Final')
    ax.set_title(rf'$\eta = {eta}$', fontsize=14)
    ax.set_xlim(-0.5, 14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    print(f"eta={eta} done — n_bar initial={np.dot(n_states, p_initial):.3f}, final={np.dot(n_states, p_final):.3f}")

fig.supxlabel('Occupation number $n$', fontsize=14)
fig.supylabel('Population', fontsize=14)
# fig.suptitle('Cooling in the Lamb-Dicke Regime', fontsize=16, y=1.01)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Figure 2

# %%
import numpy as np
import matplotlib.pyplot as plt
from qutip import *

# Define system parameters
Delta = 0
RF_strength = 1
nu = 1.0
eta = 0.05
Omega_y = 1000
n_thermal = 1
n0 = 1
gamma = 1000
kT = 100
N = 14
num_pulses = 60

Omega_values = [0.5, 1, 4, 8]

fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharey=True, sharex=True)
axes = axes.flatten()

for idx, Omega in enumerate(Omega_values):
    tau = 2 * np.pi / (eta * Omega * np.sqrt(n0))

    # QHO operators
    a = destroy(N)
    a_dag = a.dag()
    xop = a + a_dag

    # Spin operators
    sz = sigmaz()
    s_plus = sigmap()
    s_minus = sigmam()
    s_y = sigmay()
    s_x = sigmax()

    # Full size operators
    s_z_full = tensor(sz, qeye(N))
    s_plus_full = tensor(s_plus, qeye(N))
    s_minus_full = tensor(s_minus, qeye(N))
    s_y_full = tensor(s_y, qeye(N))
    a_full = tensor(qeye(2), a)
    a_dag_full = tensor(qeye(2), a_dag)
    s_x_full = tensor(s_x, qeye(N))

    D_op = tensor(qeye(2), displace(N, 1j * eta))

    H_diss = Delta * s_z_full / 2 + nu * a_dag_full * a_full

    # 1) RF dipolar pulse
    H_rf = (RF_strength / 2) * (a_full + a_dag_full)
    rf_duration = eta / RF_strength
    L_rf = liouvillian(H_rf, [])
    prop_rf = (L_rf * rf_duration).expm()
    prop_rf_undo = (-L_rf * rf_duration).expm()

    # 2) Y(π/2) pulse
    # H_y = 0.5 * Omega_y * s_y_full
    H_y = 0.5 * Delta * s_z_full + nu * a_dag_full * a_full + \
          0.5 * Omega_y *(-1j * s_plus_full * D_op + 1j * s_minus_full * D_op.dag())
    y_duration = np.pi / (2 * Omega_y)
    L_y = liouvillian(H_y, [])
    prop_y = (L_y * y_duration).expm()
    L_y_undo = liouvillian(-H_y, [])
    prop_y_undo = (L_y_undo * y_duration).expm()

    # 3) X(τ) pulse
    H_x = 0.5 * nu * s_x_full + nu * a_dag_full * a_full + \
              0.5 * 1j * Omega * eta * s_z_full * (a_dag_full - a_full)
    L_x = liouvillian(H_x, [])
    prop_x = (L_x * tau).expm()

    # Collapse operators with recoil
    cosmax = 100
    cosal = np.arange(-cosmax, cosmax + 1) / cosmax
    W = 3 * (cosal**2 + 1) / 4 * 1 / (2 * cosmax)
    W = W / np.sum(W)
    G = gamma / 2
    tg = 2 / G
    c_ops = []
    for i, c in enumerate(cosal):
        U_mot = (1j * c * eta * xop).expm()
        U_full = tensor(qeye(2), U_mot)
        c_ops.append(np.sqrt(G * W[i]) * (s_minus_full * U_full))
    c_ops.append(np.sqrt(gamma) * s_minus_full)

    # Propagators
    L_diss = liouvillian(H_diss, c_ops)
    prop_diss = (L_diss * tg).expm()
    prop_cycle = prop_diss * prop_rf_undo * prop_y_undo * prop_x * prop_y * prop_rf

    # Initial state
    thermal_state = (-nu * a_dag * a / kT).expm()
    thermal_state = thermal_state / thermal_state.tr()
    ground_state = basis(2, 1)
    initial_state = tensor(ground_state * ground_state.dag(), thermal_state)

    occupancies = np.zeros((num_pulses + 1, N))
    rho_vec = operator_to_vector(initial_state)
    rho_mat = vector_to_operator(rho_vec)
    occupancies[0, :] = rho_mat.ptrace(1).diag().real

    current_rho_vec = rho_vec
    for k in range(1, num_pulses + 1):
        current_rho_vec = prop_cycle * current_rho_vec
        rho_mat = vector_to_operator(current_rho_vec)
        occupancies[k, :] = rho_mat.ptrace(1).diag().real

    # Plot
    ax = axes[idx]
    p_initial = occupancies[0, :]
    p_final = occupancies[-1, :]
    n_states = np.arange(N)
    ax.plot(n_states, p_initial, 'o-', color='tab:red', markersize=4, label='Initial')
    ax.bar(n_states, p_final, width=0.5, alpha=0.7, color='tab:blue', label='Final')
    ax.set_title(rf'$\Omega = {Omega}\,\nu$', fontsize=14)
    ax.set_xlim(-0.5, 14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    print(f"Omega={Omega} done — n_bar initial={np.dot(n_states, p_initial):.3f}, final={np.dot(n_states, p_final):.3f}")

fig.supxlabel('Occupation number $n$', fontsize=14)
fig.supylabel('Population', fontsize=14)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Figure 3

# %%
import numpy as np
import matplotlib.pyplot as plt
from qutip import *

# Define system parameters
Delta = 0
RF_strength = 1
nu = 1.0
eta = 0.05
Omega_y = 1000
n_thermal = 1
n0 = 1
gamma = 1000
kT = 100
N = 14
num_pulses = 60

Omega_values = [1, 8]

fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

for idx, Omega in enumerate(Omega_values):
    tau = 2 * np.pi / (eta * Omega * np.sqrt(n0))

    # QHO operators
    a = destroy(N)
    a_dag = a.dag()
    xop = a + a_dag

    # Spin operators
    sz = sigmaz()
    s_plus = sigmap()
    s_minus = sigmam()
    s_y = sigmay()
    s_x = sigmax()

    # Full size operators
    s_z_full = tensor(sz, qeye(N))
    s_plus_full = tensor(s_plus, qeye(N))
    s_minus_full = tensor(s_minus, qeye(N))
    s_y_full = tensor(s_y, qeye(N))
    a_full = tensor(qeye(2), a)
    a_dag_full = tensor(qeye(2), a_dag)
    s_x_full = tensor(s_x, qeye(N))

    D_op = tensor(qeye(2), displace(N, 1j * eta))

    H_diss = Delta * s_z_full / 2 + nu * a_dag_full * a_full

    # 1) RF dipolar pulse
    H_rf = (RF_strength / 2) * (a_full + a_dag_full)
    rf_duration = eta / RF_strength
    L_rf = liouvillian(H_rf, [])
    prop_rf = (L_rf * rf_duration).expm()
    prop_rf_undo = (-L_rf * rf_duration).expm()

    # 2) Y(π/2) pulse
    # H_y = 0.5 * Omega_y * s_y_full
    H_y = 0.5 * Delta * s_z_full + nu * a_dag_full * a_full + \
          0.5 * Omega_y *(-1j * s_plus_full * D_op + 1j * s_minus_full * D_op.dag())
    y_duration = np.pi / (2 * Omega_y)
    L_y = liouvillian(H_y, [])
    prop_y = (L_y * y_duration).expm()
    L_y_undo = liouvillian(-H_y, [])
    prop_y_undo = (L_y_undo * y_duration).expm()

    # 3) X(τ) pulse
    H_x = 0.5 * nu * s_x_full + nu * a_dag_full * a_full + \
              0.5 * 1j * Omega * eta * s_z_full * (a_dag_full - a_full)
    L_x = liouvillian(H_x, [])
    prop_x = (L_x * tau).expm()

    # Collapse operators with recoil
    cosmax = 100
    cosal = np.arange(-cosmax, cosmax + 1) / cosmax
    W = 3 * (cosal**2 + 1) / 4 * 1 / (2 * cosmax)
    W = W / np.sum(W)
    G = gamma / 2
    tg = 2 / G
    c_ops = []
    for i, c in enumerate(cosal):
        U_mot = (1j * c * eta * xop).expm()
        U_full = tensor(qeye(2), U_mot)
        c_ops.append(np.sqrt(G * W[i]) * (s_minus_full * U_full))
    c_ops.append(np.sqrt(gamma) * s_minus_full)

    # Propagators
    L_diss = liouvillian(H_diss, c_ops)
    prop_diss = (L_diss * tg).expm()
    prop_cycle = prop_diss * prop_rf_undo * prop_y_undo * prop_x * prop_y * prop_rf

    # Initial state
    thermal_state = (-nu * a_dag * a / kT).expm()
    thermal_state = thermal_state / thermal_state.tr()
    ground_state = basis(2, 1)
    initial_state = tensor(ground_state * ground_state.dag(), thermal_state)

    occupancies = np.zeros((num_pulses + 1, N))
    rho_vec = operator_to_vector(initial_state)
    rho_mat = vector_to_operator(rho_vec)
    occupancies[0, :] = rho_mat.ptrace(1).diag().real

    current_rho_vec = rho_vec
    for k in range(1, num_pulses + 1):
        current_rho_vec = prop_cycle * current_rho_vec
        rho_mat = vector_to_operator(current_rho_vec)
        occupancies[k, :] = rho_mat.ptrace(1).diag().real

    # Plot populations vs pulse cycle
    ax = axes[idx]
    times_sim = np.arange(num_pulses + 1)
    for n in range(N):
        ax.plot(times_sim, occupancies[:, n], '.-', label=f'n={n}')
    ax.set_title(rf'$\Omega = {Omega}\,\nu$', fontsize=14)
    ax.set_xlabel('Pulse Cycle', fontsize=12)
    ax.legend(fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)

    print(f"Omega={Omega} done — n_bar final={np.dot(np.arange(N), occupancies[-1, :]):.3f}")

axes[0].set_ylabel('Population', fontsize=12)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Figure 4

# %%
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from qutip import *

# ── Shared parameters ──────────────────────────────────────────────
Delta_val = 0
RF_strength = 1
nu = 1.0
Omega_y = 1000
n_thermal = 1
n0 = 1
gamma = 1000
kT = 1
beta = 0.01    # Inverse temperature
N = 14
num_pulses = 30
eta = 0.05
n_show = 14  # Only show first n_show Fock states for clarity


# ── Helper: build operators ────────────────────────────────────────
def build_operators(N, eta):
    a = destroy(N); a_dag = a.dag(); xop = a + a_dag
    sz = sigmaz(); s_plus = sigmap(); s_minus = sigmam()
    s_y = sigmay(); s_x = sigmax()
    s_z_full     = tensor(sz, qeye(N))
    s_plus_full  = tensor(s_plus, qeye(N))
    s_minus_full = tensor(s_minus, qeye(N))
    s_y_full     = tensor(s_y, qeye(N))
    s_x_full     = tensor(s_x, qeye(N))
    a_full       = tensor(qeye(2), a)
    a_dag_full   = tensor(qeye(2), a_dag)
    D_op         = tensor(qeye(2), displace(N, 1j * eta))
    return (a, a_dag, xop,
            s_z_full, s_plus_full, s_minus_full,
            s_y_full, s_x_full,
            a_full, a_dag_full, D_op)


# ── Helper: build collapse operators ───────────────────────────────
def build_c_ops(eta, xop, s_minus_full, gamma, N):
    cosmax = 100
    cosal = np.arange(-cosmax, cosmax + 1) / cosmax
    W = 3 * (cosal**2 + 1) / 4 / (2 * cosmax)
    W = W / W.sum()
    G = gamma / 2
    c_ops = []
    for i, c in enumerate(cosal):
        U_mot = (1j * c * eta * xop).expm()
        U_full = tensor(qeye(2), U_mot)
        c_ops.append(np.sqrt(G * W[i]) * (s_minus_full * U_full))
    c_ops.append(np.sqrt(gamma) * s_minus_full)
    return c_ops


# ── Old protocol propagator ───────────────────────────────────────
def old_protocol_propagator(Omega, eta, N, gamma, nu, kT):
    (a, a_dag, xop, s_z_full, s_plus_full, s_minus_full,
     s_y_full, s_x_full, a_full, a_dag_full, D_op) = build_operators(N, eta)
    Delta = 1
    tau = 2 * np.pi / (eta * Omega * np.sqrt(n0))
    H_pulse = 0.5 * Delta * s_z_full + nu * a_dag_full * a_full + \
            0.5 * Omega * (s_plus_full * D_op + s_minus_full * D_op.dag())
    H_diss = Delta * s_z_full / 2 + nu * a_dag_full * a_full

    c_ops = build_c_ops(eta, xop, s_minus_full, gamma, N)
    G = gamma / 2; tg = 2 / G
    prop_pulse = (liouvillian(H_pulse, []) * tau).expm()
    prop_diss  = (liouvillian(H_diss, c_ops) * tg).expm()
    return prop_diss * prop_pulse


# ── New protocol propagator ───────────────────────────────────────
def new_protocol_propagator(Omega, eta, N, gamma, nu, Delta, kT, Omega_y):
    (a, a_dag, xop, s_z_full, s_plus_full, s_minus_full,
     s_y_full, s_x_full, a_full, a_dag_full, D_op) = build_operators(N, eta)
    tau = 2 * np.pi / (eta * Omega * np.sqrt(n0))
    H_diss = Delta * s_z_full / 2 + nu * a_dag_full * a_full

    H_rf = (RF_strength / 2) * (a_full + a_dag_full)
    rf_duration = eta / RF_strength
    L_rf = liouvillian(H_rf, [])
    prop_rf      = (L_rf * rf_duration).expm()
    prop_rf_undo = (-L_rf * rf_duration).expm()

    H_y = (0.5 * Delta * s_z_full + nu * a_dag_full * a_full +
           0.5 * Omega_y * (-1j * s_plus_full * D_op + 1j * s_minus_full * D_op.dag()))
    y_duration = np.pi / (2 * Omega_y)
    prop_y      = (liouvillian(H_y, []) * y_duration).expm()
    prop_y_undo = (liouvillian(-H_y, []) * y_duration).expm()

    H_x = (0.5 * nu * s_x_full + nu * a_dag_full * a_full +
           0.5 * 1j * Omega * eta * s_z_full * (a_dag_full - a_full))
    prop_x = (liouvillian(H_x, []) * tau).expm()

    c_ops = build_c_ops(eta, xop, s_minus_full, gamma, N)
    G = gamma / 2; tg = 2 / G
    prop_diss = (liouvillian(H_diss, c_ops) * tg).expm()
    return prop_diss * prop_rf_undo * prop_y_undo * prop_x * prop_y * prop_rf


# ── Run cooling simulation ────────────────────────────────────────
def run_trapping(prop_cycle, N, nu, kT, beta, num_pulses):
    a = destroy(N); a_dag = a.dag()
    thermal_state = (-beta * nu * a_dag * a / kT).expm()
    thermal_state = thermal_state / thermal_state.tr()
    ground_state  = basis(2, 1)
    initial_state = tensor(ground_state * ground_state.dag(), thermal_state)
    rho_vec = operator_to_vector(initial_state)
    current = rho_vec
    for k in range(num_pulses):
        current = prop_cycle * current
    final_pops = vector_to_operator(current).ptrace(1).diag().real
    return np.array(final_pops).flatten()


# ── Collect data for Panel A ──────────────────────────────────────
Omega_values_A = [0.001, 0.5, 1, 2]
results_old_A = {}
results_new_A = {}

for Om in Omega_values_A:
    print(f"Panel A: Ω = {Om} ...")
    prop_old = old_protocol_propagator(Om, eta, N, gamma, nu, kT)
    prop_new = new_protocol_propagator(Om, eta, N, gamma, nu, Delta_val, kT, Omega_y)
    results_old_A[Om] = run_trapping(prop_old, N, nu, kT, beta, num_pulses)
    results_new_A[Om] = run_trapping(prop_new, N, nu, kT, beta, num_pulses)
    print(f"  Old n̄={np.dot(np.arange(N), results_old_A[Om]):.4f},  "
          f"New n̄={np.dot(np.arange(N), results_new_A[Om]):.4f}")


# ── Collect data for Panel B ──────────────────────────────────────
Omega_values_B = [1, 2, 3, 4]
results_new_B = {}

for Om in Omega_values_B:
    print(f"Panel B: Ω = {Om} ...")
    prop_new = new_protocol_propagator(Om, eta, N, gamma, nu, Delta_val, kT, Omega_y)
    results_new_B[Om] = run_cooling(prop_new, N, nu, kT, num_pulses)
    print(f"  New n̄={np.dot(np.arange(N), results_new_B[Om]):.4f}")


# ╔══════════════════════════════════════════════════════════════════╗
# ║  3D Bar Figure — 2D slices separated along the Omega axis       ║
# ╚══════════════════════════════════════════════════════════════════╝

fig = plt.figure(figsize=(18, 7))

bar_width = 0.8   # width along fock-state axis
bar_depth = 0.02  # very thin along Omega axis → looks 2D
spacing = 1.0     # spacing between Omega slices
n_fock = np.arange(n_show)

# ── Panel A: Old (blue) vs New (orange) superimposed ──────────────
ax1 = fig.add_subplot(121, projection='3d')

for slice_idx, Om in enumerate(Omega_values_A):
    x_centre = slice_idx * spacing
    z_old = results_old_A[Om][:n_show]
    z_new = results_new_A[Om][:n_show]

    # Draw old (blue) first so it sits behind
    ax1.bar3d(np.full(n_show, x_centre - bar_depth / 2),
              n_fock - bar_width / 2, np.zeros(n_show),
              bar_depth, bar_width, z_old,
              color='tab:blue', alpha=0.85, edgecolor='k', linewidth=0.4,
              label='Old' if slice_idx == 0 else '', zsort='average')
    # Draw new (orange) on top
    ax1.bar3d(np.full(n_show, x_centre - bar_depth / 2),
              n_fock - bar_width / 2, np.zeros(n_show),
              bar_depth, bar_width, z_new,
              color='tab:orange', alpha=0.7, edgecolor='k', linewidth=0.4,
              label='New' if slice_idx == 0 else '', zsort='average')

ax1.set_xticks([i * spacing for i in range(len(Omega_values_A))])
ax1.set_xticklabels([f'{Om}' for Om in Omega_values_A], fontsize=10)
ax1.set_xlabel(r'$\Omega\;[\nu]$', fontsize=13, labelpad=12)
ax1.set_ylabel('Fock state $n$', fontsize=13, labelpad=10)
ax1.set_zlabel('Population', fontsize=13, labelpad=8)
ax1.set_title('A) Old vs New protocol', fontsize=14, fontweight='bold', pad=12)
ax1.view_init(elev=20, azim=-60)
ax1.legend(loc='upper right', fontsize=10)

# ── Panel B: New protocol only at higher Omega ────────────────────
ax2 = fig.add_subplot(122, projection='3d')

for slice_idx, Om in enumerate(Omega_values_B):
    x_centre = slice_idx * spacing
    z_new = results_new_B[Om][:n_show]

    ax2.bar3d(np.full(n_show, x_centre - bar_depth / 2),
              n_fock - bar_width / 2, np.zeros(n_show),
              bar_depth, bar_width, z_new,
              color='tab:orange', alpha=0.8, edgecolor='k', linewidth=0.4,
              label='New' if slice_idx == 0 else '', zsort='average')

ax2.set_xticks([i * spacing for i in range(len(Omega_values_B))])
ax2.set_xticklabels([f'{Om}' for Om in Omega_values_B], fontsize=10)
ax2.set_xlabel(r'$\Omega\;[\nu]$', fontsize=13, labelpad=12)
ax2.set_ylabel('Fock state $n$', fontsize=13, labelpad=10)
ax2.set_zlabel('Population', fontsize=13, labelpad=8)
ax2.set_title('B) New protocol — higher $\\Omega$', fontsize=14, fontweight='bold', pad=12)
ax2.view_init(elev=20, azim=-60)
ax2.legend(loc='upper right', fontsize=10)

plt.tight_layout()
plt.savefig('figure4_3d_comparison.png', dpi=200, bbox_inches='tight')
plt.show()


