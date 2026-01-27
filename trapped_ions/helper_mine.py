import numpy as np
import matplotlib.pyplot as plt
from qutip import *

# --- 1. Define Parameters (Strong Coupling Regime) ---
# For superfast cooling, we need the Rabi frequency to be large
nu = 1.0     # Trap frequency
Omega_fast = 1000 * nu  # Strong coupling: Omega >> nu
eta = 0.2    # Lamb-Dicke parameter
gamma = 1  # Decay rate
n_th = 1     # Initial thermal occupancy

# Trotter parameters
dt = 0.05 / (eta * Omega_fast)  # Short time step for the pulses
n_trotter = 180                  # Number of Trotter steps per cooling cycle

# Operators
N = 20
a = destroy(N)
a_dag = a.dag()
x_op = a + a_dag          # Position operator (dimensionless)
p_op = 1j * (a_dag - a)   # Momentum operator (dimensionless)

sz = sigmaz()
sx = sigmax()
sy = sigmay()
sp = sigmap()
sm = sigmam()

# Full space operators
a_full = tensor(qeye(2), a)
a_dag_full = tensor(qeye(2), a_dag)
x_full = tensor(qeye(2), x_op)
sx_full = tensor(sx, qeye(N))
sy_full = tensor(sy, qeye(N))
sz_full = tensor(sz, qeye(N))
sp_full = tensor(sp, qeye(N))
sm_full = tensor(sm, qeye(N))

# Displaced Hamiltonian operator
D_op = tensor(qeye(2), displace(N,1j*eta/2))

# --- 2. Define Hamiltonians (No RWA) ---
# Free evolution (Trap + Spin)
H_free = nu * a_dag_full * a_full

# Laser Interactions: H ~ Omega * (a + a^dag) * sigma
# We need an X-type interaction and a Y-type interaction
# H_int_X = eta * Omega_fast * sx_full * x_full
# H_int_Y = eta * Omega_fast * sy_full * x_full

H_int_X = nu*a_dag_full*a_full + 0.5 * Omega_fast * (sp_full * D_op + sm_full * D_op.dag())
H_int_Y =  1j * 0.5 * Omega_fast *(sp_full * D_op - sm_full * D_op.dag())

# --- 3. Build Propagators (The Pulse Sequence) ---

# A. The Native X Pulse
# Simply apply the laser with X-phase for duration dt
U_X = (-1j * H_int_X * dt).expm()

# B. The Synthesized P Pulse [cite: 49]
# We want an interaction ~ P * sigma_y
# We create this by: Kick(Y) -> Wait(Quarter Period) -> UnKick(Y)
# Note: The wait rotates X into P.
t_quarter = (2 * np.pi / nu) / 4 

U_kick   = (-1j * H_int_Y * dt).expm()       # Impart momentum-dependent phase
U_rot    = (-1j * H_free * t_quarter).expm() # Free evolution (Rotates X->P)
U_unkick = (-1j * H_int_Y * -dt).expm()      # Inverse kick

# Ideally, P-synthesis effectively looks like this in the interaction frame:
U_P_synth = U_unkick * U_rot * U_kick

# C. The Trotter Step [cite: 46]
# Combine X and P to approximate e^{-i(Xsx - Psy)dt} ~ Cooling
U_Trotter = U_P_synth * U_X 

# --- 4. Dissipation (Optical Pumping) ---
# Reset spin to ground state, removing entropy
c_ops = [np.sqrt(gamma) * sm_full] # Simple radiative decay
L_diss = liouvillian(H_free, c_ops) # Dissipation happens during free evolution
t_pump = 2.0 / gamma
Prop_diss = (L_diss * t_pump).expm()

# --- 5. Run Simulation ---
# One full cooling cycle = [Trotter Sequence] + [Dissipation]
# We apply the unitary evolution N times, then dissipate once.
U_coherent = U_Trotter ** n_trotter
# Convert Unitary to Superoperator for density matrix evolution
Prop_coherent = to_super(U_coherent)

Prop_Cycle = Prop_diss * Prop_coherent

# Initial State: Thermal motion, Ground spin
# rho_mot = thermal_dm(N, n_th)
# rho_mot = (- nu * a_dag * a / kT).expm()
# rho_mot = rho_mot / rho_mot.tr()
rho_mot = thermal_dm(N, n_th)
rho_spin = basis(2, 1) * basis(2, 1).dag() # Start in ground |g>
rho_0 = tensor(rho_spin, rho_mot)
initial_state = operator_to_vector(rho_0)

cycles = 10
times_sim = np.arange(cycles + 1)  # Integer steps for cycles

# Storage
occupancies_fast = np.zeros((cycles + 1, N))



# Initial occupancy
occupancies_fast[0, :] = rho_0.ptrace(1).diag().real

# Evolve step-by-step under the new pulse sequence
rho_current = initial_state
for k in range(1, cycles + 1):
    rho_current = Prop_Cycle * rho_current
    
    # Store data
    rho_mat = vector_to_operator(rho_current)
    occupancies_fast[k, :] = rho_mat.ptrace(1).diag().real


# Plot the occupancy of all motional states over cycles
plt.figure()
for n in range(N):
    plt.plot(times_sim, occupancies_fast[:, n], '.-', label=f'n={n}')
plt.xlabel('Pulse Cycle')
plt.ylabel('Occupancy')
plt.title('Occupancy vs Cooling Cycles (New Pulse Sequence)')
plt.legend()
plt.show()

# Bar plot
p_motional_initial = occupancies_fast[0, :]
p_motional_final = occupancies_fast[-1, :]
plt.figure()
plt.plot(np.arange(N), p_motional_initial, 'o-', label='Initial')
plt.bar(np.arange(N), p_motional_final, width=0.4, label='Final', alpha=0.7)
plt.xlabel('Motional State n')
plt.ylabel('Population')
plt.title('State Populations After Cooling Cycles (New Pulse Sequence)')
plt.legend()
plt.show()