"""Orbital simulations."""

import matplotlib.pyplot as plt
import numpy as np

from flex import accelerations
from flex.dynamics import CartesianDynamicSimulator
from flex.environment import Rocket, State
from flex.integration import ScipySolveIVP

sat_initial_state = State(
    t0=0, 
    pos0=np.array([7178000, 0, 0]),
    vel0=np.array([0, 0, np.sqrt(3.986004e14 / 7178000)]),
    mass0=100,
)

plt.figure(figsize=(10, 10))
# Define the rocket properties

satellite = Rocket(
    initial_state=sat_initial_state,
    specific_impulse=350,  # not used since there's no thrust
)

# Define the accelerations
acceleration_settings = accelerations.AccelerationSettings(
    accelerations_list=[
        accelerations.PointMassGravity(),
    ],
    rocket=satellite,
)

integration_settings = ScipySolveIVP(t_vec=np.arange(0, 10000, 1))

# Create the dynamic simulator
dynamic_simulator = CartesianDynamicSimulator(
    satellite,
    integration_settings,
    acceleration_settings,
)

# Perform the simulation
state_history = dynamic_simulator.simulate()

# Plotting
time = np.array(list(state_history.keys()))
state = np.vstack(list(state_history.values()))

print(f"Initial state: {state[0]}")
print(f"Final state: {state[-1]}")

energy_history = np.empty_like(time)
for idx, current_state in enumerate(state):
    energy_history[idx] = 0.5 * np.linalg.norm(current_state[3:6])**2 - \
                3.986004e14 / np.linalg.norm(current_state[0:3])

print(f"Energy history: {energy_history}")

plt.plot(state[:, 0], state[:, 2])

plt.xlabel("X position (m)")
plt.ylabel("Z position (m)")
plt.grid()
plt.legend()
plt.axis("equal")

plt.show()
