"""Main file for simulations."""

import matplotlib.pyplot as plt
import numpy as np

from flex import accelerations
from flex.dynamics import CartesianDynamicSimulator
from flex.environment import Rocket, State
from flex.integration import AltitudeEvent, MassEvent, ScipySolveIVP
from flex.thrust import ConstantAlignedThrust

# Define the rocket proprieties
rocket = Rocket(
    initial_state=State(0, np.array([6378000, 0, 0]), np.array([10, 0, 1]),
                                                            700000 / 9.81),
    specific_impulse=350,
)

# Define the kind of thrust profile
thrust_profile = ConstantAlignedThrust(constant_thrust=900000)

# Define the accelerations
acceleration_settings = accelerations.AccelerationSettings(
    accelerations_list=[
        accelerations.PointMassGravity(),
        accelerations.Thrust(thrust_profile),
    ],
    rocket=rocket,
)

# Define the termination settings
altitude_termination = AltitudeEvent(
    target_altitude=0,
    terminal=True,
    direction=-1,
)
mass_termination = MassEvent(
    target_mass=1300,
    terminal=True,
    direction=-1,
)

events = [
    altitude_termination,  # terminates when altitude is 0m or lower
    mass_termination,  # terminates when mass is 20kg or lower
]

# Define the integration settings
integration_settings = ScipySolveIVP(
    t_vec=np.arange(0, 6000, 10),
    event_settings=events)

# Create the dynamic simulator
dynamic_simulator = CartesianDynamicSimulator(
    rocket,
    integration_settings,
    acceleration_settings,
)

# Perform the simulation
state_history = dynamic_simulator.simulate()
event_times = dynamic_simulator.sol.t_events

# Plotting
time = np.array(list(state_history.keys()))
state = np.vstack(list(state_history.values()))

plt.figure(figsize=(7, 5))

plt.plot(state[:, 0], state[:, 2], linewidth=2)
plt.xlabel("X (m)")
plt.ylabel("Z (m)")
plt.title("Constant thrust trajectory")
plt.tight_layout()

plt.figure(figsize=(7, 5))

plt.plot(time, np.linalg.norm(state[:, :3], axis=1) - 6378000, linewidth=2)
plt.xlabel("Time (s)")
plt.ylabel("Altitude (m)")
plt.title("Constant thrust altitude over time")
plt.tight_layout()

plt.figure(figsize=(7, 5))

plt.plot(time, state[:, 0], linewidth=2)
print(event_times)
plt.xlabel("Time (s)")
plt.ylabel("X (m)")
plt.title("X coordinate over flat Earth")
plt.tight_layout()

plt.figure(figsize=(7, 5))

plt.plot(time, state[:, 2], linewidth=2)
plt.xlabel("Time (s)")
plt.ylabel("Z (m)")
plt.title("Z coordinate over flat Earth")
plt.tight_layout()

plt.show()
