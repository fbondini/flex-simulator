"""Multi Stage Rocket launcher simulation."""

import matplotlib.pyplot as plt
import numpy as np

from flex import accelerations
from flex.atmosphere import UniformDensityModel
from flex.dynamics import CartesianDynamicSimulator, FlightSimulator, Phase
from flex.environment import Rocket, State
from flex.integration import MassEvent, ScipySolveIVP
from flex.thrust import (
    ConstantAlignedThrust,
    ConstantOrbitalThrust,
    ConstantThrustVector,
)


# Pitch over maneuver function
def pitch_over(state: State) -> State:  # noqa: D103
    speed = np.linalg.norm(state.vel)
    pitch_angle = np.deg2rad(1.2)  # degrees in radians

    # New velocity components after pitch-over in x-z plane
    velx = speed * np.cos(pitch_angle)
    velz = speed * np.sin(pitch_angle)

    state.set_vel(np.array([velx, 0, velz]))
    return state


# ### Defining stages proprieties, and thrust profiles
# Proprieties of the Falcon Heavy with 2 boosters
first_stage_w_boosters = Rocket(
    initial_state=State(0, np.array([6378000, 0, 0]), np.zeros(3), 1360000),
    specific_impulse=2800 / 9.81,
    cd=0.5,
    area=32,
)

# Phase 1 - vertical ascent, thrust profile
phase1_thrust = ConstantThrustVector(
    thrust_vector=np.array([19760000, 0, 0]),
)
# Phase 2 - gravity turn, thrust profile
phase2_thrust = ConstantAlignedThrust(
    constant_thrust=19760000,
)

# Proprieties of the Falcon Heave core (one falcon 9)
first_stage_core = Rocket(
    specific_impulse=2800 / 9.81,
)
falcon9_dry_mass = 22000

# Phase 3 - gravity turn thrust profile
phase3_thrust = ConstantAlignedThrust(
    constant_thrust=7600000,
)
first_stage_dry_mass = 140000

# Proprieties of the second stage
second_stage = Rocket(
    specific_impulse=3400 / 9.81,
)

# Phase 4 - no thrust (coasting)
# Phase 5 - constant thrust profile
angle = np.deg2rad(0)
phase5_thrust = ConstantOrbitalThrust(
    thrust_magnitude=981000,
    thrust_direction=np.array([np.cos(angle), 0, -np.sin(angle)]),
)
# phase5_thrust = ConstantAlignedThrust(981000)

# ### Defining the accelerations
# Gravity acceleration
gravity_acceleration = accelerations.PointMassGravity()

# Acceleration settings
# Phase 1 - vertical ascent with boosters
phase1_acceleration_settings = accelerations.AccelerationSettings(
    accelerations_list=[
        gravity_acceleration,
        accelerations.Thrust(phase1_thrust),
        # accelerations.AtmosphericDrag(UniformDensityModel(1.225)),
    ],
    rocket=first_stage_w_boosters,
)

# Phase 2 - gravity turn with boosters
phase2_acceleration_settings = accelerations.AccelerationSettings(
    accelerations_list=[
        gravity_acceleration,
        accelerations.Thrust(phase2_thrust),
        # accelerations.AtmosphericDrag(UniformDensityModel(0.225)),
    ],
    rocket=first_stage_w_boosters,
)

# Phase 3 - gravity turn with only core stage (booster jettinsoned)
phase3_acceleration_settings = accelerations.AccelerationSettings(
    accelerations_list=[
        gravity_acceleration,
        accelerations.Thrust(phase3_thrust),
    ],
    rocket=first_stage_core,
)

# Phase 4 - coasting with only second stage (first stage jettinsoned)
phase4_acceleration_settings = accelerations.AccelerationSettings(
    accelerations_list=[
        gravity_acceleration,
    ],
    rocket=second_stage,
)

# Phase 5 - circularisation burn with second stage
phase5_acceleration_settings = accelerations.AccelerationSettings(
    accelerations_list=[
        gravity_acceleration,
        accelerations.Thrust(phase5_thrust),
    ],
    rocket=second_stage,
)

# Phase 6 - orbital phase
phase6_acceleration_settings = accelerations.AccelerationSettings(
    accelerations_list=[
        gravity_acceleration,
    ],
    rocket=second_stage,
)


# ### Defining the integration settings
phase1_integration_settings = ScipySolveIVP(
    t_vec=np.arange(0, 10, 0.1),
)

phase2_integration_settings = ScipySolveIVP(
    t_vec=np.arange(0, 154, 0.1),
)

phase3_integration_settings = ScipySolveIVP(
    t_vec=np.arange(0, 100, 0.1),
    event_settings=[
        MassEvent(
            target_mass=first_stage_dry_mass,
            terminal=True,
            direction=-1.0,
        ),
    ],
)

phase4_integration_settings = ScipySolveIVP(
    t_vec=np.arange(0, 300, 0.1),
)

phase5_integration_settings = ScipySolveIVP(
    t_vec=np.arange(0, 300, 0.1),
    event_settings=[
        MassEvent(
            target_mass=14000,
            terminal=True,
            direction=-1.0,
        ),
    ],
)

phase6_integration_settings = ScipySolveIVP(
    t_vec=np.arange(0, 5000, 10),
)


# ### Creating the dynamic simulators and phases
phase1 = Phase(
    dynamics_simulator=CartesianDynamicSimulator(
        rocket=first_stage_w_boosters,
        integrator_settings=phase1_integration_settings,
        acceleration_settings=phase1_acceleration_settings,
    ),
)

phase2 = Phase(
    dynamics_simulator=CartesianDynamicSimulator(
        rocket=first_stage_w_boosters,
        integrator_settings=phase2_integration_settings,
        acceleration_settings=phase2_acceleration_settings,
    ),
    initial_state_variation=pitch_over,
)

phase3 = Phase(
    dynamics_simulator=CartesianDynamicSimulator(
        rocket=first_stage_core,
        integrator_settings=phase3_integration_settings,
        acceleration_settings=phase3_acceleration_settings,
    ),
    # remove the jettinsoned booosters mass from the initial state
    initial_state_variation=lambda state: (
        state.set_mass(state.mass - 2 * falcon9_dry_mass), state)[1],
)

phase4 = Phase(
    dynamics_simulator=CartesianDynamicSimulator(
        rocket=second_stage,
        integrator_settings=phase4_integration_settings,
        acceleration_settings=phase4_acceleration_settings,
    ),
    # remove the jettinsoned booosters mass from the initial state
    initial_state_variation=lambda state: (
        state.set_mass(state.mass - falcon9_dry_mass), state)[1],
)

phase5 = Phase(
    dynamics_simulator=CartesianDynamicSimulator(
        rocket=second_stage,
        integrator_settings=phase5_integration_settings,
        acceleration_settings=phase5_acceleration_settings,
    ),
)

phase6 = Phase(
    dynamics_simulator=CartesianDynamicSimulator(
        rocket=second_stage,
        integrator_settings=phase6_integration_settings,
        acceleration_settings=phase6_acceleration_settings,
    ),
)


# ### Create the full flight simulator and run the simulation
# Simulator
falcon_heavy_launch = FlightSimulator(
    phases=[phase1, phase2, phase3, phase4, phase5, phase6],
)

# Run the simulation
state_history = falcon_heavy_launch.simulate()


# # ### Plotting
time = np.array(list(state_history.keys()))
state = np.vstack(list(state_history.values()))

plt.figure(figsize=(7, 5))

plt.plot(state[:, 0], state[:, 2], linewidth=2)
plt.xlabel("X (m)")
plt.ylabel("Z (m)")
plt.title("Falcon Heavy Trajectory")
plt.tight_layout()

plt.figure(figsize=(7, 5))

plt.plot(time, np.linalg.norm(state[:, 3:6], axis=1) * 1e-3, linewidth=2)
plt.xlabel("Time (s)")
plt.ylabel("V (km/s)")
plt.title("Falcon Heavy velocity")
plt.tight_layout()

plt.figure(figsize=(7, 5))

plt.plot(time, np.linalg.norm(state[:, :3], axis=1) * 1e-3 - 6378, linewidth=2)
plt.xlabel("Time (s)")
plt.ylabel("h (km)")
plt.title("Falcon Heavy altitude")
plt.tight_layout()

plt.show()
