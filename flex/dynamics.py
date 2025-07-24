"""Dynamics simulator abstract class."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Callable, Dict, List, Optional

import numpy as np

from .accelerations import AccelerationSettings, Thrust
from .environment import Rocket, State

if TYPE_CHECKING:
    from .integration import IntegratorSettings


class DynamicsSimulator(ABC):
    """Abstract class for dynamics simulators."""

    def __init__(self, rocket: Rocket,
                    integrator_settings: IntegratorSettings,
                    acceleration_settings: AccelerationSettings,
                    ) -> None:
        """Initialise the dynamics simulator.

        Parameters
        ----------
        rocket: Rocket
            Rocket object
        integrator_settings: IntegratorSettings
            Object defining the integrator algorithm and settings.
        acceleration_settings: AccelerationSettings
            Object defining the accelerations acring on the rocket
        """
        self.rocket = rocket
        self.integrator_settings = integrator_settings
        self.acceleration_settings = acceleration_settings

    def simulate(self) -> Dict:
        """Integrates the dynamical equations at the provided timesteps.

        Returns
        -------
        dict
            Dictionary containing the times as keys and np.ndarrays of
            the state at that time for every entry.
        """
        state_history, sol = self.integrator_settings.integrate(
            self.dynamics_equation,
            self.rocket.initial_state.state,
            (self.rocket, self.acceleration_settings),
        )

        self.state_history = state_history
        self.sol = sol

        return state_history

    @staticmethod
    @abstractmethod
    def dynamics_equation(t: float, x: np.ndarray, rocket: Rocket,
                            acceleration_settings: AccelerationSettings) -> np.ndarray:
        """Define the dynamics differential equations without gravity torque.

        To be passed to the integrator to integrate the state.

        Parameters
        ----------
        t: float
            Time (s).
        x: np.ndarray
            Current attitude state (either 6 components for eul angles + w,
            or 7 for quats + w).
        rocket: Rocket
            Object defining the rocket parameters.
        acceleration_settings: AccelerationSettings
            Object defining the accelerations acring on the rocket

        Returns
        -------
        ndarray
            State derivative.
        """


class CartesianDynamicSimulator(DynamicsSimulator):
    """Dynamic simulator propagating cartesian inertial state."""

    def __init__(self, rocket: Rocket,
                    integrator_settings: IntegratorSettings,
                    acceleration_settings: AccelerationSettings,
                    ) -> None:
        """Initialise the dynamics simulator.

        Parameters
        ----------
        rocket: Rocket
            Rocket object
        integrator_settings: IntegratorSettings
            Object defining the integrator algorithm and settings.
        acceleration_settings: AccelerationSettings
            Object defining the accelerations acring on the rocket
        """
        super().__init__(rocket, integrator_settings, acceleration_settings)

    @staticmethod
    def dynamics_equation(t: float, x: np.ndarray, rocket: Rocket,
                            acceleration_settings: AccelerationSettings) -> np.ndarray:
        """Define the dynamics differential equations without gravity torque.

        To be passed to the integrator to integrate the state.

        Parameters
        ----------
        t: float
            Time (s).
        x: np.ndarray
            Current attitude state (either 6 components for eul angles + w,
            or 7 for quats + w).
        rocket: Rocket
            Object defining the rocket parameters.
        acceleration_settings: AccelerationSettings
            Object defining the accelerations acring on the rocket

        Returns
        -------
        ndarray
            State derivative.
        """
        current_pos = x[:3]
        current_vel = x[3:6]
        current_mass = x[-1]

        current_state = State(t, current_pos, current_vel, current_mass)
        acc = acceleration_settings.acceleration(current_state)
        vel = x[3:6]

        # Checks if there's a thrust acceleration in the acceleration list
        # If there is, it saves the current thrust vector, to calculate the mass rate
        acceleration_list = acceleration_settings.accelerations
        mass_flow_rate = 0.0
        for acceleration_instance in acceleration_list:
            if isinstance(acceleration_instance, Thrust):
                current_thrust = acceleration_instance(current_state, rocket) * current_mass  # noqa: E501
                mass_flow_rate = -np.linalg.norm(current_thrust) / (rocket.specific_impulse * 9.81)  # noqa: E501

        # state derivative
        return np.append(vel, np.append(acc, mass_flow_rate)).flatten()


class Phase:
    """Define a single flight phase."""

    def __init__(self, dynamics_simulator: DynamicsSimulator,
                    initial_state: Optional[State] = None,
                    initial_state_variation: Optional[Callable[[State], State]] = None,
                    ) -> None:
        """Initialise the flight phase."""
        self.dynamics_simulator = dynamics_simulator
        self.initial_state = initial_state
        self.initial_state_variation = initial_state_variation

    @staticmethod
    def _no_var(state: State) -> State:
        return state

    def state_mod(self, previous_state: State) -> State:
        """Modify the initial state.

        Returns
        -------
        state: State
            Modified state.
        """
        if self.initial_state_variation is None and self.initial_state is None:
            return previous_state

        state = previous_state
        if self.initial_state is not None:
            state = self.initial_state
        if self.initial_state_variation is not None:
            state = self.initial_state_variation(state)

        return state


class FlightSimulator:
    """Simulate a flight with multiple phases."""

    def __init__(self, phases: List[Phase]) -> None:
        """Initialise the flight simulator with a list of phases."""
        self.phases = phases
        self.state_history = dict()
        self.sol = []
        self.phase_times = []

    def simulate(self) -> Dict:
        """Simulate the flight by iterating thorugh the phases.

        Returns
        -------
        Dict
            Combined state history of all phases.
        """
        current_state_history = dict()
        previous_phase_end_time = 0
        for phase_idx, phase in enumerate(self.phases):
            if phase_idx == 0:
                phase.dynamics_simulator.rocket.initial_state = phase.state_mod(
                    phase.dynamics_simulator.rocket.initial_state,
                )
            else:
                final_state_vector = list(current_state_history.values())[-1]
                phase.dynamics_simulator.rocket.initial_state = phase.state_mod(
                    State(
                        t0=list(current_state_history.keys())[-1],
                        pos0=final_state_vector[:3],
                        vel0=final_state_vector[3:6],
                        mass0=final_state_vector[-1],
                    ),
                )
                self.phase_times.append(current_state_history[list(current_state_history.keys())[-1]])

            current_state_history = phase.dynamics_simulator.simulate()
            current_state_history = self.shift_state_history(
                            current_state_history, previous_phase_end_time)

            previous_phase_end_time = list(current_state_history.keys())[-1]

            self.state_history.update(current_state_history)
            self.sol.append(phase.dynamics_simulator.sol)

        return self.state_history

    @staticmethod
    def shift_state_history(state_history: dict[float, np.ndarray], time_offset: float) -> dict[float, np.ndarray]:  # noqa: D102, E501
        return {t + time_offset: state for t, state in state_history.items()}
