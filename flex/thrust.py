"""Thrust profiles.

All thrust progiles should objects with __call__ methods that take a 'State' object as
input and return the thrust at that state in N.
"""

from abc import ABC, abstractmethod

import numpy as np

from .environment import State


class ThrustProfile(ABC):
    """Abstract base class for thrust profiles."""

    @abstractmethod
    def __call__(self, state: State) -> np.ndarray:
        """Calculate the thrust at the given state.

        Parameters
        ----------
        state: State
            Current state of the rocket

        Returns
        -------
        ndarray
            Thrust [N, N, N]
        """


class ConstantAlignedThrust(ThrustProfile):
    """Constant aligned thrust profile.

    Aligned meaning it is always in the direction of the rocket's velocity.
    """

    def __init__(self, constant_thrust: float) -> None:
        """Initialise the constant aligned thrust profile.

        Parameters
        ----------
        constant_thrust: float
            Constant thrust magnitude in N
        """
        self.constant_thrust = constant_thrust

    def __call__(self, state: State) -> np.ndarray:
        """Calculate the thrust at the given state.

        Parameters
        ----------
        state: State
            Current state of the rocket

        Returns
        -------
        ndarray
            Thrust [N, N, N]
        """
        return self.constant_thrust * state.vel / np.linalg.norm(state.vel) if np.linalg.norm(state.vel) > 0 else np.zeros(3)  # noqa: E501


class ConstantAlignedThrustLoad(ThrustProfile):
    """Constant aligned thrust load, thrust profile.

    Aligned meaning it is always in the direction of the rocket's velocity.
    """

    def __init__(self, constant_thrust_load: float) -> None:
        """Initialise the constant aligned thrust load thrust profile.

        Parameters
        ----------
        constant_thrust_load: float
            Constant thrust load in N/kg (defined as T0/m0, where T0 is the initial
            thrust and m0 is the initial mass)
        """
        self.constant_thrust_load = constant_thrust_load

    def __call__(self, state: State) -> np.ndarray:
        """Calculate the thrust at the given state.

        Parameters
        ----------
        state: State
            Current state of the rocket

        Returns
        -------
        ndarray
            Thrust [N, N, N]
        """
        return self.constant_thrust_load * state.mass * state.vel / np.linalg.norm(state.vel) if np.linalg.norm(state.vel) > 0 else np.zeros(3)  # noqa: E501


class ConstantThrustVector(ThrustProfile):
    """Constant thrust vector profile.

    The thrust vector is defined by a 3D vector.
    """

    def __init__(self, thrust_vector: np.ndarray) -> None:
        """Initialise the constant thrust vector profile.

        Parameters
        ----------
        thrust_vector: ndarray
            Thrust vector [N, N, N]
        """
        self.thrust_vector = thrust_vector

    def __call__(self, state: State) -> np.ndarray:
        """Calculate the thrust at the given state.

        Parameters
        ----------
        state: State
            Current state of the rocket

        Returns
        -------
        ndarray
            Thrust [N, N, N]
        """
        return self.thrust_vector


class ConstantOrbitalThrust(ThrustProfile):
    """Constant thrust vector profile relative to orbital parameters.

    The thrust vector is defined by a 3D vector.
    """

    def __init__(self, thrust_magnitude: float,
                        thrust_direction: np.ndarray) -> None:
        """Initialise the constant thrust vector profile.

        Defined in the LVLH frame.

        Parameters
        ----------
        thrust_magnitude: float
            Thrust intenisty in N
        thrust_direction: ndarray
            Thrust direction in the LVLH frame
        """
        self.thrust_magnitude = thrust_magnitude
        self.thrust_direction = thrust_direction

    def __call__(self, state: State) -> np.ndarray:
        """Calculate the thrust at the given state.

        Parameters
        ----------
        state: State
            Current state of the rocket

        Returns
        -------
        ndarray
            Thrust [N, N, N]
        """
        lvlh_z_dir = -state.pos / np.linalg.norm(state.pos)
        vel_dir = state.vel / np.linalg.norm(state.vel)

        lvlh_y_dir = np.cross(lvlh_z_dir, vel_dir)
        lvlh_x_dir = np.cross(lvlh_y_dir, lvlh_z_dir)

        rot_matrix = np.array([
            lvlh_x_dir,
            lvlh_y_dir,
            lvlh_z_dir,
        ]).transpose()

        return rot_matrix @ self.thrust_direction * self.thrust_magnitude
