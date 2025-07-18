"""Rocket class to define the rocket parameters and initial state."""
from __future__ import annotations

from typing import Optional

import numpy as np


class State:
    """Current state and time of the rocket in the inertial frame.

    The state is defined as a vector of the form [rx, ry, rz, vx, vy, vz, m]
    where rx, ry, rz are the position coordinates in meters and vx, vy, vz
    are the velocity components in meters per second, and m is the mass in kg.

    Attributes
    ----------
    t: float
        Current time in seconds
    pos: ndarray
        Current position of the rocket in [m, m, m]
    vel: ndarray
        Current velocity of the rocket in [m/s, m/s, m/s]
    mass: float
        Current mass of the rocket in kg
    state: ndarray
        State vector [rx, ry, rz, vx, vy, vz, m] in [m, m, m, m/s, m/s, m/s, kg]
    """

    def __init__(self, t0: float, pos0: np.ndarray, vel0: np.ndarray,
                    mass0: float) -> None:
        """Initialise the state of the rocket in the inertial frame.

        Parameters
        ----------
        t0: float
            Initial time in seconds
        pos0: ndarray
            Initial position of the rocket in [m, m, m]
        vel0: ndarray
            Initial velocity of the rocket in [m/s, m/s, m/s]
        mass0: float
            Initial mass of the rocket in kg
        """
        self.t = t0
        self.state = np.array([*pos0, *vel0, mass0], dtype=np.float64)

    @property
    def pos(self) -> np.ndarray:
        """Get the position vector of the rocket in [m, m, m]."""
        return self.state[:3]

    @property
    def vel(self) -> np.ndarray:
        """Get the velocity vector of the rocket in [m/s, m/s, m/s]."""
        return self.state[3:6]

    @property
    def mass(self) -> float:
        """Get the mass of the rocket in kg."""
        return self.state[-1]

    def set_mass(self, mass: float) -> None:
        """Set the mass of the rocket.

        Parameters
        ----------
        mass: float
            New mass of the rocket in kg
        """
        self.state[-1] = mass

    def set_pos(self, pos: np.ndarray) -> None:
        """Set the position of the rocket.

        Parameters
        ----------
        pos: ndarray
            New position of the rocket in [m, m, m]
        """
        self.state[:3] = pos

    def set_vel(self, vel: np.ndarray) -> None:
        """Set the velocity of the rocket.

        Parameters
        ----------
        vel: ndarray
            New velocity of the rocket in [m/s, m/s, m/s]
        """
        self.state[3:6] = vel


class Rocket:
    """Set the rocket parameters and initial state."""

    def __init__(self,
                    specific_impulse: float,
                    cd: float = 0,
                    area: float = 0,
                    parachute_cd: float = 0,
                    parachute_area: float = 0,
                    initial_state: Optional[State] = None) -> None:
        """Initialise the rocket with its initial state.

        Parameters
        ----------
        initial_state: State
            Initial state of the rocket [rx, ry, rz, vx, vy, vz, m]
            in [m, m, m, m/s, m/s, m/s, kg]
        specific_impulse: float
            Specific impulse of the rocket engine (assumed constant)
        cd: float
            Drag coefficint of the rocket
        area: float
            Drag area of the rocket [m^2]
        parachute_cd: float
            Parachute drag coefficient
        parachute_area: float
            Parachute drag area [m^2]
        """
        self.initial_state = initial_state if initial_state is not None else _empty_state()  # noqa: E501
        self.specific_impulse = specific_impulse
        self.cd = cd
        self.area = area
        self.parachute_cd = parachute_cd
        self.parachute_area = parachute_area


def _empty_state() -> State:
    """Create an empty state with zero position, velocity, and mass.

    Returns
    -------
    State
        Empty state.
    """
    return State(0.0, np.zeros(3), np.zeros(3), 0.0)
