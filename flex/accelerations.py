"""Acceleration settings."""
from abc import ABC, abstractmethod
from typing import List

import numpy as np

from .atmosphere import DensityModel
from .environment import Rocket, State
from .thrust import ThrustProfile


class AccelerationSettings:
    """Defines the accelerations acting on the rocket."""

    def __init__(self, accelerations_list: List["Acceleration"],
                    rocket: Rocket) -> None:
        """Initialise the acceleration settings.

        Parameters
        ----------
        accelerations_list: List[Acceleration]
            List describing the accelerations acting on the rocket.
        rocket: Rocket
            Object defining the rocket parameters.
        """
        self.accelerations = accelerations_list
        self.rocket = rocket

    def acceleration(self, state: State) -> np.ndarray:
        """Calculate the acceleration of the rocket.

        Parameters
        ----------
        state: State
            State of the rocket

        Returns
        -------
        ndarray
            Acceleration vector of the rocket in m/s^2
        """
        acceleration = np.zeros(3)

        for acceleration_instance in self.accelerations:
            acceleration += acceleration_instance(state, self.rocket)

        return acceleration


class Acceleration(ABC):
    """Abstract class to define an acceleration."""

    @abstractmethod
    def __init__(self) -> None:
        """Initialise the acceleration."""
        return

    @abstractmethod
    def __call__(self, state: State, rocket: Rocket) -> np.ndarray:
        """Calculate the acceleration vector.

        Parameters
        ----------
        state: State
            State of the rocket
        rocket: Rocket
            Object defining the rocket parameters.

        Returns
        -------
        ndarray
            Acceleration vector of the rocket in m/s^2
        """


class ConstantGravity(Acceleration):
    """Constnt gravity acceleration."""

    def __init__(self, gravitational_acceleration: float = 9.81) -> None:
        """Initialise the constant gravity acceleration.

        Parameters
        ----------
        gravitational_acceleration: float
            Gravitational acceleration in m/s^2
        """
        super().__init__()
        self.gravitational_acceleration = gravitational_acceleration

    def __call__(self, state: State, rocket: Rocket) -> np.ndarray:
        """Calculate the constant gravity acceleration vector.

        Parameters
        ----------
        state: State
            State of the rocket
        rocket: Rocket
            Object defining the rocket parameters.

        Returns
        -------
        ndarray
            Acceleration vector of the rocket in m/s^2
        """
        return np.array([0.0, 0.0, -self.gravitational_acceleration])


class PointMassGravity(Acceleration):
    """Point mass gravity acceleration."""

    def __init__(self, gravitational_parameter: float = 3.986004e14) -> None:
        """Initialise the point mass gravity acceleration.

        Parameters
        ----------
        gravitational_parameter: float
            Gravitational parameter in m^3/s^2
        """
        super().__init__()
        self.gravitational_parameter = gravitational_parameter

    def __call__(self, state: State, rocket: Rocket) -> np.ndarray:
        """Calculate the constant gravity acceleration vector.

        Parameters
        ----------
        state: State
            State of the rocket
        rocket: Rocket
            Object defining the rocket parameters.

        Returns
        -------
        ndarray
            Acceleration vector of the rocket in m/s^2
        """
        mu = self.gravitational_parameter
        r = state.pos
        return -mu * r / np.linalg.norm(r)**3


class AtmosphericDrag(Acceleration):
    """Atmospheric drag acceleration."""

    def __init__(self, density_model: DensityModel) -> None:
        """Initialise the atmospheric drag acceleration.

        Parameters
        ----------
        density_model: DensityModel
            Atmospheric density model to use for calculating drag.
        """
        super().__init__()
        self.density_model = density_model

    def __call__(self, state: State, rocket: Rocket) -> np.ndarray:
        """Calculate the atmospheric drag acceleration vector.

        Parameters
        ----------
        state: State
            State of the rocket
        rocket: Rocket
            Object defining the rocket parameters.

        Returns
        -------
        ndarray
            Acceleration vector of the rocket in m/s^2
        """
        rho = self.density_model(state)
        v = state.vel
        cd = rocket.cd
        area = rocket.area
        return -0.5 * rho * cd * area * np.linalg.norm(v) * v / state.mass


class Thrust(Acceleration):
    """Thrust acceleration."""

    def __init__(self, thrust_profile: ThrustProfile) -> None:
        """Initialise the thrust acceleration.

        Parameters
        ----------
        thrust_profile: ThrustProfile
            Thrust profile to use for calculating the thrust.
        """
        super().__init__()
        self.thrust_profile = thrust_profile

    def __call__(self, state: State, rocket: Rocket) -> np.ndarray:
        """Calculate the thrust acceleration vector.

        Parameters
        ----------
        state: State
            State of the rocket
        rocket: Rocket
            Object defining the rocket parameters.

        Returns
        -------
        ndarray
            Acceleration vector of the rocket in m/s^2
        """
        return self.thrust_profile(state) / state.mass
