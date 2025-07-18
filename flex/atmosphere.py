"""Atmospheric density models.

All density models should objects with __call__ methods that take a 'State' object as
input and return the atmospheric density at that state in kg/m^3.
"""

from abc import ABC, abstractmethod

from .environment import State


class DensityModel(ABC):
    """Abstract base class for atmospheric density models."""

    @abstractmethod
    def __call__(self, state: State) -> float:
        """Calculate the atmospheric density at the given state.

        Parameters
        ----------
        state: State
            Current state of the rocket

        Returns
        -------
        float
            Atmospheric density [kg/m^3]
        """


class UniformDensityModel(DensityModel):
    """Uniform atmospheric density model."""

    def __init__(self, density: float = 1.225):
        """Initialize the uniform density model with a constant density.

        Parameters
        ----------
        density: float
            Constant atmospheric density [kg/m^3]
        """
        self.density = density

    def __call__(self, state: State) -> float:
        """Uniform atmospheric density model.

        Parameters
        ----------
        state: State
            Current state of the rocket

        Returns
        -------
        float
            Constant atmospheric density [kg/m^3]
        """
        return self.density
