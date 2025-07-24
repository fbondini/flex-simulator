"""Integrator class."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, List, Optional, Tuple

from scipy.integrate import odeint, solve_ivp

import numpy as np

PRECISION = 8  # precision digits in time steps indices


class IntegratorSettings(ABC):
    """Defines the integrator settings and sets the ODE solver."""

    def __init__(self, tspan: List) -> None:
        """Initialise the generic integrator.

        Parameters
        ----------
        tspan: list
            Start and end time of the simulation.
        """
        self.tspan = tspan

    @abstractmethod
    def integrate(self, fun: Callable, x0: np.ndarray, args: Tuple) -> Tuple:
        """Run the integration.

        Parameters
        ----------
        fun: Callable
            Function describing the derivative of the state.
        x0: np.ndarray
            Initial state
        args: Tuple
            Additional arguments to be passed to fun

        Returns
        -------
        Tuple[state_history, y]
            state history: Dictionary containing the times as keys and np.ndarrays of
            the state at that time for every entry.2s
            y: full integrator output
        """
        return  # type: ignore


class ScipyOdeint(IntegratorSettings):
    """Defines the class for odeint settings and sets the ODE solver.

    Does not support events.
    """

    def __init__(self, t_vec: np.ndarray) -> None:
        """Initialise the integrator.

        Parameters
        ----------
        t_vec: ndarray
            Time vector at which to evaluate the state
        """
        self.t_vec = t_vec

    def integrate(self, fun: Callable, x0: np.ndarray, args: Tuple) -> Tuple:
        """Run the integration.

        Parameters
        ----------
        fun: Callable
            Function describing the derivative of the state (expects t, x).
        x0: np.ndarray
            Initial state
        args: Tuple
            Additional arguments to be passed to fun

        Returns
        -------
        Tuple[state_history, y]
            state history: Dictionary containing the times as keys and np.ndarrays of
            the state at that time for every entry.
            y: full odeint output
        """
        # Patch fun to swap arguments for odeint
        def odeint_fun(x, t, *args):  # noqa: ANN001, ANN002, ANN202
            return fun(t, x, *args)

        y = odeint(odeint_fun, x0, self.t_vec, args)

        state_history = dict()
        for i, t in enumerate(self.t_vec):
            state_history[round(float(t), PRECISION)] = y[i, :]
        return state_history, y


class ScipySolveIVP(IntegratorSettings):
    """Defines the class for solve_ivp settings and sets the ODE solver."""

    def __init__(self, 
                 t_vec: List[float] | np.ndarray,
                 event_settings: Optional[List["EventSetting"]] = None,
                 method: str = 'RK45',
                 rtol: float = 1e-10,
                 atol: float = 1e-12,
                 max_step: float = np.inf,
                 first_step: Optional[float] = None,
                 dense_output: bool = False,
                 vectorized: bool = False,
                 **options
                 ) -> None:
        """Initialise the integrator.

        Parameters
        ----------
        t_vec: List | ndarray
            Times at which to evaluate the solution.
        event_settings: List | None
            List of event setting objects
        method: str
            Integration method. One of 'RK45' (default), 'RK23', 'DOP853', 
            'Radau', 'BDF', 'LSODA'.
        rtol: float
            Relative tolerance parameter. Default 1e-10 for orbital mechanics.
        atol: float
            Absolute tolerance parameter. Default 1e-12 for orbital mechanics.
        max_step: float
            Maximum allowed step size.
        first_step: float | None
            Initial step size. If None, the algorithm will choose.
        dense_output: bool
            Whether to compute a continuous solution.
        vectorized: bool
            Whether the function is implemented in a vectorized fashion.
        **options
            Additional options passed to the integration method.
        """
        self.t_span = [t_vec[0], t_vec[-1]]
        self.t_vec = t_vec
        self.event_settings = event_settings
        self.method = method
        self.rtol = rtol
        self.atol = atol
        self.max_step = max_step
        self.first_step = first_step
        self.dense_output = dense_output
        self.vectorized = vectorized
        self.options = options

    def integrate(self, fun: Callable, x0: np.ndarray, args: Tuple) -> Tuple:
        """Run the integration.

        Parameters
        ----------
        fun: Callable
            Function describing the derivative of the state.
        x0: np.ndarray
            Initial state
        args: Tuple
            Additional arguments to be passed to fun

        Returns
        -------
        Tuple[state_history, sol]
            state history: Dictionary containing the times as keys and np.ndarrays of
            the state at that time for every entry.
            sol: full solve_ivp output
        """
        sol = solve_ivp(
            fun, 
            self.t_span, 
            x0, 
            method=self.method,
            t_eval=self.t_vec,
            dense_output=self.dense_output,
            events=self.event_settings,
            vectorized=self.vectorized,
            args=args,
            rtol=self.rtol,
            atol=self.atol,
            max_step=self.max_step,
            first_step=self.first_step,
            **self.options
        )

        state_history = dict()
        for i, t in enumerate(sol.t):
            state_history[round(float(t), PRECISION)] = sol.y[:, i]
        return state_history, sol



class EventSetting(ABC):
    """Abstract class for event setting."""

    @abstractmethod
    def __init__(self, terminal: bool = True, direction: float = 0.0) -> None:  # noqa: FBT001, FBT002
        self.terminal = terminal
        self.direction = direction

    @abstractmethod
    def __call__(self, t: float, y: np.ndarray, *args: Tuple) -> float:
        """Event function.

        Parameters
        ----------
        t: float
            Current time in s
        y: ndarray
            current state
        args: Tuple
            Additional arguments to be passed

        Returns
        -------
        float
            The value should be zero when a given event happens.
        """

    def __getattr__(self, name: str) -> float:
        """Make solve_ivp see self.terminal and self.direction.

        Parameters
        ----------
        name: str
            Name of the attriubute to get.

        Returns
        -------
        float
            The value of the attribute.

        Raises
        ------
        AttributeError
            If the attribute does not exist.
        """
        if name in ["terminal", "direction"]:
            return getattr(self, name)
        error_msg = f"{type(self).__name__} object has no attribute {name}"
        raise AttributeError(error_msg)


class AltitudeEvent(EventSetting):
    """Flat Earth altitude event setting.

    Assumes the z coordinate to be the altitude relative to the
    flat Earth surface. The event value is negative for altitudes lower
    than the target altitude.
    """

    def __init__(self, target_altitude: float, earth_radius: float = 6378000,
                    terminal: bool = True, direction: float = -1.0) -> None:  # noqa: FBT001, FBT002
        super().__init__(terminal=terminal, direction=direction)
        self.target_altitude = target_altitude
        self.earth_radius = earth_radius

    def __call__(self, t: float, y: np.ndarray, *args: Tuple) -> float:
        """Event function.

        Parameters
        ----------
        t: float
            Current time in s
        y: ndarray
            current state
        args: Tuple
            Additional arguments to be passed

        Returns
        -------
        float
            Value is zero when the target altitude is reached.
            The event value is negative for altitudes lower
            than the target altitude.
        """
        return (y[2] - self.earth_radius) - self.target_altitude


class MassEvent(EventSetting):
    """Mass event setting."""

    def __init__(self, target_mass: float,
                    terminal: bool = True, direction: float = 1.0) -> None:  # noqa: FBT001, FBT002
        super().__init__(terminal=terminal, direction=direction)
        self.target_mass = target_mass

    def __call__(self, t: float, y: np.ndarray, *args: Tuple) -> float:
        """Event function.

        Parameters
        ----------
        t: float
            Current time in s
        y: ndarray
            current state
        args: Tuple
            Additional arguments to be passed

        Returns
        -------
        float
            Value is zero when the target mass is reached.
            The event value is positive when mass is larger
            than the target mass.
        """
        return y[-1] - self.target_mass
