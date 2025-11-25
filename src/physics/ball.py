"""
Ball physics module

Implements ball dynamics including:
- Position and velocity
- Spin (angular velocity)
- Magnus effect (accurate model)
- Air resistance
- Spin decay
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any
from .parameters import PhysicsParameters, SpinType


class Ball:
    def __init__(
        self,
        params: PhysicsParameters,
        position: np.ndarray = None,
        velocity: np.ndarray = None,
        spin: np.ndarray = None
    ):
        self.params = params
        self.position = np.array(position, dtype=float) if position is not None else np.array([0.0, 0.0, 1.0])
        self.velocity = np.array(velocity, dtype=float) if velocity is not None else np.array([0.0, 0.0, 0.0])
        self.spin = np.array(spin, dtype=float) if spin is not None else np.array([0.0, 0.0, 0.0])
        self.trajectory = [self.position.copy()]
        self.velocity_history = [self.velocity.copy()]
        self.spin_history = [self.spin.copy()]
        self.time_history = [0.0]
        self.current_time = 0.0

    def reset(self, position: np.ndarray = None, velocity: np.ndarray = None, spin: np.ndarray = None):
        if position is not None:
            self.position = np.array(position, dtype=float)
        if velocity is not None:
            self.velocity = np.array(velocity, dtype=float)
        if spin is not None:
            self.spin = np.array(spin, dtype=float)
        self.trajectory = [self.position.copy()]
        self.velocity_history = [self.velocity.copy()]
        self.spin_history = [self.spin.copy()]
        self.time_history = [0.0]
        self.current_time = 0.0

    def get_spin_parameter(self) -> float:
        speed = np.linalg.norm(self.velocity)
        spin_rate = np.linalg.norm(self.spin)
        if speed < 1e-6:
            return 0.0
        return spin_rate * self.params.ball_radius / speed

    def get_lift_coefficient(self) -> float:
        S = self.get_spin_parameter()
        if S < 1e-6:
            return 0.0
        Cl = 1.0 / (2.0 + 1.0 / S)
        return min(Cl, 0.6) * self.params.spin.magnus_coefficient

    def compute_gravity_force(self) -> np.ndarray:
        return np.array([0.0, 0.0, -self.params.ball_mass * self.params.gravity])

    def compute_drag_force(self) -> np.ndarray:
        speed = np.linalg.norm(self.velocity)
        if speed < 1e-6:
            return np.zeros(3)
        cross_section = self.params.ball.cross_section_area
        drag_coeff = self.params.ball.drag_coefficient
        return -0.5 * self.params.air_density * drag_coeff * cross_section * speed * self.velocity

    def compute_magnus_force(self) -> np.ndarray:
        speed = np.linalg.norm(self.velocity)
        spin_rate = np.linalg.norm(self.spin)
        if speed < 1e-6 or spin_rate < 1e-6:
            return np.zeros(3)
        Cl = self.get_lift_coefficient()
        cross_section = self.params.ball.cross_section_area
        magnus_direction = np.cross(self.spin, self.velocity)
        magnus_direction_norm = np.linalg.norm(magnus_direction)
        if magnus_direction_norm < 1e-6:
            return np.zeros(3)
        magnus_direction = magnus_direction / magnus_direction_norm
        magnus_magnitude = 0.5 * self.params.air_density * Cl * cross_section * speed * speed
        return magnus_magnitude * magnus_direction

    def compute_forces(self) -> np.ndarray:
        return self.compute_gravity_force() + self.compute_drag_force() + self.compute_magnus_force()

    def compute_spin_decay(self, dt: float) -> np.ndarray:
        decay_factor = self.params.get_spin_decay_factor(dt)
        return self.spin * decay_factor

    def update(self, dt: Optional[float] = None) -> None:
        if dt is None:
            dt = self.params.dt
        force = self.compute_forces()
        acceleration = force / self.params.ball_mass
        self.velocity += acceleration * dt
        self.position += self.velocity * dt
        self.spin = self.compute_spin_decay(dt)
        self.current_time += dt
        self.trajectory.append(self.position.copy())
        self.velocity_history.append(self.velocity.copy())
        self.spin_history.append(self.spin.copy())
        self.time_history.append(self.current_time)

    def apply_impulse(self, impulse: np.ndarray, spin_change: np.ndarray = None):
        self.velocity += impulse / self.params.ball_mass
        if spin_change is not None:
            self.spin += spin_change

    def get_kinetic_energy(self) -> float:
        translational = 0.5 * self.params.ball_mass * np.dot(self.velocity, self.velocity)
        moment_of_inertia = self.params.ball.moment_of_inertia
        rotational = 0.5 * moment_of_inertia * np.dot(self.spin, self.spin)
        return translational + rotational

    def get_speed(self) -> float:
        return np.linalg.norm(self.velocity)

    def get_spin_rate(self) -> float:
        return np.linalg.norm(self.spin)

    def get_spin_rpm(self) -> float:
        return self.get_spin_rate() * 60.0 / (2.0 * np.pi)

    def get_state(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.position.copy(), self.velocity.copy(), self.spin.copy()

    def get_trajectory_data(self) -> Dict[str, np.ndarray]:
        return {
            "time": np.array(self.time_history),
            "position": np.array(self.trajectory),
            "velocity": np.array(self.velocity_history),
            "spin": np.array(self.spin_history)
        }

    def __repr__(self) -> str:
        return f"Ball(pos={self.position}, vel={self.velocity}, spin={self.spin})"


def create_topspin_ball(params: PhysicsParameters, speed: float = 15.0, spin_rpm: float = 3000.0):
    velocity = np.array([speed, 0.0, 2.0])
    spin = np.array([0.0, spin_rpm * 2 * np.pi / 60.0, 0.0])
    return Ball(params, position=np.array([-1.0, 0.0, 0.9]), velocity=velocity, spin=spin)


def create_backspin_ball(params: PhysicsParameters, speed: float = 10.0, spin_rpm: float = 2000.0):
    velocity = np.array([speed, 0.0, 3.0])
    spin = np.array([0.0, -spin_rpm * 2 * np.pi / 60.0, 0.0])
    return Ball(params, position=np.array([-1.0, 0.0, 0.9]), velocity=velocity, spin=spin)
