"""
Racket physics module

Implements racket dynamics and collision with ball
Including detailed rubber properties

Coordinate System: Y-up (X horizontal, Y height, Z horizontal)
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any
from .parameters import PhysicsParameters, RubberParameters, RubberType


class Racket:
    def __init__(
        self,
        params: PhysicsParameters,
        position: np.ndarray = None,
        orientation: np.ndarray = None,
        side: int = 1,
        is_forehand: bool = True
    ):
        self.params = params
        self.side = side
        self.is_forehand = is_forehand
        
        default_x = side * (params.table_length / 2 + 0.5)
        # Y-up: [x, height, z]
        self.position = np.array(position, dtype=float) if position is not None else np.array([default_x, 1.0, 0.0])
        
        default_normal = np.array([-side, 0.0, 0.0], dtype=float)
        self.orientation = np.array(orientation, dtype=float) if orientation is not None else default_normal
        self.orientation = self.orientation / np.linalg.norm(self.orientation)
        
        self.velocity = np.zeros(3)
        self.angular_velocity = np.zeros(3)
        
        self.radius_major = params.blade.length / 2
        self.radius_minor = params.blade.width / 2
        
        self.trajectory = [self.position.copy()]
        self.orientation_history = [self.orientation.copy()]
        self.velocity_history = [self.velocity.copy()]
        self.time_stamps = [0.0]

    @property
    def rubber(self) -> RubberParameters:
        return self.params.get_rubber_by_side(self.is_forehand)

    def update_position(self, position: np.ndarray, orientation: Optional[np.ndarray] = None, dt: Optional[float] = None):
        if dt is None:
            dt = self.params.dt
        self.velocity = (np.array(position) - self.position) / dt
        self.position = np.array(position, dtype=float)
        if orientation is not None:
            self.orientation = np.array(orientation) / np.linalg.norm(orientation)
        self.trajectory.append(self.position.copy())
        self.orientation_history.append(self.orientation.copy())
        self.velocity_history.append(self.velocity.copy())
        if len(self.time_stamps) > 0:
            self.time_stamps.append(self.time_stamps[-1] + dt)
        else:
            self.time_stamps.append(dt)

    def check_collision(self, ball_position: np.ndarray, ball_radius: float) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray]]:
        diff = ball_position - self.position
        normal_distance = abs(np.dot(diff, self.orientation))
        projection = diff - np.dot(diff, self.orientation) * self.orientation
        planar_distance = np.linalg.norm(projection)
        
        if normal_distance < ball_radius + 0.005 and planar_distance < self.radius_major:
            contact_point = self.position + projection
            normal = self.orientation.copy()
            if np.dot(diff, self.orientation) < 0:
                normal = -normal
            return True, contact_point, normal
        return False, None, None

    def compute_impact(self, ball_velocity: np.ndarray, ball_spin: np.ndarray, contact_point: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        rubber = self.rubber
        relative_velocity = ball_velocity - self.velocity
        
        v_normal = np.dot(relative_velocity, self.orientation) * self.orientation
        v_tangent = relative_velocity - v_normal
        
        # Restitution with rubber properties
        effective_restitution = (rubber.restitution + self.params.blade.base_restitution) / 2
        v_normal_new = -effective_restitution * v_normal
        
        # Friction and spin from rubber
        friction = rubber.dynamic_friction
        spin_coeff = rubber.spin_coefficient
        
        # Spin transfer
        spin_axis = np.cross(self.orientation, v_tangent)
        spin_axis_norm = np.linalg.norm(spin_axis)
        if spin_axis_norm > 1e-6:
            spin_axis = spin_axis / spin_axis_norm
            spin_magnitude = np.linalg.norm(v_tangent) * friction * spin_coeff / self.params.ball_radius
            spin_change = spin_axis * spin_magnitude * self.params.spin.spin_transfer_efficiency
        else:
            spin_change = np.zeros(3)
        
        # Handle incoming spin (rubber sensitivity)
        incoming_spin_effect = np.cross(ball_spin, self.orientation) * self.params.ball_radius
        incoming_spin_effect *= rubber.spin_sensitivity
        
        # Energy absorption
        energy_factor = 1.0 - rubber.energy_absorption
        
        # Final velocity and spin
        new_velocity = self.velocity + v_normal_new * energy_factor + v_tangent * (1 - friction * 0.3) + incoming_spin_effect * 0.1
        new_spin = ball_spin * self.params.spin.collision_spin_retention + spin_change
        
        # Add racket swing contribution to spin
        racket_speed = np.linalg.norm(self.velocity)
        if racket_speed > 0.1:
            swing_spin = np.cross(self.orientation, self.velocity / racket_speed) * racket_speed * spin_coeff * 50
            new_spin += swing_spin
        
        return new_velocity, new_spin

    def get_motion_data(self) -> Dict[str, np.ndarray]:
        return {
            "trajectory": np.array(self.trajectory),
            "orientation": np.array(self.orientation_history),
            "velocity": np.array(self.velocity_history),
            "time_stamps": np.array(self.time_stamps)
        }

    def reset(self, position: Optional[np.ndarray] = None, orientation: Optional[np.ndarray] = None):
        if position is not None:
            self.position = np.array(position, dtype=float)
        if orientation is not None:
            self.orientation = np.array(orientation) / np.linalg.norm(orientation)
        self.velocity = np.zeros(3)
        self.trajectory = [self.position.copy()]
        self.orientation_history = [self.orientation.copy()]
        self.velocity_history = [self.velocity.copy()]
        self.time_stamps = [0.0]

    def __repr__(self) -> str:
        return f"Racket(pos={self.position}, side={self.side}, rubber={self.rubber.rubber_type.value})"
