"""
Physical parameters for table tennis simulation

All parameters can be adjusted for research purposes.
Based on real-world physics and official ITTF regulations.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from enum import Enum


class RubberType(Enum):
    INVERTED = "inverted"
    PIMPLES_OUT = "pimples_out"
    LONG_PIMPLES = "long_pimples"
    ANTI_SPIN = "anti_spin"


class SpinType(Enum):
    TOPSPIN = "topspin"
    BACKSPIN = "backspin"
    SIDESPIN_LEFT = "sidespin_left"
    SIDESPIN_RIGHT = "sidespin_right"
    NO_SPIN = "no_spin"


@dataclass
class RubberParameters:
    rubber_type: RubberType = RubberType.INVERTED
    thickness: float = 0.002
    sponge_thickness: float = 0.002
    total_thickness: float = 0.004
    mass: float = 0.045
    static_friction: float = 1.2
    dynamic_friction: float = 0.9
    restitution: float = 0.85
    spin_coefficient: float = 1.0
    spin_sensitivity: float = 0.8
    hardness: float = 40.0
    energy_absorption: float = 0.15

    @classmethod
    def create_inverted_offensive(cls):
        return cls(rubber_type=RubberType.INVERTED, thickness=0.002, sponge_thickness=0.0022,
                   mass=0.048, static_friction=1.3, dynamic_friction=1.0, restitution=0.88,
                   spin_coefficient=1.2, spin_sensitivity=0.9, hardness=45.0, energy_absorption=0.12)

    @classmethod
    def create_inverted_control(cls):
        return cls(rubber_type=RubberType.INVERTED, thickness=0.002, sponge_thickness=0.0018,
                   mass=0.042, static_friction=1.1, dynamic_friction=0.85, restitution=0.82,
                   spin_coefficient=1.0, spin_sensitivity=0.85, hardness=38.0, energy_absorption=0.18)

    @classmethod
    def create_pimples_out(cls):
        return cls(rubber_type=RubberType.PIMPLES_OUT, thickness=0.0015, sponge_thickness=0.0018,
                   mass=0.040, static_friction=0.7, dynamic_friction=0.5, restitution=0.90,
                   spin_coefficient=0.6, spin_sensitivity=0.5, hardness=42.0, energy_absorption=0.10)

    @classmethod
    def create_long_pimples(cls):
        return cls(rubber_type=RubberType.LONG_PIMPLES, thickness=0.0015, sponge_thickness=0.001,
                   mass=0.035, static_friction=0.4, dynamic_friction=0.3, restitution=0.75,
                   spin_coefficient=0.3, spin_sensitivity=0.2, hardness=30.0, energy_absorption=0.25)

    @classmethod
    def create_anti_spin(cls):
        return cls(rubber_type=RubberType.ANTI_SPIN, thickness=0.0015, sponge_thickness=0.0015,
                   mass=0.038, static_friction=0.2, dynamic_friction=0.15, restitution=0.70,
                   spin_coefficient=0.1, spin_sensitivity=0.1, hardness=35.0, energy_absorption=0.30)


@dataclass
class BladeParameters:
    mass: float = 0.085
    length: float = 0.157
    width: float = 0.150
    thickness: float = 0.006
    stiffness: float = 0.8
    base_restitution: float = 0.75
    vibration_damping: float = 0.3

    @classmethod
    def create_offensive(cls):
        return cls(mass=0.088, thickness=0.0058, stiffness=0.9, base_restitution=0.82, vibration_damping=0.2)

    @classmethod
    def create_allround(cls):
        return cls(mass=0.085, thickness=0.006, stiffness=0.7, base_restitution=0.75, vibration_damping=0.35)

    @classmethod
    def create_defensive(cls):
        return cls(mass=0.080, thickness=0.0055, stiffness=0.5, base_restitution=0.68, vibration_damping=0.45)


@dataclass
class BallParameters:
    mass: float = 0.0027
    diameter: float = 0.040
    radius: float = 0.020
    material_density: float = 1050.0
    wall_thickness: float = 0.0004
    restitution: float = 0.89
    drag_coefficient: float = 0.45
    lift_coefficient: float = 0.25
    surface_roughness: float = 0.001

    def __post_init__(self):
        self.radius = self.diameter / 2
        self.cross_section_area = np.pi * self.radius ** 2
        self.moment_of_inertia = (2.0 / 3.0) * self.mass * self.radius ** 2

    @classmethod
    def create_competition_ball(cls):
        return cls(mass=0.0027, diameter=0.040, restitution=0.90, drag_coefficient=0.44,
                   lift_coefficient=0.28, surface_roughness=0.0008)

    @classmethod
    def create_training_ball(cls):
        return cls(mass=0.0028, diameter=0.040, restitution=0.87, drag_coefficient=0.46,
                   lift_coefficient=0.23, surface_roughness=0.0012)


@dataclass
class SpinParameters:
    max_spin_rate: float = 1000.0
    air_spin_decay: float = 0.02
    collision_spin_retention: float = 0.7
    spin_transfer_efficiency: float = 0.6
    magnus_coefficient: float = 1.0


@dataclass
class EnvironmentParameters:
    gravity: float = 9.81
    air_density: float = 1.225
    air_viscosity: float = 1.81e-5
    temperature: float = 20.0
    humidity: float = 50.0
    altitude: float = 0.0

    def get_adjusted_air_density(self) -> float:
        T = self.temperature + 273.15
        T0 = 293.15
        altitude_factor = np.exp(-self.altitude / 8500.0)
        temperature_factor = T0 / T
        return self.air_density * altitude_factor * temperature_factor


@dataclass
class TableParameters:
    length: float = 2.74
    width: float = 1.525
    height: float = 0.76
    net_height: float = 0.1525
    net_overhang: float = 0.1525
    surface_restitution: float = 0.89
    surface_friction: float = 0.5
    edge_thickness: float = 0.02


@dataclass
class PhysicsParameters:
    ball: BallParameters = field(default_factory=BallParameters)
    rubber_forehand: RubberParameters = field(default_factory=RubberParameters.create_inverted_offensive)
    rubber_backhand: RubberParameters = field(default_factory=RubberParameters.create_inverted_offensive)
    blade: BladeParameters = field(default_factory=BladeParameters.create_allround)
    spin: SpinParameters = field(default_factory=SpinParameters)
    environment: EnvironmentParameters = field(default_factory=EnvironmentParameters)
    table: TableParameters = field(default_factory=TableParameters)
    dt: float = 0.0005
    max_simulation_time: float = 10.0

    @property
    def ball_mass(self) -> float:
        return self.ball.mass

    @property
    def ball_radius(self) -> float:
        return self.ball.radius

    @property
    def ball_restitution(self) -> float:
        return self.ball.restitution

    @property
    def table_length(self) -> float:
        return self.table.length

    @property
    def table_width(self) -> float:
        return self.table.width

    @property
    def table_height(self) -> float:
        return self.table.height

    @property
    def table_restitution(self) -> float:
        return self.table.surface_restitution

    @property
    def table_friction(self) -> float:
        return self.table.surface_friction

    @property
    def racket_mass(self) -> float:
        return self.blade.mass + self.rubber_forehand.mass + self.rubber_backhand.mass

    @property
    def racket_restitution(self) -> float:
        rubber_effect = (self.rubber_forehand.restitution + self.rubber_backhand.restitution) / 2
        return (self.blade.base_restitution + rubber_effect) / 2

    @property
    def racket_friction(self) -> float:
        return (self.rubber_forehand.dynamic_friction + self.rubber_backhand.dynamic_friction) / 2

    @property
    def gravity(self) -> float:
        return self.environment.gravity

    @property
    def air_density(self) -> float:
        return self.environment.get_adjusted_air_density()

    @property
    def air_drag_coeff(self) -> float:
        return self.ball.drag_coefficient

    def get_magnus_coefficient(self) -> float:
        return self.ball.lift_coefficient * self.spin.magnus_coefficient

    def get_spin_decay_factor(self, dt: float) -> float:
        return np.exp(-self.spin.air_spin_decay * dt)

    def get_rubber_by_side(self, is_forehand: bool) -> RubberParameters:
        return self.rubber_forehand if is_forehand else self.rubber_backhand

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ball_mass": self.ball.mass,
            "ball_radius": self.ball.radius,
            "ball_restitution": self.ball.restitution,
            "table_length": self.table.length,
            "table_width": self.table.width,
            "table_height": self.table.height,
            "racket_mass": self.racket_mass,
            "gravity": self.gravity,
            "air_density": self.air_density,
            "dt": self.dt,
            "max_simulation_time": self.max_simulation_time,
        }

    @classmethod
    def from_dict(cls, params: Dict[str, Any]):
        instance = cls()
        if "ball_mass" in params:
            instance.ball.mass = params["ball_mass"]
        if "dt" in params:
            instance.dt = params["dt"]
        return instance

    def update(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown parameter: {key}")

    def __repr__(self) -> str:
        return f"PhysicsParameters(racket_mass={self.racket_mass:.3f}kg)"


def create_offensive_setup() -> PhysicsParameters:
    params = PhysicsParameters()
    params.blade = BladeParameters.create_offensive()
    params.rubber_forehand = RubberParameters.create_inverted_offensive()
    params.rubber_backhand = RubberParameters.create_inverted_offensive()
    return params


def create_defensive_setup() -> PhysicsParameters:
    params = PhysicsParameters()
    params.blade = BladeParameters.create_defensive()
    params.rubber_forehand = RubberParameters.create_inverted_control()
    params.rubber_backhand = RubberParameters.create_long_pimples()
    return params


def create_allround_setup() -> PhysicsParameters:
    params = PhysicsParameters()
    params.blade = BladeParameters.create_allround()
    params.rubber_forehand = RubberParameters.create_inverted_offensive()
    params.rubber_backhand = RubberParameters.create_inverted_control()
    return params
