"""
Game object system for multiple balls and rackets.

All positions use Y-up coordinate system internally.
Rendering converts to Z-up for OpenGL.
"""

import numpy as np
import math
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from enum import Enum
import uuid


class EntityType(Enum):
    BALL = "ball"
    RACKET = "racket"
    TABLE = "table"


class RubberType(Enum):
    """4 types of rubber surfaces"""
    INVERTED = "inverted"      # 裏ソフト - smooth surface, high spin
    PIMPLES = "pimples"        # 表ソフト - short pimples, speed oriented
    LONG_PIMPLES = "long_pimples"  # 粒高 - long pimples, spin reversal
    ANTI = "anti"              # アンチ - anti-spin, low friction


# Default properties for each rubber type
RUBBER_DEFAULTS = {
    RubberType.INVERTED: {
        "friction": 0.9,
        "spin_coefficient": 1.2,
        "restitution": 0.85,
        "spin_reversal": 0.0,  # No spin reversal
    },
    RubberType.PIMPLES: {
        "friction": 0.7,
        "spin_coefficient": 0.7,
        "restitution": 0.90,
        "spin_reversal": 0.1,  # Slight spin disruption
    },
    RubberType.LONG_PIMPLES: {
        "friction": 0.5,
        "spin_coefficient": 0.3,
        "restitution": 0.75,
        "spin_reversal": 0.8,  # High spin reversal
    },
    RubberType.ANTI: {
        "friction": 0.3,
        "spin_coefficient": 0.1,
        "restitution": 0.70,
        "spin_reversal": 0.5,  # Moderate spin absorption
    },
}


@dataclass
class RubberSideData:
    """Rubber properties for one side of racket"""
    rubber_type: RubberType = RubberType.INVERTED
    friction: float = 0.9
    spin_coefficient: float = 1.2
    restitution: float = 0.85
    spin_reversal: float = 0.0  # 0=normal, 1=full reversal (for long pimples)

    @classmethod
    def from_type(cls, rubber_type: RubberType) -> 'RubberSideData':
        """Create rubber data with defaults for given type"""
        defaults = RUBBER_DEFAULTS[rubber_type]
        return cls(
            rubber_type=rubber_type,
            friction=defaults["friction"],
            spin_coefficient=defaults["spin_coefficient"],
            restitution=defaults["restitution"],
            spin_reversal=defaults["spin_reversal"],
        )


@dataclass
class VelocityData:
    """Velocity with rotation-based definition"""
    vector: np.ndarray = field(default_factory=lambda: np.zeros(3))
    # For racket swing
    acceleration: np.ndarray = field(default_factory=lambda: np.zeros(3))


@dataclass
class GameEntity:
    """Base class for game entities"""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    entity_type: EntityType = EntityType.BALL
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))  # Y-up
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    rotation: np.ndarray = field(default_factory=lambda: np.zeros(2))  # [yaw, pitch]
    active: bool = False  # Whether simulation is running for this entity


@dataclass
class BallEntity(GameEntity):
    """Ball entity with physics properties"""
    entity_type: EntityType = EntityType.BALL
    spin: np.ndarray = field(default_factory=lambda: np.zeros(3))  # rad/s
    radius: float = 0.02  # 40mm diameter
    mass: float = 0.0027  # 2.7g
    trail: List[np.ndarray] = field(default_factory=list)
    bounce_count: int = 0
    # Orientation (angle-axis representation)
    orientation_angle: float = 0.0  # Rotation angle in radians
    orientation_axis: np.ndarray = field(default_factory=lambda: np.array([0, 1, 0]))  # Default: Y-up


@dataclass
class RacketEntity(GameEntity):
    """Racket entity with swing properties"""
    entity_type: EntityType = EntityType.RACKET
    acceleration: np.ndarray = field(default_factory=lambda: np.zeros(3))
    mass: float = 0.18  # 180g typical
    # Rubber for each side (red = forehand, black = backhand typically)
    rubber_red: RubberSideData = field(default_factory=RubberSideData)
    rubber_black: RubberSideData = field(default_factory=RubberSideData)
    # Coefficient [red, black] - friction coefficients
    coefficient: List[float] = field(default_factory=lambda: [0.9, 0.9])
    # Restitution [red, black] - bounce coefficients
    restitution: List[float] = field(default_factory=lambda: [0.85, 0.85])
    orientation_angle: float = 0.0  # Rotation angle in radians
    orientation_axis: np.ndarray = field(default_factory=lambda: np.array([0, 1, 0]))  # Rotation axis
    # Swing state
    swing_active: bool = False
    swing_time: float = 0.0


@dataclass
class TableEntity(GameEntity):
    """Table entity with physics properties"""
    entity_type: EntityType = EntityType.TABLE
    # Table dimensions (ITTF standard: 2.74m x 1.525m, height 0.76m)
    length: float = 2.74  # X direction
    width: float = 1.525  # Z direction
    height: float = 0.76  # Surface height (Y)
    thickness: float = 0.03  # Table top thickness
    net_height: float = 0.1525  # Net height above table
    # Physics properties
    mass: float = 100.0  # kg (heavy, essentially immovable)
    restitution: float = 0.85  # Bounce coefficient
    coefficient: float = 0.4  # Surface friction coefficient
    # Orientation (angle-axis)
    orientation_angle: float = 0.0
    orientation_axis: np.ndarray = field(default_factory=lambda: np.array([0, 1, 0]))


class EntityManager:
    """Manages all game entities"""

    def __init__(self):
        self.entities: Dict[str, GameEntity] = {}
        self.balls: List[BallEntity] = []
        self.rackets: List[RacketEntity] = []
        self.tables: List[TableEntity] = []
        self.simulation_running: bool = False

    def summon(self, entity_type: str, position: np.ndarray, nbt: Dict[str, Any]) -> GameEntity:
        """
        Create and register a new entity.

        Args:
            entity_type: "ball" or "racket"
            position: [x, y, z] in Y-up coordinates
            nbt: NBT data with properties

        Returns:
            Created entity
        """
        if entity_type == "ball":
            entity = self._create_ball(position, nbt)
            self.balls.append(entity)
        elif entity_type == "racket":
            entity = self._create_racket(position, nbt)
            self.rackets.append(entity)
        elif entity_type == "table":
            entity = self._create_table(position, nbt)
            self.tables.append(entity)
        else:
            raise ValueError(f"Unknown entity type: {entity_type}")

        self.entities[entity.id] = entity
        return entity

    def _create_ball(self, position: np.ndarray, nbt: Dict[str, Any]) -> BallEntity:
        """Create a ball entity from NBT data"""
        ball = BallEntity(position=position.copy())

        # Velocity
        if 'velocity' in nbt:
            if isinstance(nbt['velocity'], np.ndarray):
                ball.velocity = nbt['velocity'].copy()
            elif isinstance(nbt['velocity'], list):
                ball.velocity = np.array(nbt['velocity'], dtype=float)

        # Spin
        if 'spin' in nbt:
            if isinstance(nbt['spin'], list):
                ball.spin = np.array(nbt['spin'], dtype=float)
            elif isinstance(nbt['spin'], dict):
                # Handle RPM format: {rpm: 3000, axis: [0, 1, 0]}
                rpm = nbt['spin'].get('rpm', 0)
                axis = np.array(nbt['spin'].get('axis', [0, 1, 0]), dtype=float)
                axis = axis / np.linalg.norm(axis) if np.linalg.norm(axis) > 0 else axis
                ball.spin = axis * (rpm * 2 * math.pi / 60)

        # Mass (optional)
        if 'mass' in nbt:
            ball.mass = float(nbt['mass'])

        # Radius (optional)
        if 'radius' in nbt:
            ball.radius = float(nbt['radius'])

        # Orientation (rotation: {angle:..., axis:...})
        if 'rotation' in nbt:
            rot = nbt['rotation']
            if isinstance(rot, dict):
                ball.orientation_angle = float(rot.get('angle', 0))
                axis = rot.get('axis', [0, 1, 0])
                if isinstance(axis, np.ndarray):
                    ball.orientation_axis = axis.copy()
                else:
                    ball.orientation_axis = np.array(axis, dtype=float)
                norm = np.linalg.norm(ball.orientation_axis)
                if norm > 0:
                    ball.orientation_axis = ball.orientation_axis / norm

        return ball

    def _create_racket(self, position: np.ndarray, nbt: Dict[str, Any]) -> RacketEntity:
        """Create a racket entity from NBT data"""
        racket = RacketEntity(position=position.copy())

        # Velocity
        if 'velocity' in nbt:
            if isinstance(nbt['velocity'], np.ndarray):
                racket.velocity = nbt['velocity'].copy()
            elif isinstance(nbt['velocity'], list):
                racket.velocity = np.array(nbt['velocity'], dtype=float)

        # Acceleration
        if 'acceleration' in nbt:
            if isinstance(nbt['acceleration'], np.ndarray):
                racket.acceleration = nbt['acceleration'].copy()
            elif isinstance(nbt['acceleration'], list):
                racket.acceleration = np.array(nbt['acceleration'], dtype=float)

        # Mass
        if 'mass' in nbt:
            racket.mass = float(nbt['mass'])

        # Rubber properties for red side
        if 'rubber_red' in nbt:
            rubber_data = nbt['rubber_red']
            rubber_type_str = rubber_data.get('type', 'inverted')
            rubber_type = self._parse_rubber_type(rubber_type_str)
            racket.rubber_red = RubberSideData.from_type(rubber_type)
        elif 'rubber' in nbt:
            # Legacy support: single rubber for both sides
            rubber_data = nbt['rubber']
            if isinstance(rubber_data, str):
                rubber_type = self._parse_rubber_type(rubber_data)
                racket.rubber_red = RubberSideData.from_type(rubber_type)
            elif isinstance(rubber_data, list) and len(rubber_data) >= 2:
                # [red_type, black_type]
                racket.rubber_red = RubberSideData.from_type(self._parse_rubber_type(rubber_data[0]))
                racket.rubber_black = RubberSideData.from_type(self._parse_rubber_type(rubber_data[1]))

        # Rubber properties for black side
        if 'rubber_black' in nbt:
            rubber_data = nbt['rubber_black']
            rubber_type_str = rubber_data.get('type', 'inverted')
            rubber_type = self._parse_rubber_type(rubber_type_str)
            racket.rubber_black = RubberSideData.from_type(rubber_type)

        # Coefficient [red, black] friction
        if 'coefficient' in nbt:
            coeff = nbt['coefficient']
            if isinstance(coeff, list) and len(coeff) >= 2:
                racket.coefficient = [float(coeff[0]), float(coeff[1])]
            elif isinstance(coeff, (int, float)):
                racket.coefficient = [float(coeff), float(coeff)]

        # Restitution [red, black] bounce coefficients
        if 'restitution' in nbt:
            rest = nbt['restitution']
            if isinstance(rest, list) and len(rest) >= 2:
                racket.restitution = [float(rest[0]), float(rest[1])]
            elif isinstance(rest, (int, float)):
                racket.restitution = [float(rest), float(rest)]

        # Rotation (angle + axis)
        if 'rotation' in nbt:
            rot = nbt['rotation']
            if isinstance(rot, dict):
                racket.orientation_angle = float(rot.get('angle', 0))
                axis = rot.get('axis', [0, 1, 0])
                racket.orientation_axis = np.array(axis, dtype=float)
                norm = np.linalg.norm(racket.orientation_axis)
                if norm > 0:
                    racket.orientation_axis = racket.orientation_axis / norm

        return racket

    def _create_table(self, position: np.ndarray, nbt: Dict[str, Any]) -> TableEntity:
        """Create a table entity from NBT data"""
        table = TableEntity(position=position.copy())

        # Dimensions
        if 'length' in nbt:
            table.length = float(nbt['length'])
        if 'width' in nbt:
            table.width = float(nbt['width'])
        if 'height' in nbt:
            table.height = float(nbt['height'])
        if 'thickness' in nbt:
            table.thickness = float(nbt['thickness'])
        if 'net_height' in nbt:
            table.net_height = float(nbt['net_height'])

        # Physics
        if 'mass' in nbt:
            table.mass = float(nbt['mass'])
        if 'restitution' in nbt:
            table.restitution = float(nbt['restitution'])
        if 'coefficient' in nbt:
            table.coefficient = float(nbt['coefficient'])
        elif 'friction' in nbt:  # Legacy support
            table.coefficient = float(nbt['friction'])

        # Rotation (angle + axis)
        if 'rotation' in nbt:
            rot = nbt['rotation']
            if isinstance(rot, dict):
                table.orientation_angle = float(rot.get('angle', 0))
                axis = rot.get('axis', [0, 1, 0])
                table.orientation_axis = np.array(axis, dtype=float)
                norm = np.linalg.norm(table.orientation_axis)
                if norm > 0:
                    table.orientation_axis = table.orientation_axis / norm

        return table

    def _parse_rubber_type(self, type_str: str) -> RubberType:
        """Parse rubber type string to enum"""
        type_map = {
            'inverted': RubberType.INVERTED,
            'pimples': RubberType.PIMPLES,
            'short_pimples': RubberType.PIMPLES,
            'long_pimples': RubberType.LONG_PIMPLES,
            'long': RubberType.LONG_PIMPLES,
            'anti': RubberType.ANTI,
        }
        return type_map.get(type_str.lower(), RubberType.INVERTED)

    def kill(self, selector: str) -> int:
        """
        Remove entities matching selector.

        Args:
            selector: @e (all), @e[type=ball], entity_id, etc.

        Returns:
            Number of entities killed
        """
        count = 0

        if selector == '@e':
            # Kill all
            count = len(self.entities)
            self.entities.clear()
            self.balls.clear()
            self.rackets.clear()
            self.tables.clear()
        elif selector.startswith('@e[type='):
            # Kill by type
            type_match = selector[8:-1]  # Extract type from @e[type=X]
            to_remove = [e for e in self.entities.values()
                        if e.entity_type.value == type_match]
            for entity in to_remove:
                self._remove_entity(entity)
                count += 1
        else:
            # Kill by ID
            if selector in self.entities:
                self._remove_entity(self.entities[selector])
                count = 1

        return count

    def _remove_entity(self, entity: GameEntity):
        """Remove entity from all lists"""
        if entity.id in self.entities:
            del self.entities[entity.id]
        if isinstance(entity, BallEntity) and entity in self.balls:
            self.balls.remove(entity)
        if isinstance(entity, RacketEntity) and entity in self.rackets:
            self.rackets.remove(entity)
        if isinstance(entity, TableEntity) and entity in self.tables:
            self.tables.remove(entity)

    def start(self):
        """Start simulation for all entities"""
        self.simulation_running = True
        for entity in self.entities.values():
            entity.active = True
            if isinstance(entity, RacketEntity):
                entity.swing_active = True
                entity.swing_time = 0.0

    def stop(self):
        """Stop simulation"""
        self.simulation_running = False
        for entity in self.entities.values():
            entity.active = False

    def update(self, dt: float, physics_params):
        """
        Update all entities for one time step.

        Args:
            dt: Time step in seconds
            physics_params: Physics parameters from game
        """
        if not self.simulation_running:
            return

        # Update rackets (swing motion)
        for racket in self.rackets:
            if racket.active and racket.swing_active:
                self._update_racket(racket, dt)

        # Update balls
        for ball in self.balls:
            if ball.active:
                self._update_ball(ball, dt, physics_params)

        # Check collisions
        self._check_collisions(physics_params)

    def _update_racket(self, racket: RacketEntity, dt: float):
        """Update racket position during swing"""
        racket.swing_time += dt

        # Apply acceleration to velocity
        racket.velocity = racket.velocity + racket.acceleration * dt

        # Update position
        racket.position = racket.position + racket.velocity * dt

    def _update_ball(self, ball: BallEntity, dt: float, params):
        """Update ball physics (simplified)"""
        # Gravity (Y is up)
        gravity = np.array([0, -params.gravity, 0])

        # Get air properties from params
        air_density = params.air_density
        drag_coeff = params.air_drag_coeff

        # Air drag
        speed = np.linalg.norm(ball.velocity)
        if speed > 0:
            drag_force = -0.5 * air_density * drag_coeff * \
                        (math.pi * ball.radius ** 2) * speed * ball.velocity
            drag_accel = drag_force / ball.mass
        else:
            drag_accel = np.zeros(3)

        # Magnus effect
        if speed > 0 and np.linalg.norm(ball.spin) > 0:
            # Simplified Magnus
            omega = np.linalg.norm(ball.spin)
            S = omega * ball.radius / speed  # Spin parameter
            Cl = 1.0 / (2.0 + 1.0 / S) if S > 0 else 0

            # Lift direction: perpendicular to velocity and spin
            lift_dir = np.cross(ball.spin, ball.velocity)
            if np.linalg.norm(lift_dir) > 0:
                lift_dir = lift_dir / np.linalg.norm(lift_dir)
                lift_force = 0.5 * air_density * Cl * \
                            (math.pi * ball.radius ** 2) * speed ** 2 * lift_dir
                magnus_accel = lift_force / ball.mass
            else:
                magnus_accel = np.zeros(3)
        else:
            magnus_accel = np.zeros(3)

        # Total acceleration
        accel = gravity + drag_accel + magnus_accel

        # Update velocity and position
        ball.velocity = ball.velocity + accel * dt
        ball.position = ball.position + ball.velocity * dt

        # Add to trail
        ball.trail.append(ball.position.copy())
        if len(ball.trail) > 150:
            ball.trail.pop(0)

        # Spin decay
        spin_decay_factor = params.get_spin_decay_factor(dt)
        ball.spin = ball.spin * spin_decay_factor

    def _check_collisions(self, params):
        """Check and handle collisions"""
        table_height = params.table_height

        for ball in self.balls:
            if not ball.active:
                continue

            # Ground/table collision (Y is height)
            if ball.position[1] < ball.radius:
                ball.position[1] = ball.radius
                ball.velocity[1] = -ball.velocity[1] * 0.8
                ball.bounce_count += 1

            # Table surface collision
            hl = params.table_length / 2
            hw = params.table_width / 2

            if (abs(ball.position[0]) < hl and
                abs(ball.position[2]) < hw and
                ball.position[1] < table_height + ball.radius and
                ball.velocity[1] < 0):
                ball.position[1] = table_height + ball.radius
                ball.velocity[1] = -ball.velocity[1] * 0.85
                ball.bounce_count += 1

            # Ball-Racket collision
            for racket in self.rackets:
                self._check_ball_racket_collision(ball, racket)

            # Out of bounds
            if ball.position[1] < -1 or np.linalg.norm(ball.position) > 10:
                ball.active = False

    def _check_ball_racket_collision(self, ball: BallEntity, racket: RacketEntity):
        """Check and handle ball-racket collision"""
        # Racket dimensions
        blade_width = 0.15   # X direction
        blade_length = 0.16  # Z direction
        blade_thick = 0.01   # Y direction (including rubber)

        # Get ball position relative to racket
        rel_pos = ball.position - racket.position

        # Apply inverse rotation to get position in racket's local frame
        angle = racket.orientation_angle
        axis = racket.orientation_axis

        if abs(angle) > 1e-6:
            # Inverse rotation using Rodrigues' formula with -angle
            k = axis
            v = rel_pos
            cos_a = math.cos(-angle)
            sin_a = math.sin(-angle)
            local_pos = v * cos_a + np.cross(k, v) * sin_a + k * np.dot(k, v) * (1 - cos_a)
        else:
            local_pos = rel_pos

        # Check if ball is within racket bounds (elliptical blade in XZ plane)
        # Local coords: X = width, Y = thickness (up), Z = length
        x_norm = local_pos[0] / (blade_width / 2)
        z_norm = local_pos[2] / (blade_length / 2)
        in_ellipse = (x_norm ** 2 + z_norm ** 2) <= 1.0

        # Check Y distance (thickness)
        y_dist = abs(local_pos[1])
        collision_dist = blade_thick / 2 + ball.radius

        if in_ellipse and y_dist < collision_dist:
            # Collision detected!
            # Determine which side was hit (red = +Y, black = -Y in local)
            is_red_side = local_pos[1] > 0

            # Get rubber properties
            if is_red_side:
                rubber = racket.rubber_red
                restitution = racket.restitution[0]
                friction = racket.coefficient[0]
            else:
                rubber = racket.rubber_black
                restitution = racket.restitution[1]
                friction = racket.coefficient[1]

            # Calculate normal in world coordinates
            local_normal = np.array([0.0, 1.0 if is_red_side else -1.0, 0.0])

            if abs(angle) > 1e-6:
                # Rotate normal to world frame
                k = axis
                v = local_normal
                cos_a = math.cos(angle)
                sin_a = math.sin(angle)
                world_normal = v * cos_a + np.cross(k, v) * sin_a + k * np.dot(k, v) * (1 - cos_a)
            else:
                world_normal = local_normal

            # Relative velocity (ball relative to racket)
            rel_vel = ball.velocity - racket.velocity

            # Velocity component along normal
            vel_normal = np.dot(rel_vel, world_normal)

            # Only process if ball is approaching the surface
            if vel_normal < 0:
                # Reflect velocity
                ball.velocity = ball.velocity - (1 + restitution) * vel_normal * world_normal

                # Add racket velocity contribution
                ball.velocity = ball.velocity + racket.velocity * 0.8

                # Apply spin from rubber friction
                # Tangential velocity
                vel_tangent = rel_vel - vel_normal * world_normal
                if np.linalg.norm(vel_tangent) > 0:
                    # Add spin based on tangential velocity and friction
                    spin_axis = np.cross(world_normal, vel_tangent)
                    if np.linalg.norm(spin_axis) > 0:
                        spin_axis = spin_axis / np.linalg.norm(spin_axis)
                        spin_magnitude = np.linalg.norm(vel_tangent) * friction * 50  # Spin factor
                        ball.spin = ball.spin + spin_axis * spin_magnitude

                # Push ball out of racket
                penetration = collision_dist - y_dist
                ball.position = ball.position + world_normal * (penetration + 0.001)

    def get_entity(self, entity_id: str) -> Optional[GameEntity]:
        """Get entity by ID"""
        return self.entities.get(entity_id)

    def get_all_balls(self) -> List[BallEntity]:
        """Get all ball entities"""
        return self.balls.copy()

    def get_all_rackets(self) -> List[RacketEntity]:
        """Get all racket entities"""
        return self.rackets.copy()
