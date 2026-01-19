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
    tags: List[str] = field(default_factory=list)  # Tags for selector filtering


@dataclass
class BallEntity(GameEntity):
    """Ball entity with physics properties"""
    entity_type: EntityType = EntityType.BALL
    spin: np.ndarray = field(default_factory=lambda: np.zeros(3))  # rad/s
    radius: float = 0.02  # 40mm diameter
    mass: float = 0.0027  # 2.7g
    trail: List[np.ndarray] = field(default_factory=list)
    bounce_count: int = 0
    # Previous position for swept collision detection
    prev_position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    # Orientation (angle-axis representation)
    orientation_angle: float = 0.0  # Rotation angle in radians
    orientation_axis: np.ndarray = field(default_factory=lambda: np.array([0, 1, 0]))  # Default: Y-up
    # Acceleration (angle-axis + speed format)
    accel_angle: float = 0.0  # Direction angle in radians
    accel_axis: np.ndarray = field(default_factory=lambda: np.array([0, 1, 0]))
    accel_speed: float = 0.0  # Acceleration magnitude (m/s^2)
    # Circular motion (caret notation): [left, up, forward] relative to velocity direction
    # X=left turn, Y=climb, Z=forward acceleration modifier
    circular: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0]))


@dataclass
class RacketEntity(GameEntity):
    """Racket entity with swing properties"""
    entity_type: EntityType = EntityType.RACKET
    # Previous position for swept collision detection
    prev_position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    # Acceleration (angle-axis format)
    accel_angle: float = 0.0
    accel_axis: np.ndarray = field(default_factory=lambda: np.array([0, 1, 0]))
    accel_speed: float = 0.0
    mass: float = 0.18  # 180g typical
    # Rubber for each side (red = forehand, black = backhand typically)
    rubber_red: RubberSideData = field(default_factory=RubberSideData)
    rubber_black: RubberSideData = field(default_factory=RubberSideData)
    # Coefficient [red, black] - friction coefficients
    coefficient: List[float] = field(default_factory=lambda: [0.9, 0.9])
    # Restitution [red, black] - bounce coefficients
    restitution: List[float] = field(default_factory=lambda: [0.85, 0.85])
    orientation_angle: float = 0.0  # Rotation angle in radians (primary rotation)
    orientation_axis: np.ndarray = field(default_factory=lambda: np.array([0, 1, 0]))  # Rotation axis
    # Secondary rotation (applied after primary) - useful for base angle + adjustment
    orientation_angle2: float = 0.0  # Second rotation angle in radians
    orientation_axis2: np.ndarray = field(default_factory=lambda: np.array([0, 1, 0]))  # Second rotation axis
    # Circular motion (caret notation): [left, up, forward] relative to velocity direction
    circular: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0]))
    # Swing state
    swing_active: bool = False
    swing_time: float = 0.0
    # Manual control mode (position controlled externally, not by physics)
    manual_control: bool = False


@dataclass
class TableEntity(GameEntity):
    """Table entity with physics properties"""
    entity_type: EntityType = EntityType.TABLE
    # Table dimensions (ITTF standard: 2.74m x 1.525m, height 0.76m)
    length: float = 2.74  # X direction
    width: float = 1.525  # Z direction
    height: float = 0.76  # Surface height (Y)
    thickness: float = 0.03  # Table top thickness
    # Net properties (ITTF standard)
    net_height: float = 0.1525  # 15.25cm above table
    net_length: float = 1.83  # 183cm total (extends 15.25cm beyond table each side)
    net_tape_width: float = 0.015  # White tape at top: 15mm
    # Net physics
    net_tape_restitution: float = 0.75  # Top tape: high bounce (0.7-0.8)
    net_tape_friction: float = 0.3  # Low friction - ball slides over
    net_mesh_restitution: float = 0.15  # Mesh: absorbs energy (0.1-0.2)
    net_mesh_friction: float = 0.6  # Higher friction - ball gets caught
    # Table physics properties
    mass: float = 100.0  # kg (heavy, essentially immovable)
    restitution: float = 0.85  # Bounce coefficient
    coefficient: float = 0.4  # Surface friction
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

        # Tags
        if 'Tags' in nbt:
            tags = nbt['Tags']
            if isinstance(tags, list):
                ball.tags = [str(t) for t in tags]

        # Acceleration (angle-axis format: {angle:1.57, axis:[0,1,0], speed:0.1})
        if 'acceleration' in nbt:
            accel = nbt['acceleration']
            if isinstance(accel, dict):
                ball.accel_angle = float(accel.get('angle', 0))
                axis = accel.get('axis', [0, 1, 0])
                if isinstance(axis, np.ndarray):
                    ball.accel_axis = axis.copy()
                else:
                    ball.accel_axis = np.array(axis, dtype=float)
                norm = np.linalg.norm(ball.accel_axis)
                if norm > 0:
                    ball.accel_axis = ball.accel_axis / norm
                ball.accel_speed = float(accel.get('speed', 0))

        # Circular motion (caret notation: [left, up, forward])
        if 'circular' in nbt:
            circ = nbt['circular']
            if isinstance(circ, list):
                ball.circular = np.array(circ, dtype=float)
            elif isinstance(circ, np.ndarray):
                ball.circular = circ.copy()

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

        # Acceleration (angle-axis format)
        if 'acceleration' in nbt:
            accel = nbt['acceleration']
            if isinstance(accel, dict):
                racket.accel_angle = float(accel.get('angle', 0))
                axis = accel.get('axis', [0, 1, 0])
                if isinstance(axis, np.ndarray):
                    racket.accel_axis = axis.copy()
                else:
                    racket.accel_axis = np.array(axis, dtype=float)
                norm = np.linalg.norm(racket.accel_axis)
                if norm > 0:
                    racket.accel_axis = racket.accel_axis / norm
                racket.accel_speed = float(accel.get('speed', 0))

        # Circular motion (caret notation: [left, up, forward])
        if 'circular' in nbt:
            circ = nbt['circular']
            if isinstance(circ, list):
                racket.circular = np.array(circ, dtype=float)
            elif isinstance(circ, np.ndarray):
                racket.circular = circ.copy()

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

        # Rotation (angle + axis) - primary rotation
        if 'rotation' in nbt:
            rot = nbt['rotation']
            if isinstance(rot, dict):
                racket.orientation_angle = float(rot.get('angle', 0))
                axis = rot.get('axis', [0, 1, 0])
                racket.orientation_axis = np.array(axis, dtype=float)
                norm = np.linalg.norm(racket.orientation_axis)
                if norm > 0:
                    racket.orientation_axis = racket.orientation_axis / norm

        # Secondary rotation (rotation2) - applied after primary
        if 'rotation2' in nbt:
            rot2 = nbt['rotation2']
            if isinstance(rot2, dict):
                racket.orientation_angle2 = float(rot2.get('angle', 0))
                axis2 = rot2.get('axis', [0, 1, 0])
                racket.orientation_axis2 = np.array(axis2, dtype=float)
                norm2 = np.linalg.norm(racket.orientation_axis2)
                if norm2 > 0:
                    racket.orientation_axis2 = racket.orientation_axis2 / norm2

        # Tags
        if 'Tags' in nbt:
            tags = nbt['Tags']
            if isinstance(tags, list):
                racket.tags = [str(t) for t in tags]

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

        # Tags
        if 'Tags' in nbt:
            tags = nbt['Tags']
            if isinstance(tags, list):
                table.tags = [str(t) for t in tags]

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

        Uses sub-stepping to prevent tunnel effect (ball/racket penetration)
        when racket moves fast in play mode.

        Args:
            dt: Time step in seconds
            physics_params: Physics parameters from game
        """
        if not self.simulation_running:
            return

        # Track physics time for collision cooldowns
        if not hasattr(self, '_physics_time'):
            self._physics_time = 0.0

        # Sub-stepping: divide update into smaller steps for accurate collision
        num_substeps = 4
        sub_dt = dt / num_substeps

        # Save initial positions for interpolation
        # For manual control rackets (play mode), we interpolate from prev to current
        racket_start_positions = {}
        racket_end_positions = {}
        for racket in self.rackets:
            if racket.manual_control:
                # In play mode, prev_position and position are set externally
                # We need to interpolate between them
                start_pos = racket.prev_position.copy() if hasattr(racket, 'prev_position') else racket.position.copy()
                racket_start_positions[id(racket)] = start_pos
                racket_end_positions[id(racket)] = racket.position.copy()
                # Reset to start for interpolation
                racket.position = start_pos.copy()

        # Save ball starting positions
        for ball in self.balls:
            ball.prev_position = ball.position.copy()

        # Save non-manual racket positions
        for racket in self.rackets:
            if not racket.manual_control:
                racket.prev_position = racket.position.copy()

        # Sub-stepping loop
        for substep in range(num_substeps):
            self._physics_time += sub_dt
            alpha = (substep + 1) / num_substeps

            # Interpolate manual control racket positions and calculate velocity
            for racket in self.rackets:
                if racket.manual_control and id(racket) in racket_start_positions:
                    racket.prev_position = racket.position.copy()
                    start = racket_start_positions[id(racket)]
                    end = racket_end_positions[id(racket)]
                    racket.position = start + (end - start) * alpha
                    # Calculate velocity from position change for collision response
                    if sub_dt > 0:
                        racket.velocity = (racket.position - racket.prev_position) / sub_dt

            # Update non-manual rackets (swing motion)
            for racket in self.rackets:
                if racket.active and racket.swing_active and not racket.manual_control:
                    self._update_racket(racket, sub_dt)

            # Update balls
            for ball in self.balls:
                if ball.active:
                    ball.prev_position = ball.position.copy()
                    self._update_ball(ball, sub_dt, physics_params)

            # Check collisions at each substep
            self._check_collisions(physics_params)

        # Ensure final positions are correct for manual control rackets
        for racket in self.rackets:
            if racket.manual_control and id(racket) in racket_end_positions:
                racket.position = racket_end_positions[id(racket)]

    def _update_racket(self, racket: RacketEntity, dt: float):
        """Update racket position during swing"""
        # Skip position update if manually controlled (play mode)
        if racket.manual_control:
            return

        racket.swing_time += dt

        # Apply acceleration (angle-axis format) to velocity
        if racket.accel_speed != 0:
            default_dir = np.array([1.0, 0.0, 0.0])
            if abs(racket.accel_angle) > 1e-6:
                k = racket.accel_axis
                v = default_dir
                cos_a = math.cos(racket.accel_angle)
                sin_a = math.sin(racket.accel_angle)
                accel_dir = v * cos_a + np.cross(k, v) * sin_a + k * np.dot(k, v) * (1 - cos_a)
            else:
                accel_dir = default_dir
            accel_vec = accel_dir * racket.accel_speed
            
            # Circular motion (caret notation): modifies acceleration based on velocity direction
            if hasattr(racket, 'circular') and np.linalg.norm(racket.velocity) > 1e-6:
                vel_dir = racket.velocity / np.linalg.norm(racket.velocity)
                
                # Build local coordinate system from velocity direction
                forward = vel_dir
                world_up = np.array([0.0, 1.0, 0.0])
                right = np.cross(forward, world_up)
                if np.linalg.norm(right) < 1e-6:
                    right = np.array([1.0, 0.0, 0.0])
                else:
                    right = right / np.linalg.norm(right)
                up = np.cross(right, forward)
                up = up / np.linalg.norm(up)
                left = -right
                
                circ = racket.circular
                circular_accel = (left * circ[0] + up * circ[1] + forward * circ[2]) * racket.accel_speed
                accel_vec = accel_vec + circular_accel
            
            racket.velocity = racket.velocity + accel_vec * dt

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

        # User-defined acceleration (angle-axis format)
        user_accel = np.zeros(3)
        if ball.accel_speed != 0:
            # Convert angle-axis to direction vector using Rodrigues formula
            default_dir = np.array([1.0, 0.0, 0.0])  # Same as player/velocity default
            if abs(ball.accel_angle) > 1e-6:
                k = ball.accel_axis
                v = default_dir
                cos_a = math.cos(ball.accel_angle)
                sin_a = math.sin(ball.accel_angle)
                accel_dir = v * cos_a + np.cross(k, v) * sin_a + k * np.dot(k, v) * (1 - cos_a)
            else:
                accel_dir = default_dir
            user_accel = accel_dir * ball.accel_speed

        # Circular motion (caret notation): modifies acceleration based on velocity direction
        # circular = [left, up, forward] relative to velocity
        circular_accel = np.zeros(3)
        if hasattr(ball, 'circular') and np.linalg.norm(ball.velocity) > 1e-6:
            vel_dir = ball.velocity / np.linalg.norm(ball.velocity)
            
            # Build local coordinate system from velocity direction
            # Forward = velocity direction
            forward = vel_dir
            
            # Up = world up, adjusted to be perpendicular to forward
            world_up = np.array([0.0, 1.0, 0.0])
            right = np.cross(forward, world_up)
            if np.linalg.norm(right) < 1e-6:
                # Velocity is vertical, use world X as reference
                right = np.array([1.0, 0.0, 0.0])
            else:
                right = right / np.linalg.norm(right)
            up = np.cross(right, forward)
            up = up / np.linalg.norm(up)
            left = -right
            
            # Apply circular motion: [left, up, forward] * accel_speed
            circ = ball.circular
            circular_accel = (left * circ[0] + up * circ[1] + forward * circ[2]) * ball.accel_speed

        # Total acceleration
        accel = gravity + drag_accel + magnus_accel + user_accel + circular_accel

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

            # Ground collision (Y is height)
            if ball.position[1] < ball.radius:
                ball.position[1] = ball.radius
                # Ground has low restitution and high friction
                ball.velocity[1] = -ball.velocity[1] * 0.5

                # Friction with spin-to-roll conversion
                # Contact point velocity: v_contact = v_ball + ω × r
                # r = [0, -radius, 0], so ω × r = [ω_z*r, 0, -ω_x*r]
                ground_friction = 0.6
                r = ball.radius

                # Slip velocity at contact point
                contact_vel_x = ball.velocity[0] + ball.spin[2] * r
                contact_vel_z = ball.velocity[2] - ball.spin[0] * r

                # Friction reduces slip and transfers energy
                # Gradually approach pure rolling condition
                friction_factor = ground_friction * 0.3
                ball.velocity[0] -= contact_vel_x * friction_factor
                ball.velocity[2] -= contact_vel_z * friction_factor

                # Spin also reduced by friction (moment of inertia effect)
                ball.spin[0] += contact_vel_z * friction_factor / r * 0.4
                ball.spin[2] -= contact_vel_x * friction_factor / r * 0.4

                # General damping
                ball.velocity[0] *= 0.85
                ball.velocity[2] *= 0.85
                ball.spin *= 0.9

                ball.bounce_count += 1

            # Rolling on ground (ball resting on ground with low bounce)
            elif ball.position[1] < ball.radius + 0.002 and abs(ball.velocity[1]) < 0.5:
                # Ball is essentially on the ground, apply rolling physics
                ball.position[1] = ball.radius
                ball.velocity[1] = 0  # Stop vertical bouncing

                r = ball.radius
                rolling_friction = 0.02  # Rolling resistance

                # Convert spin to velocity (rolling)
                # Pure rolling: v = ω × r
                target_vel_x = -ball.spin[2] * r
                target_vel_z = ball.spin[0] * r

                # Blend toward rolling condition
                blend = 0.1
                ball.velocity[0] += (target_vel_x - ball.velocity[0]) * blend
                ball.velocity[2] += (target_vel_z - ball.velocity[2]) * blend

                # Apply rolling friction (slows both velocity and spin)
                ball.velocity[0] *= (1 - rolling_friction)
                ball.velocity[2] *= (1 - rolling_friction)
                ball.spin *= (1 - rolling_friction)

            # Table surface collision - check against actual table entities
            for table in self.tables:
                if not table.active:
                    continue

                hl = table.length / 2
                hw = table.width / 2
                th = table.height + table.position[1]  # Table surface Y position
                
                # Ball position relative to table center
                rel_x = ball.position[0] - table.position[0]
                rel_z = ball.position[2] - table.position[2]

                # Check if ball is over the table and hitting from above
                if (abs(rel_x) < hl and
                    abs(rel_z) < hw and
                    ball.position[1] < th + ball.radius and
                    ball.position[1] > th - 0.05 and  # Not too far below
                    ball.velocity[1] < 0):

                    ball.position[1] = th + ball.radius

                    # Use table's restitution coefficient
                    restitution = table.restitution
                    friction = table.coefficient

                    # Normal collision (Y direction)
                    ball.velocity[1] = -ball.velocity[1] * restitution

                    # Friction affects horizontal velocity
                    horizontal_speed = math.sqrt(ball.velocity[0]**2 + ball.velocity[2]**2)
                    if horizontal_speed > 0.01:
                        # Friction force reduces horizontal velocity
                        friction_factor = max(0.85, 1.0 - friction * 0.3)
                        ball.velocity[0] *= friction_factor
                        ball.velocity[2] *= friction_factor

                    # Spin-surface interaction
                    # Top spin increases forward velocity slightly after bounce
                    # Back spin decreases it
                    if np.linalg.norm(ball.spin) > 0.1:
                        # Spin around X axis affects Z velocity (forward/back)
                        # Spin around Z axis affects X velocity (left/right)
                        spin_effect_z = ball.spin[0] * ball.radius * friction * 0.3
                        spin_effect_x = -ball.spin[2] * ball.radius * friction * 0.3
                        ball.velocity[0] += spin_effect_x
                        ball.velocity[2] += spin_effect_z

                        # Surface friction reduces spin
                        spin_reduction = 1.0 - friction * 0.4
                        ball.spin *= spin_reduction

                    ball.bounce_count += 1
                    break  # Only collide with one table

            # Ball-Net collision
            for table in self.tables:
                if not table.active:
                    continue
                self._check_ball_net_collision(ball, table)

            # Ball-Racket collision
            for racket in self.rackets:
                self._check_ball_racket_collision(ball, racket)

        # Ball-Ball collisions (separate loop to avoid modifying list during iteration)
        self._check_ball_ball_collisions()

        # Out of bounds check
        for ball in self.balls:
            if not ball.active:
                continue
            # Out of bounds
            if ball.position[1] < -1 or np.linalg.norm(ball.position) > 10:
                ball.active = False

    def _check_ball_net_collision(self, ball: BallEntity, table: TableEntity):
        """Check and handle ball-net collision"""
        # Net is at center of table (X direction), spans Z direction
        # Net dimensions - relative to table position
        net_x = table.position[0]  # Center of table X
        net_z_half = table.net_length / 2  # 0.915m each side
        net_bottom = table.height + table.position[1]  # Top of table surface
        net_top = net_bottom + table.net_height  # 15.25cm above table
        tape_bottom = net_top - table.net_tape_width  # Where tape starts
        
        # Ball position relative to net
        ball_rel_x = ball.position[0] - net_x
        ball_rel_z = ball.position[2] - table.position[2]
        ball_y = ball.position[1]
        
        # Check if ball is within net Z range
        if abs(ball_rel_z) > net_z_half:
            return
        
        # Check if ball is at net height
        if ball_y - ball.radius > net_top or ball_y + ball.radius < net_bottom:
            return
        
        # Check X collision with net (net is thin, at table center X)
        net_thickness = 0.005  # 5mm effective thickness
        if abs(ball_rel_x) > ball.radius + net_thickness:
            return
        
        # Ball is hitting the net!
        # Determine if hitting tape (top) or mesh (bottom)
        ball_center_y = ball_y
        
        if ball_center_y >= tape_bottom:
            # Hitting the white tape (top part)
            restitution = table.net_tape_restitution  # 0.75
            friction = table.net_tape_friction  # 0.3
            # Tape can deflect ball over - more elastic
        else:
            # Hitting the mesh (bottom part)
            restitution = table.net_mesh_restitution  # 0.15
            friction = table.net_mesh_friction  # 0.6
            # Mesh absorbs energy - ball drops
        
        # Collision normal is in X direction (perpendicular to net)
        if ball.velocity[0] > 0:
            normal = np.array([-1.0, 0.0, 0.0])  # Ball moving +X, push back -X
        else:
            normal = np.array([1.0, 0.0, 0.0])   # Ball moving -X, push back +X
        
        # Decompose velocity
        vel_normal = np.dot(ball.velocity, normal)
        
        # Only process if ball is moving into the net
        if vel_normal >= 0:
            return
        
        vel_normal_vec = vel_normal * normal
        vel_tangent = ball.velocity - vel_normal_vec
        
        # Apply restitution to normal component
        ball.velocity = -restitution * vel_normal_vec + vel_tangent * (1 - friction * 0.3)
        
        # Push ball out of net
        if ball_rel_x > 0:
            ball.position[0] = net_x + ball.radius + net_thickness
        else:
            ball.position[0] = net_x - (ball.radius + net_thickness)
        
        # Spin interaction - net can affect spin
        ball.spin *= (1 - friction * 0.2)
        
        # If hitting mesh with low speed, ball might just drop
        if ball_y < tape_bottom:
            speed = np.linalg.norm(ball.velocity)
            if speed < 1.0:
                # Ball gets caught in mesh, just drops
                ball.velocity[0] *= 0.1
                ball.velocity[2] *= 0.3

    def _check_ball_ball_collisions(self):
        """Check and handle ball-ball collisions"""
        balls = [b for b in self.balls if b.active]
        n = len(balls)
        
        for i in range(n):
            for j in range(i + 1, n):
                ball1 = balls[i]
                ball2 = balls[j]
                
                # Vector from ball1 to ball2
                delta = ball2.position - ball1.position
                dist = np.linalg.norm(delta)
                min_dist = ball1.radius + ball2.radius
                
                if dist < min_dist and dist > 0:
                    # Collision detected - elastic collision
                    normal = delta / dist
                    
                    # Relative velocity
                    rel_vel = ball1.velocity - ball2.velocity
                    vel_along_normal = np.dot(rel_vel, normal)
                    
                    # Only resolve if balls are approaching
                    if vel_along_normal > 0:
                        # Coefficient of restitution for ball-ball collision
                        e = 0.9
                        
                        # Impulse magnitude (equal mass assumption)
                        m1, m2 = ball1.mass, ball2.mass
                        impulse = (-(1 + e) * vel_along_normal) / (1/m1 + 1/m2)
                        
                        # Apply impulse
                        ball1.velocity = ball1.velocity + (impulse / m1) * normal
                        ball2.velocity = ball2.velocity - (impulse / m2) * normal
                        
                        # Separate balls to prevent overlap
                        overlap = min_dist - dist
                        separation = normal * (overlap / 2 + 0.001)
                        ball1.position = ball1.position - separation
                        ball2.position = ball2.position + separation

    def _check_ball_racket_collision(self, ball: BallEntity, racket: RacketEntity):
        """Check and handle ball-racket collision using capture-and-release method"""
        # Check spawn cooldown - newly spawned balls (e.g. serve toss) need time before collision
        spawn_cooldown = 0.25  # 250ms after spawn before collision is allowed
        spawn_time = getattr(ball, 'spawn_time', -1)
        if hasattr(self, '_physics_time') and spawn_time >= 0:
            if self._physics_time - spawn_time < spawn_cooldown:
                return  # Skip - ball just spawned (serve toss rising)

        # Check collision cooldown to prevent double hits
        cooldown_time = 0.15  # 150ms cooldown between hits
        current_time = getattr(ball, 'last_racket_hit_time', -1)
        if hasattr(self, '_physics_time') and current_time > 0:
            if self._physics_time - current_time < cooldown_time:
                return  # Skip - too soon after last hit

        # Racket dimensions
        blade_width = 0.17   # X direction in local space
        blade_length = 0.18  # Z direction in local space
        blade_thick = 0.02   # Actual thickness

        # Helper: Rodrigues rotation
        def rotate_vector(v, k, angle):
            """Rotate vector v around axis k by angle (radians)"""
            if abs(angle) < 1e-6:
                return v.copy()
            cos_a = math.cos(angle)
            sin_a = math.sin(angle)
            return v * cos_a + np.cross(k, v) * sin_a + k * np.dot(k, v) * (1 - cos_a)

        # Get rotation parameters
        angle = racket.orientation_angle
        axis = racket.orientation_axis
        angle2 = getattr(racket, 'orientation_angle2', 0.0)
        axis2_local = getattr(racket, 'orientation_axis2', np.array([0, 1, 0]))

        # Transform rotation2's axis by rotation to get world-space axis
        axis2_world = rotate_vector(axis2_local, axis, angle)

        # Calculate racket normal in world coordinates
        local_normal_plus = np.array([0.0, 1.0, 0.0])
        world_normal = rotate_vector(local_normal_plus, axis, angle)
        world_normal = rotate_vector(world_normal, axis2_world, angle2)

        # Get ball position relative to racket in local coordinates
        rel_pos = ball.position - racket.position
        local_pos = rotate_vector(rel_pos, axis2_world, -angle2)
        local_pos = rotate_vector(local_pos, axis, -angle)

        # === ACTUAL COLLISION CHECK ===
        # Check if ball is actually touching the racket face (not just nearby)
        ball_margin = ball.radius + 0.005  # Ball radius + small tolerance

        # Check if ball center projects onto racket face (ellipse check)
        x_norm = local_pos[0] / (blade_width / 2)
        z_norm = local_pos[2] / (blade_length / 2)
        on_racket_face = (x_norm ** 2 + z_norm ** 2) <= 1.0

        if not on_racket_face:
            return  # Ball not over racket face

        y_dist = local_pos[1]

        # Check previous frame position for crossing detection
        rel_pos_prev = ball.prev_position - racket.prev_position
        local_pos_prev = rotate_vector(rel_pos_prev, axis2_world, -angle2)
        local_pos_prev = rotate_vector(local_pos_prev, axis, -angle)
        y_dist_prev = local_pos_prev[1]

        # Detect if ball crossed the racket plane
        crossed_from_plus = y_dist_prev > 0 and y_dist <= 0
        crossed_from_minus = y_dist_prev < 0 and y_dist >= 0
        crossed_plane = crossed_from_plus or crossed_from_minus

        # Check actual contact: ball surface touching racket surface
        contact_distance = blade_thick / 2 + ball_margin
        is_touching = abs(y_dist) < contact_distance

        # Determine if approaching (relative velocity toward racket)
        rel_vel = ball.velocity - racket.velocity
        vel_toward_racket = -np.dot(rel_vel, world_normal)
        is_approaching = vel_toward_racket > 0.1  # Must be moving toward racket

        # Collision requires: (crossed plane) OR (touching AND approaching)
        if not (crossed_plane or (is_touching and is_approaching)):
            return  # No collision

        # === COLLISION DETECTED - Apply physics ===
        # Determine which side was hit
        if crossed_plane:
            is_red_side = crossed_from_plus  # Came from +Y side = red
        else:
            is_red_side = y_dist > 0

        # Get rubber properties
        if is_red_side:
            rubber = racket.rubber_red
            restitution = racket.restitution[0]
            friction = racket.coefficient[0]
            surface_normal = world_normal
        else:
            rubber = racket.rubber_black
            restitution = racket.restitution[1]
            friction = racket.coefficient[1]
            surface_normal = -world_normal

        # === CAPTURE AND RELEASE METHOD ===
        # 1. Place ball on racket surface at the actual contact point (not center)
        # Keep the ball's local x,z position, only adjust y to surface
        contact_local = local_pos.copy()
        surface_offset = blade_thick / 2 + ball.radius + 0.003
        if is_red_side:
            contact_local[1] = surface_offset
        else:
            contact_local[1] = -surface_offset
        # Transform back to world coordinates
        contact_world = rotate_vector(contact_local, axis, angle)
        contact_world = rotate_vector(contact_world, axis2_world, angle2)
        ball.position = racket.position + contact_world

        # 2. Calculate output velocity based on racket velocity
        racket_speed = np.linalg.norm(racket.velocity)

        # Base reflection: reflect ball velocity off surface
        vel_normal_component = np.dot(ball.velocity, surface_normal)
        vel_tangent_component = ball.velocity - vel_normal_component * surface_normal

        # Friction affects tangent component retention
        # High friction = more grip = ball follows racket more
        # Low friction = slip = ball keeps more of its original tangent velocity
        tangent_retention = max(0.2, 0.9 - friction * 0.7)  # friction 0.3->0.69, friction 0.9->0.27

        # New velocity = racket velocity contribution + reflection
        # The faster the racket moves, the more it dominates the output
        racket_contribution = min(racket_speed * 1.2, 12.0)  # Cap at 12 m/s (reduced from 15)

        if racket_speed > 0.5:
            # Racket is moving - use racket velocity as primary
            racket_dir = racket.velocity / racket_speed
            # Output in racket direction, friction affects control
            new_velocity = racket_dir * racket_contribution * restitution
            # Friction reduces tangent slip, low friction retains more tangent
            new_velocity = new_velocity + vel_tangent_component * tangent_retention * 0.4
            # Add upward component to clear net (reduced)
            new_velocity[1] = max(new_velocity[1], 0.5)
        else:
            # Racket is slow - simple reflection with friction consideration
            new_velocity = -vel_normal_component * restitution * surface_normal
            new_velocity = new_velocity + vel_tangent_component * tangent_retention
            new_velocity = new_velocity + racket.velocity * 0.5

        ball.velocity = new_velocity

        # === SPIN CALCULATION ===
        # Spin is generated by relative motion between ball and racket surface
        # Key: the tangent component of relative velocity creates spin
        rubber_type = rubber.rubber_type

        # Calculate relative velocity (ball relative to racket)
        # This is what creates the "rubbing" effect
        relative_vel = ball.velocity - racket.velocity

        # Get tangent component of relative velocity (parallel to racket surface)
        rel_normal_vel = np.dot(relative_vel, surface_normal)
        rel_tangent_vel = relative_vel - rel_normal_vel * surface_normal
        rel_tangent_speed = np.linalg.norm(rel_tangent_vel)

        # Also consider racket's own tangent motion for active spin generation
        racket_normal_vel = np.dot(racket.velocity, surface_normal)
        racket_tangent_vel = racket.velocity - racket_normal_vel * surface_normal
        racket_tangent_speed = np.linalg.norm(racket_tangent_vel)

        # Spin is generated when there's tangent motion (either from ball or racket)
        # Use the larger of the two for spin calculation
        effective_tangent_speed = max(rel_tangent_speed, racket_tangent_speed)
        effective_tangent_vel = rel_tangent_vel if rel_tangent_speed >= racket_tangent_speed else -racket_tangent_vel

        # Spin axis is perpendicular to surface normal and tangent velocity
        if effective_tangent_speed > 0.01:  # Very low threshold
            spin_axis = np.cross(surface_normal, effective_tangent_vel)
            spin_axis_norm = np.linalg.norm(spin_axis)
            if spin_axis_norm > 0:
                spin_axis = spin_axis / spin_axis_norm

                # Spin intensity depends on friction, speed, and rubber type
                if rubber_type == RubberType.LONG_PIMPLES:
                    # Long pimples: reverse incoming spin, weak new spin
                    spin_reversal = rubber.spin_reversal
                    ball.spin = ball.spin * (-spin_reversal * 0.7)
                    ball.spin = ball.spin + spin_axis * effective_tangent_speed * friction * 15
                elif rubber_type == RubberType.PIMPLES:
                    # Short pimples: moderate spin generation
                    ball.spin = ball.spin * 0.4
                    ball.spin = ball.spin + spin_axis * effective_tangent_speed * friction * 35
                elif rubber_type == RubberType.ANTI:
                    # Anti-spin: minimal spin transfer
                    ball.spin = ball.spin * 0.1
                    ball.spin = ball.spin + spin_axis * effective_tangent_speed * friction * 8
                else:
                    # Inverted (normal rubber): high spin generation
                    ball.spin = ball.spin * 0.2
                    ball.spin = ball.spin + spin_axis * effective_tangent_speed * friction * 60
        else:
            # Very slow contact - just reduce existing spin based on rubber type
            if rubber_type == RubberType.LONG_PIMPLES:
                ball.spin = ball.spin * (-0.4)  # Reverse
            elif rubber_type == RubberType.ANTI:
                ball.spin = ball.spin * 0.1
            else:
                ball.spin = ball.spin * 0.4

        # Record hit time for cooldown
        ball.last_racket_hit_time = self._physics_time

    def get_entity(self, entity_id: str) -> Optional[GameEntity]:
        """Get entity by ID"""
        return self.entities.get(entity_id)

    def get_all_balls(self) -> List[BallEntity]:
        """Get all ball entities"""
        return self.balls.copy()

    def get_all_rackets(self) -> List[RacketEntity]:
        """Get all racket entities"""
        return self.rackets.copy()
