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


@dataclass
class RubberData:
    """Rubber properties for racket"""
    type: str = "inverted"  # inverted, pimples, anti
    friction: float = 0.8
    spin_coefficient: float = 1.0
    restitution: float = 0.85


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


@dataclass
class RacketEntity(GameEntity):
    """Racket entity with swing properties"""
    entity_type: EntityType = EntityType.RACKET
    acceleration: np.ndarray = field(default_factory=lambda: np.zeros(3))
    mass: float = 0.18  # 180g typical
    rubber: RubberData = field(default_factory=RubberData)
    # Swing state
    swing_active: bool = False
    swing_time: float = 0.0


class EntityManager:
    """Manages all game entities"""

    def __init__(self):
        self.entities: Dict[str, GameEntity] = {}
        self.balls: List[BallEntity] = []
        self.rackets: List[RacketEntity] = []
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

        # Rotation
        if 'rotation' in nbt:
            if isinstance(nbt['rotation'], list) and len(nbt['rotation']) >= 2:
                racket.rotation = np.array(nbt['rotation'][:2], dtype=float)

        # Mass
        if 'mass' in nbt:
            racket.mass = float(nbt['mass'])

        # Rubber properties
        if 'rubber' in nbt:
            rubber_data = nbt['rubber']
            racket.rubber = RubberData(
                type=rubber_data.get('type', 'inverted'),
                friction=float(rubber_data.get('friction', 0.8)),
                spin_coefficient=float(rubber_data.get('spin', 1.0)),
                restitution=float(rubber_data.get('restitution', 0.85))
            )

        return racket

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

        # Air drag
        speed = np.linalg.norm(ball.velocity)
        if speed > 0:
            drag_force = -0.5 * params.air_density * params.drag_coefficient * \
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
                lift_force = 0.5 * params.air_density * Cl * \
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
        ball.spin = ball.spin * (1.0 - params.spin_decay_rate * dt)

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

            # Out of bounds
            if ball.position[1] < -1 or np.linalg.norm(ball.position) > 10:
                ball.active = False

    def get_entity(self, entity_id: str) -> Optional[GameEntity]:
        """Get entity by ID"""
        return self.entities.get(entity_id)

    def get_all_balls(self) -> List[BallEntity]:
        """Get all ball entities"""
        return self.balls.copy()

    def get_all_rackets(self) -> List[RacketEntity]:
        """Get all racket entities"""
        return self.rackets.copy()
