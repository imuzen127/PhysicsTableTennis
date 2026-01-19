"""
Table Tennis Physics Simulation - Interactive Game

A Minecraft-like experience where you're IN the game world,
can type commands, and watch the ball fly in real-time.

Controls:
    WASD         - Move
    Mouse Side 2 - Up (ascend)
    Mouse Side 1 - Down (descend)
    Mouse        - Look around (always active)
    / (slash)    - Open chat/command
    ESC          - Toggle menu / Close chat
    Enter        - Send command
"""

import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
import sys
import math
import time
import re

sys.path.insert(0, '.')

from src.physics.parameters import PhysicsParameters, create_offensive_setup
from src.physics.table import Table
from src.physics.collision import CollisionHandler
from src.command.parser import CommandParser
from src.command.objects import EntityManager, BallEntity, RacketEntity, TableEntity


class PlayMode:
    """
    Play mode system for mouse-controlled table tennis gameplay.

    Features:
    - Auto mode: Match with score, serve rotation, 10-second serve limit
    - Free mode: Practice mode with manual serve command
    - Mouse-controlled racket movement
    - Fixed camera position relative to table
    - Swing-based spin mechanics
    """

    # Table side positions (relative to table center)
    # Based on user investigation for stable viewing positions
    # Side 1: -X direction (player at -X end, facing +X toward opponent)
    # Side 2: +X direction (player at +X end, facing -X toward opponent)
    SIDE_POSITIONS = {
        1: {'offset': np.array([-2.084, 1.42, -0.001]), 'yaw': 0.0, 'pitch': -27.0},
        2: {'offset': np.array([2.084, 1.42, 0.001]), 'yaw': 180.0, 'pitch': -27.0}
    }

    # Racket base rotation for each side (to face opponent)
    # rotation makes red face point toward opponent
    RACKET_ROTATIONS = {
        1: {'angle': 1.570, 'axis': np.array([0.0, 0.0, -1.0])},  # Face +X
        2: {'angle': 1.570, 'axis': np.array([0.0, 0.0, 1.0])}    # Face -X
    }

    # Racket height (fixed)
    RACKET_HEIGHT = 0.85  # Table height + small offset

    def __init__(self, game_world):
        self.game = game_world

        # Play mode state
        self.active = False
        self.mode = None  # 'auto' or 'free'
        self.racket = None
        self.table = None
        self.side = None  # 1 or 2

        # Camera/view state
        self.fixed_camera_pos = np.array([0.0, 0.0, 0.0])
        self.fixed_camera_yaw = 0.0
        self.fixed_camera_pitch = 0.0

        # Mouse control state (left button - move and swing)
        self.mouse_down = False
        self.mouse_down_time = 0
        self.last_mouse_pos = (0, 0)
        self.swing_start_pos = (0, 0)
        self.swing_history = []  # Track mouse positions for swing curve analysis
        self.is_swinging = False

        # Right mouse button state (spin control - absolute axis rotation)
        self.right_mouse_down = False
        self.right_mouse_start_pos = (0, 0)
        self.right_mouse_last_pos = (0, 0)
        # Absolute axis rotation control
        self.rotation_y_axis = 0.0  # Rotation around absolute Y axis (mouse up/down)
        self.rotation_z_axis = 0.0  # Rotation around absolute Z axis (mouse left/right)

        # Double-click detection for serve toss
        self.last_click_time = 0
        self.double_click_threshold = 300  # ms

        # Serve state
        self.serve_ready = False
        self.serve_toss_active = False
        self.serve_ball = None

        # Auto match mode state
        self.score_player = 0
        self.score_opponent = 0
        self.current_server = 1  # 1 or 2 (which side serves)
        self.serve_count = 0  # Serves in current rotation
        self.serve_time_start = 0
        self.serve_timeout = 10000  # 10 seconds in ms

        # Racket position in play area (relative to player's side)
        self.racket_x = 0.0  # Left-right on table
        self.racket_z = 0.0  # Forward-backward from edge

        # Racket height adjustment (for lobbing/smash)
        self.base_racket_height = 0.18  # Base height above table
        self.auto_height_offset = 0.0  # Auto-adjusted for high balls

    def enter(self, mode: str, racket, table, side: int):
        """Enter play mode"""
        self.active = True
        self.mode = mode
        self.racket = racket
        self.table = table
        self.side = side

        # Add play_controlled tag to prevent replay commands from affecting this racket
        if not hasattr(racket, 'tags'):
            racket.tags = []
        if 'play_controlled' not in racket.tags:
            racket.tags.append('play_controlled')

        # Calculate fixed camera position
        side_config = self.SIDE_POSITIONS[side]
        table_pos = table.position
        self.fixed_camera_pos = table_pos + side_config['offset']
        self.fixed_camera_yaw = side_config['yaw']
        self.fixed_camera_pitch = side_config['pitch']

        # Position racket at player's side
        self._update_racket_position(0.0, 0.0)

        # Set racket to manual control (position controlled by mouse, not physics)
        self.racket.manual_control = True

        # Start simulation (for both free and auto mode)
        self.game.entity_manager.start()

        # Reset scores if auto mode
        if mode == 'auto':
            self.score_player = 0
            self.score_opponent = 0
            # Random first server
            import random
            self.current_server = random.choice([1, 2])
            self.serve_count = 0
            self.serve_time_start = pygame.time.get_ticks()

        # Reset racket control state
        self.racket_x = 0.0
        self.racket_z = 0.0
        self.swing_history = []
        self.is_swinging = False
        self.serve_ready = False
        self.serve_toss_active = False
        # Reset absolute axis rotation
        self.rotation_y_axis = 0.0
        self.rotation_z_axis = 0.0

    def exit(self):
        """Exit play mode"""
        # Remove play_controlled tag from racket
        if self.racket and hasattr(self.racket, 'tags') and 'play_controlled' in self.racket.tags:
            self.racket.tags.remove('play_controlled')

        self.active = False
        self.mode = None
        self.racket = None
        self.table = None
        self.side = None

    def update(self, dt):
        """Update play mode state"""
        if not self.active:
            return

        # Update racket position based on mouse movement
        if self.is_swinging and self.swing_history:
            # Apply swing velocity to racket
            self._apply_swing_motion()

        # Manual height adjustment via side buttons
        self._update_manual_height()

        # Check serve timeout in auto mode
        if self.mode == 'auto' and self.serve_ready:
            elapsed = pygame.time.get_ticks() - self.serve_time_start
            if elapsed > self.serve_timeout:
                # Serve timeout - point to opponent
                self._serve_timeout()

        # Update serve toss physics
        if self.serve_toss_active and self.serve_ball:
            self._update_serve_toss()

    def get_camera_state(self):
        """Return fixed camera state for play mode"""
        return (self.fixed_camera_pos.copy(),
                self.fixed_camera_yaw,
                self.fixed_camera_pitch)

    def handle_mouse_down(self, button, pos):
        """Handle mouse button down"""
        if not self.active:
            return False

        current_time = pygame.time.get_ticks()

        if button == 1:  # Left button - move racket and swing
            # Check for double-click (serve toss)
            if current_time - self.last_click_time < self.double_click_threshold:
                self._start_serve_toss()
                self.last_click_time = 0  # Reset to prevent triple-click
                return True

            self.last_click_time = current_time

            # Start tracking for swing
            self.mouse_down = True
            self.mouse_down_time = current_time
            self.swing_start_pos = pos
            self.last_mouse_pos = pos
            self.drag_last_mouse_pos = pos  # Initialize for delta tracking
            self.swing_history = [(current_time, pos)]
            return True

        elif button == 3:  # Right button - spin control
            self.right_mouse_down = True
            self.right_mouse_start_pos = pos
            self.right_mouse_last_pos = pos
            return True

        return False

    def handle_mouse_up(self, button, pos):
        """Handle mouse button up"""
        if not self.active:
            return False

        if button == 1 and self.mouse_down:
            self.mouse_down = False

            # Calculate swing if there was movement
            if len(self.swing_history) > 2:
                self._execute_swing()

            # Stop racket movement (but keep angle)
            if self.racket:
                self.racket.velocity = np.zeros(3)
                # rotation2 is NOT reset - keep the swing angle

            self.swing_history = []
            self.is_swinging = False
            self.drag_last_mouse_pos = None  # Reset for next drag
            return True

        elif button == 3 and getattr(self, 'right_mouse_down', False):
            self.right_mouse_down = False
            return True

        return False

    def handle_mouse_move(self, pos, rel):
        """Handle mouse movement"""
        if not self.active:
            return False

        if self.mouse_down:
            # Left button held - move racket and record swing
            current_time = pygame.time.get_ticks()
            self.swing_history.append((current_time, pos))

            # Keep only recent history (last 200ms)
            cutoff_time = current_time - 200
            self.swing_history = [(t, p) for t, p in self.swing_history if t >= cutoff_time]

            self.is_swinging = True
            self.last_mouse_pos = pos

            # Update racket position only when left button is held
            self._update_racket_from_mouse(pos)

        if getattr(self, 'right_mouse_down', False):
            # Right button held - adjust racket angle (spherical rotation)
            last_pos = getattr(self, 'right_mouse_last_pos', pos)
            dx = pos[0] - last_pos[0]
            dy = pos[1] - last_pos[1]
            self._adjust_racket_angle(dx, dy)
            self.right_mouse_last_pos = pos

        return True

    def _update_racket_from_mouse(self, mouse_pos):
        """Update racket position based on mouse movement (delta)

        Moves racket by the amount the mouse moved, not absolute position.
        """
        if not self.table or not self.racket:
            return

        # Get last position for delta calculation
        last_pos = getattr(self, 'drag_last_mouse_pos', None)
        if last_pos is None:
            # First frame of drag - just store position, don't move
            self.drag_last_mouse_pos = mouse_pos
            return

        # Calculate mouse delta
        dx = mouse_pos[0] - last_pos[0]
        dy = mouse_pos[1] - last_pos[1]
        self.drag_last_mouse_pos = mouse_pos

        # Sensitivity for mouse-to-world movement
        sensitivity = 0.002  # Reduced for smoother control

        table_pos = self.table.position
        net_x = table_pos[0]

        # Current racket position
        current_pos = self.racket.position.copy()

        # Mouse X movement -> World Z (left-right from player view)
        # Side 2 is facing opposite direction, so invert Z movement
        if self.side == 1:
            current_pos[2] += dx * sensitivity
        else:
            current_pos[2] -= dx * sensitivity

        # Mouse Y movement -> World X (forward-back)
        if self.side == 1:
            # Side 1: mouse down = back (-X), mouse up = forward (+X)
            current_pos[0] -= dy * sensitivity
            # Only prevent net crossing (no backward limit)
            max_x = net_x - 0.05
            current_pos[0] = min(max_x, current_pos[0])
        else:
            # Side 2: mouse down = back (+X), mouse up = forward (-X)
            current_pos[0] += dy * sensitivity
            # Only prevent net crossing (no backward limit)
            min_x = net_x + 0.05
            current_pos[0] = max(min_x, current_pos[0])

        # Clamp Z to table width
        half_width = self.table.width / 2 + 0.3
        current_pos[2] = max(table_pos[2] - half_width, min(table_pos[2] + half_width, current_pos[2]))

        # Calculate velocity from position change (for collision response)
        old_pos = self.racket.position.copy()
        position_delta = current_pos - old_pos
        # Estimate velocity based on ~60fps (0.016s per frame)
        vel_scale = 40.0  # Convert position delta to velocity
        raw_velocity = position_delta * vel_scale
        speed = np.linalg.norm(raw_velocity)

        # Non-linear scaling: boost slow movements, cap fast movements
        # power < 1 compresses the range, making slow movements more effective
        max_speed = 12.0
        if speed > 0.1:  # Minimum threshold
            power = 0.7  # Boost slow movements
            # Scale the speed through power curve then rescale
            scaled_speed = (speed ** power) * 2.5  # Multiplier to maintain reasonable range
            scaled_speed = min(scaled_speed, max_speed)
            self.racket.velocity = (raw_velocity / speed) * scaled_speed
        else:
            self.racket.velocity = raw_velocity

        # Save previous position for swept collision detection
        self.racket.prev_position = old_pos
        self.racket.position = current_pos

        # Apply base rotation (but don't touch rotation2 - that's for swing)
        self._apply_base_rotation_only()

    def _apply_base_rotation_only(self):
        """Apply base rotation with absolute Y/Z axis adjustments"""
        if not self.racket or not self.side:
            return

        rot_config = self.RACKET_ROTATIONS[self.side]
        # rotation1: base rotation (around Z axis) + Z axis rotation from mouse X
        self.racket.orientation_angle = rot_config['angle'] + self.rotation_z_axis
        self.racket.orientation_axis = rot_config['axis'].copy()

        # rotation2: Y axis rotation from mouse Y (horizontal spin)
        self.racket.orientation_angle2 = self.rotation_y_axis
        self.racket.orientation_axis2 = np.array([0.0, 1.0, 0.0])

    def _apply_racket_orientation(self):
        """Apply full racket orientation (both rotation and rotation2)"""
        if not self.racket or not self.side:
            return

        rot_config = self.RACKET_ROTATIONS[self.side]
        # rotation1: base rotation + Z axis rotation
        self.racket.orientation_angle = rot_config['angle'] + self.rotation_z_axis
        self.racket.orientation_axis = rot_config['axis'].copy()

        # rotation2: Y axis rotation
        self.racket.orientation_angle2 = self.rotation_y_axis
        self.racket.orientation_axis2 = np.array([0.0, 1.0, 0.0])

    def _update_racket_position(self, x_offset, z_offset):
        """Update racket world position (used for initial positioning)"""
        if not self.table or not self.racket:
            return

        table_pos = self.table.position
        table_height = self.table.height

        # Base position at player's edge of table
        # Side 1: player at -X end, facing +X
        # Side 2: player at +X end, facing -X
        if self.side == 1:
            base_x = table_pos[0] - self.table.length / 2 - 0.15
        else:
            base_x = table_pos[0] + self.table.length / 2 + 0.15

        # Calculate racket position
        pos = np.array([
            base_x,
            table_pos[1] + table_height + 0.25,  # 25cm above table surface
            table_pos[2] + x_offset  # Z is left-right from player view
        ])

        # Apply forward/backward offset based on side
        if self.side == 1:
            pos[0] += z_offset  # Forward is +X for side 1
        else:
            pos[0] -= z_offset  # Forward is -X for side 2

        # Save previous position for swept collision detection
        self.racket.prev_position = self.racket.position.copy()
        self.racket.position = pos
        self._apply_racket_orientation()

    def _adjust_racket_angle(self, dx, dy):
        """Adjust racket angle based on right-click drag (absolute axis rotation)

        dy (mouse up/down) -> rotate around absolute Z axis (tilt)
        dx (mouse left/right) -> rotate around absolute Y axis (horizontal spin)
        """
        if not self.racket:
            return

        # Sensitivity for angle adjustment
        sensitivity = 0.008

        # Update absolute axis rotation values
        self.rotation_z_axis -= dy * sensitivity  # Mouse up/down -> Z axis rotation
        self.rotation_y_axis -= dx * sensitivity  # Mouse left/right -> Y axis rotation

        # Apply rotation
        self._apply_base_rotation_only()

    def _apply_swing_motion(self):
        """Track swing motion for physics - does NOT change racket angle"""
        # Angle is controlled only by right-click drag
        # This function is kept for compatibility but does nothing
        pass

    def _update_manual_height(self):
        """Manual racket height adjustment via mouse side buttons"""
        if not self.table or not self.racket:
            return

        # Use game's side button state for height control
        height_speed = 0.008  # Height change per frame (reduced)
        min_y = self.table.height + 0.19
        max_y = self.table.height + 2.5  # Increased limit

        if self.game.mouse_side2_held:  # Side button 2 = up
            self.racket.position[1] = min(max_y, self.racket.position[1] + height_speed)
        if self.game.mouse_side1_held:  # Side button 1 = down
            self.racket.position[1] = max(min_y, self.racket.position[1] - height_speed)

    def _start_serve_toss(self):
        """Start serve toss (double-click) - works in both free and auto mode"""
        # Delete existing serve ball if it exists
        if self.serve_ball:
            try:
                self.game.entity_manager._remove_entity(self.serve_ball)
            except:
                pass
            self.serve_ball = None
            self.serve_toss_active = False

        # Create ball for serve toss - position toward net (奥 = beyond racket toward net)
        toss_pos = self.racket.position.copy()
        toss_pos[1] += 0.05  # Start slightly above racket
        # Move ball toward net by 1 ball diameter (~4cm)
        if self.side == 1:
            toss_pos[0] += 0.04  # Toward net for side 1 (+X direction)
        else:
            toss_pos[0] -= 0.04  # Toward net for side 2 (-X direction)

        # Spawn a ball for the toss
        # v = sqrt(2*g*h) for h=0.16m (16cm minimum rule): v ≈ 1.8 m/s
        nbt = {
            'velocity': np.array([0.0, 1.8, 0.0]),  # Toss up (~16cm height for legal serve)
            'spin': np.zeros(3)
        }
        # Debug: log spawn position
        self.game._debug_log(f"SERVE_TOSS spawn: racket={self.racket.position} side={self.side} toss_pos={toss_pos}")
        self.serve_ball = self.game.entity_manager.summon('ball', toss_pos, nbt)
        self.serve_ball.active = True
        # Set spawn time for collision cooldown
        self.serve_ball.spawn_time = self.game.entity_manager._physics_time if hasattr(self.game.entity_manager, '_physics_time') else 0
        self.serve_toss_active = True
        self.serve_ready = True
        self.serve_time_start = pygame.time.get_ticks()

        self.game.add_output("Serve toss!")

    def _update_serve_toss(self):
        """Update serve toss ball physics"""
        if not self.serve_ball:
            return

        # Check if ball has fallen back down (ready to hit)
        if self.serve_ball.position[1] < self.table.height + 0.3:
            # Ball is low, can be hit during swing
            pass

        # Check if ball hit the ground (failed toss)
        if self.serve_ball.position[1] < 0:
            self._serve_fault()

    def _execute_swing(self):
        """Execute swing based on mouse movement history"""
        if len(self.swing_history) < 3:
            return

        # Analyze swing curve
        swing_data = self._analyze_swing()

        # Apply swing to racket (update orientation and velocity)
        self._apply_swing_to_racket(swing_data)

        # Check for ball collision
        self._check_swing_ball_collision(swing_data)

    def _analyze_swing(self):
        """Analyze swing curve to determine spin and angle"""
        if len(self.swing_history) < 2:
            return {'direction': (0, 0), 'speed': 0, 'curve': 0, 'pull_back': False}

        # Get start and end points
        _, start_pos = self.swing_history[0]
        _, end_pos = self.swing_history[-1]

        # Calculate direction and distance
        dx = end_pos[0] - start_pos[0]
        dy = end_pos[1] - start_pos[1]
        distance = math.sqrt(dx * dx + dy * dy)

        # Calculate time for speed
        start_time = self.swing_history[0][0]
        end_time = self.swing_history[-1][0]
        duration = max(1, end_time - start_time) / 1000.0  # seconds

        speed = distance / duration if duration > 0 else 0

        # Analyze curve (for left/right spin)
        curve = self._calculate_curve()

        # Check for pull-back motion (backspin)
        pull_back = self._check_pull_back()

        return {
            'direction': (dx, dy),
            'speed': speed,
            'curve': curve,
            'pull_back': pull_back,
            'distance': distance
        }

    def _calculate_curve(self):
        """Calculate curvature of swing path (positive = clockwise, negative = counter-clockwise)"""
        if len(self.swing_history) < 5:
            return 0.0

        # Sample points along the path
        n = len(self.swing_history)
        mid_idx = n // 2

        _, start = self.swing_history[0]
        _, mid = self.swing_history[mid_idx]
        _, end = self.swing_history[-1]

        # Calculate signed area of triangle (cross product)
        # Positive = counter-clockwise curve, Negative = clockwise curve
        ax, ay = mid[0] - start[0], mid[1] - start[1]
        bx, by = end[0] - start[0], end[1] - start[1]

        cross = ax * by - ay * bx

        # Normalize by distance
        dist = math.sqrt(bx*bx + by*by)
        if dist < 1:
            return 0.0

        return cross / dist

    def _check_pull_back(self):
        """Check if swing ends with pull-back motion (for backspin)"""
        if len(self.swing_history) < 4:
            return False

        # Look at last quarter of swing
        n = len(self.swing_history)
        quarter = n // 4
        if quarter < 2:
            return False

        # Check if Y increased (pulled toward player) in last part
        _, early = self.swing_history[-quarter-1]
        _, late = self.swing_history[-1]

        dy = late[1] - early[1]  # Positive = pulled toward player (down on screen)

        return dy > 20  # Threshold for pull-back detection

    def _apply_swing_to_racket(self, swing_data):
        """Apply swing velocity for physics - does NOT change racket angle"""
        # Angle is controlled only by right-click drag
        # This function only updates velocity for collision physics
        if not self.racket:
            return

        direction = swing_data['direction']
        speed = swing_data['speed']

        if speed < 50:
            return

        dx, dy = direction
        dist = math.sqrt(dx * dx + dy * dy)
        if dist > 0:
            # Set velocity based on swing direction for physics calculations
            vel_scale = speed * 0.002
            if self.side == 1:
                self.racket.velocity = np.array([dy * vel_scale, 0.0, dx * vel_scale])
            else:
                self.racket.velocity = np.array([-dy * vel_scale, 0.0, -dx * vel_scale])

    def _check_swing_ball_collision(self, swing_data):
        """Check if swing hits any balls and apply physics"""
        if not self.racket:
            return

        speed = swing_data['speed']
        if speed < 100:  # Too slow for effective hit
            return

        # Check for nearby balls
        for ball in self.game.entity_manager.balls:
            dist = np.linalg.norm(ball.position - self.racket.position)

            if dist < 0.2:  # Hit range
                self._apply_hit(ball, swing_data)
                break

    def _apply_hit(self, ball, swing_data):
        """Apply hit physics to ball"""
        direction = swing_data['direction']
        speed = swing_data['speed']
        curve = swing_data['curve']
        pull_back = swing_data['pull_back']

        # Determine ball height relative to table (for lob/smash detection)
        table_height = self.table.height if self.table else 0.76
        ball_height_above_table = ball.position[1] - table_height
        is_high_ball = ball_height_above_table > 0.25  # 25cm above table = high ball

        # Determine hit type based on swing speed and ball height
        is_lob = speed < 200  # Very slow swing = lob
        is_smash = is_high_ball and speed > 800  # Fast swing on high ball = smash

        # Base hit velocity
        if is_lob:
            # Lob: high, slow arc
            hit_speed = 3.0  # Slow
            hit_type = "Lob"
        elif is_smash:
            # Smash: fast, downward trajectory
            hit_speed = min(20.0, speed / 40.0)  # Faster scaling for smash
            hit_type = "SMASH!"
        else:
            # Normal hit
            hit_speed = min(15.0, speed / 50.0)  # Scale mouse speed to ball speed
            hit_type = "Hit"

        # Direction based on swing and side
        dx, dy = direction
        swing_len = math.sqrt(dx*dx + dy*dy)
        if swing_len < 1:
            swing_len = 1

        # Normalize and scale
        norm_dx = dx / swing_len
        norm_dy = dy / swing_len

        # Map screen direction to world direction
        # For side 1: +screen_x -> -world_z, +screen_y -> +world_x
        # For side 2: +screen_x -> +world_z, +screen_y -> -world_x
        if self.side == 1:
            vel_x = -hit_speed * 0.8  # Toward opponent
            vel_z = -norm_dx * hit_speed * 0.3
            if is_lob:
                vel_y = 2.5  # High arc for lob
            elif is_smash:
                vel_y = -2.0  # Downward for smash
            else:
                vel_y = -norm_dy * hit_speed * 0.2 + 1.0  # Slight upward arc
        else:
            vel_x = hit_speed * 0.8
            vel_z = norm_dx * hit_speed * 0.3
            if is_lob:
                vel_y = 2.5  # High arc for lob
            elif is_smash:
                vel_y = -2.0  # Downward for smash
            else:
                vel_y = -norm_dy * hit_speed * 0.2 + 1.0

        # Apply velocity
        ball.velocity = np.array([vel_x, vel_y, vel_z])

        # Calculate spin based on swing characteristics
        spin = np.zeros(3)

        # Curve adds sidespin (left/right rotation)
        if abs(curve) > 10:
            # Clockwise curve (negative) = right spin, counter-clockwise = left spin
            spin[1] = -curve * 50  # Y-axis spin (sidespin)

        # Pull-back adds backspin
        if pull_back:
            spin[2] = -3000 * (math.pi / 30)  # Convert to rad/s
            self.game.add_output("Backspin!")
        elif is_lob:
            # Lobs have minimal spin
            spin[2] = 500 * (math.pi / 30)
        elif is_smash:
            # Smash has strong topspin
            spin[2] = 4000 * (math.pi / 30)
        else:
            # Default forward swing adds topspin
            spin[2] = 2000 * (math.pi / 30)

        ball.spin = spin
        ball.active = True

        # Clear serve state
        if self.serve_toss_active:
            self.serve_toss_active = False
            self.serve_ready = False

        self.game.add_output(f"{hit_type}! Speed: {hit_speed:.1f}")

    def _serve_timeout(self):
        """Handle serve timeout in auto mode"""
        if self.mode != 'auto':
            return

        # Point to opponent
        if self.current_server == self.side:
            self.score_opponent += 1
            self.game.add_output(f"Serve timeout! Score: {self.score_player}-{self.score_opponent}")
        else:
            self.score_player += 1
            self.game.add_output(f"Opponent serve timeout! Score: {self.score_player}-{self.score_opponent}")

        self._next_serve()

    def _serve_fault(self):
        """Handle serve fault (ball hit ground)"""
        if self.serve_ball:
            self.game.entity_manager._remove_entity(self.serve_ball)
            self.serve_ball = None

        self.serve_toss_active = False
        self.serve_ready = False

        if self.mode == 'auto' and self.current_server == self.side:
            self.score_opponent += 1
            self.game.add_output(f"Serve fault! Score: {self.score_player}-{self.score_opponent}")
            self._next_serve()

    def _next_serve(self):
        """Move to next serve in auto mode"""
        self.serve_count += 1

        # Rotate serve every 2 points
        if self.serve_count >= 2:
            self.serve_count = 0
            self.current_server = 2 if self.current_server == 1 else 1

        self.serve_time_start = pygame.time.get_ticks()
        self.serve_ready = False

        if self.current_server == self.side:
            self.game.add_output("Your serve!")
        else:
            self.game.add_output("Opponent's serve")

    def render_ui(self, surface):
        """Render play mode UI elements"""
        if not self.active:
            return

        font = self.game.font_medium

        # Score display for auto mode
        if self.mode == 'auto':
            score_text = f"Score: {self.score_player} - {self.score_opponent}"
            text_surf = font.render(score_text, True, (255, 255, 255))
            surface.blit(text_surf, (self.game.width // 2 - text_surf.get_width() // 2, 10))

            # Serve indicator
            if self.current_server == self.side:
                serve_text = "YOUR SERVE"
                if self.serve_ready:
                    elapsed = (pygame.time.get_ticks() - self.serve_time_start) / 1000.0
                    remaining = 10.0 - elapsed
                    serve_text = f"SERVE NOW! {remaining:.1f}s"
                text_surf = font.render(serve_text, True, (255, 255, 100))
            else:
                serve_text = "Opponent's serve"
                text_surf = font.render(serve_text, True, (180, 180, 180))
            surface.blit(text_surf, (self.game.width // 2 - text_surf.get_width() // 2, 45))

        # Mode indicator
        mode_text = f"[{self.mode.upper()} MODE] Side {self.side}"
        text_surf = self.game.font_small.render(mode_text, True, (150, 200, 255))
        surface.blit(text_surf, (10, 10))

        # Instructions
        instructions = [
            "Double-click: Serve toss",
            "Hold + drag: Swing",
            "ESC or /played @s: Exit"
        ]
        y = 35
        for line in instructions:
            text_surf = self.game.font_small.render(line, True, (180, 180, 180))
            surface.blit(text_surf, (10, y))
            y += 18


class GameWorld:
    """The 3D game world with real-time physics"""

    def __init__(self, width=1280, height=720):
        self.width = width
        self.height = height

        # Initialize pygame
        pygame.init()
        pygame.font.init()

        # Create resizable window with OpenGL (smaller default size)
        self.screen = pygame.display.set_mode((width, height), DOUBLEBUF | OPENGL | RESIZABLE)
        pygame.display.set_caption("Table Tennis Simulation - Press / for commands")

        # Lock mouse to center for FPS-style control
        pygame.mouse.set_visible(False)
        pygame.event.set_grab(True)

        # Fonts
        self.font_large = pygame.font.SysFont('consolas', 32)
        self.font_medium = pygame.font.SysFont('consolas', 24)
        self.font_small = pygame.font.SysFont('consolas', 18)

        # Physics
        self.params = create_offensive_setup()
        self.params.dt = 0.002

        self.table = Table(self.params)
        self.collision = CollisionHandler(self.params)

        # Entity system
        self.entity_manager = EntityManager()
        self.command_parser = None  # Initialized after self is ready

        # Player state (Y-up coordinate system)
        # Player is also an entity with angle-axis rotation
        self.camera_pos = np.array([-3.0, 1.5, 2.0])  # [x, height, z]
        self.camera_yaw = -30.0    # Degrees (for backward compat)
        self.camera_pitch = 15.0   # Degrees (for backward compat)
        # Player rotation in angle-axis format
        self.player_rotation_angle = 0.0
        self.player_rotation_axis = np.array([0.0, 1.0, 0.0])
        self.camera_speed = 0.08

        # Mouse sensitivity
        self.mouse_sensitivity = 0.15

        # Console/Chat state
        self.console_open = False
        self.console_input = ""
        self.console_cursor = 0  # Cursor position in input
        self.console_history = []
        self.history_index = -1
        self.console_output = []
        self.max_output_lines = 8

        # Menu state
        self.menu_open = False

        # Game state
        self.running = True
        self.paused = False
        self.show_help = True
        self.show_debug = False  # F3 debug screen
        self.show_ball_orientation = False  # F3+B: show ball orientation line
        self.time_scale = 1.0

        # Data popup state
        self.data_popup_open = False
        self.data_popup_content = []  # Lines of NBT data to display
        self.data_popup_title = ""
        self.data_popup_close_rect = (0, 0, 0, 0)  # (x, y, w, h) for click detection

        # Mouse button state for up/down
        self.mouse_side1_held = False  # Button 7 - down
        self.mouse_side2_held = False  # Button 6 - up
        
        # Speed control
        self.base_speed = 0.08
        self.speed_min = 0.02
        self.speed_max = 0.3

        # Clock
        self.clock = pygame.time.Clock()
        self.fps = 60

        # Initialize OpenGL
        self._init_gl()

        # Welcome message
        self.add_output("Welcome! Press / to type commands")
        self.add_output("Try: summon ball ~ ~1 ~ or start")

        # Initialize command parser (needs self reference)
        self.command_parser = CommandParser(self)

        # Scheduled commands for delayed execution
        # List of (execute_time_ms, command_str)
        self.scheduled_commands = []
        self.function_start_time = 0  # Time when function started (ms)

        # Recording system for save/replay (memo-based approach)
        self.recording_active = False
        self.recording_name = ""
        self.recording_start_time = 0
        self.recording_memo = []  # List of frame snapshots: {timestamp, entities: [...]}
        self.recording_last_frame_time = 0
        self.recording_frame_interval = 16  # Record every ~16ms (60fps)
        self.recording_entity_tags = {}  # Maps id(entity) -> tag
        self.recording_tracked_entities = set()  # Set of id(entity) currently tracked
        self.recording_tag_counter = {}  # Counter for generating unique tags

        # Debug logging to file
        self.debug_log_file = None
        self._init_debug_log()

        # Spawn default table at origin
        self._spawn_default_table()

        # Play mode system
        self.play_mode = PlayMode(self)

    def _init_debug_log(self):
        """Initialize debug log file"""
        import os
        base_path = os.path.dirname(os.path.abspath(__file__))
        log_path = os.path.join(base_path, 'debug_log.txt')
        try:
            self.debug_log_file = open(log_path, 'w', encoding='utf-8')
            self._debug_log("=== Debug Log Started ===")
        except Exception as e:
            print(f"Could not open debug log: {e}")
            self.debug_log_file = None

    def _debug_log(self, message: str):
        """Write debug message to log file"""
        if self.debug_log_file:
            import time
            timestamp = time.strftime("%H:%M:%S")
            self.debug_log_file.write(f"[{timestamp}] {message}\n")
            self.debug_log_file.flush()

    def _init_gl(self):
        """Initialize OpenGL"""
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_LIGHT1)
        glEnable(GL_COLOR_MATERIAL)
        glEnable(GL_NORMALIZE)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        # Main light
        glLightfv(GL_LIGHT0, GL_POSITION, [5.0, 5.0, 10.0, 0.0])
        glLightfv(GL_LIGHT0, GL_AMBIENT, [0.4, 0.4, 0.45, 1.0])
        glLightfv(GL_LIGHT0, GL_DIFFUSE, [0.9, 0.9, 0.85, 1.0])

        # Fill light
        glLightfv(GL_LIGHT1, GL_POSITION, [-5.0, -3.0, 5.0, 0.0])
        glLightfv(GL_LIGHT1, GL_DIFFUSE, [0.3, 0.3, 0.4, 1.0])

        # Sky color
        glClearColor(0.5, 0.7, 0.9, 1.0)

        # Perspective
        glMatrixMode(GL_PROJECTION)
        gluPerspective(60, self.width / self.height, 0.1, 100.0)
        glMatrixMode(GL_MODELVIEW)

    def _spawn_default_table(self):
        """Spawn the default table at origin"""
        import numpy as np
        # Table position is at origin (center of table is at x=0, z=0)
        # The table surface is at y=0.76 (standard height)
        self.entity_manager.summon('table', np.array([0.0, 0.0, 0.0]), {})

    def _update_camera(self):
        """First-person camera (Y-up coordinate system)"""
        glLoadIdentity()

        yaw_rad = math.radians(self.camera_yaw)
        pitch_rad = math.radians(self.camera_pitch)

        # Y-up: look direction in XZ plane, pitch affects Y
        look_x = self.camera_pos[0] + math.cos(pitch_rad) * math.cos(yaw_rad)
        look_y = self.camera_pos[1] + math.sin(pitch_rad)
        look_z = self.camera_pos[2] + math.cos(pitch_rad) * math.sin(yaw_rad)

        gluLookAt(
            self.camera_pos[0], self.camera_pos[1], self.camera_pos[2],
            look_x, look_y, look_z,
            0, 1, 0  # Y is up
        )

    def _draw_ground(self):
        """Draw the floor (Y-up: ground is XZ plane at y=0)"""
        glDisable(GL_LIGHTING)

        # Main floor
        glColor4f(0.3, 0.4, 0.3, 1.0)
        glBegin(GL_QUADS)
        glVertex3f(-10, 0, -10)
        glVertex3f(10, 0, -10)
        glVertex3f(10, 0, 10)
        glVertex3f(-10, 0, 10)
        glEnd()

        # Grid
        glColor4f(0.35, 0.45, 0.35, 0.7)
        glLineWidth(1.0)
        glBegin(GL_LINES)
        for i in range(-10, 11):
            glVertex3f(i, 0.002, -10)
            glVertex3f(i, 0.002, 10)
            glVertex3f(-10, 0.002, i)
            glVertex3f(10, 0.002, i)
        glEnd()

        glEnable(GL_LIGHTING)

    def _draw_table_entity(self, table):
        """Draw table entity (Y-up coordinate system)"""
        # Get table properties from entity
        pos = table.position
        hl = table.length / 2  # X direction (half length)
        hw = table.width / 2   # Z direction (half width)
        h = table.height       # Y direction (surface height)
        th = table.thickness
        nh = table.net_height

        glPushMatrix()
        glTranslatef(*pos)

        # Apply rotation if any
        angle_deg = math.degrees(table.orientation_angle)
        if abs(angle_deg) > 0.01:
            glRotatef(angle_deg, *table.orientation_axis)

        # Table top (XZ plane at height Y=h)
        glColor3f(0.0, 0.25, 0.5)
        glBegin(GL_QUADS)
        glNormal3f(0, 1, 0)  # Up
        glVertex3f(-hl, h, -hw)
        glVertex3f(hl, h, -hw)
        glVertex3f(hl, h, hw)
        glVertex3f(-hl, h, hw)
        glEnd()

        # Table sides
        glColor3f(0.0, 0.2, 0.4)
        glBegin(GL_QUADS)
        # Front (negative Z)
        glNormal3f(0, 0, -1)
        glVertex3f(-hl, h - th, -hw)
        glVertex3f(hl, h - th, -hw)
        glVertex3f(hl, h, -hw)
        glVertex3f(-hl, h, -hw)
        # Back (positive Z)
        glNormal3f(0, 0, 1)
        glVertex3f(-hl, h - th, hw)
        glVertex3f(hl, h - th, hw)
        glVertex3f(hl, h, hw)
        glVertex3f(-hl, h, hw)
        # Left (negative X)
        glNormal3f(-1, 0, 0)
        glVertex3f(-hl, h - th, -hw)
        glVertex3f(-hl, h - th, hw)
        glVertex3f(-hl, h, hw)
        glVertex3f(-hl, h, -hw)
        # Right (positive X)
        glNormal3f(1, 0, 0)
        glVertex3f(hl, h - th, -hw)
        glVertex3f(hl, h - th, hw)
        glVertex3f(hl, h, hw)
        glVertex3f(hl, h, -hw)
        glEnd()

        # White lines
        glDisable(GL_LIGHTING)
        glColor3f(1.0, 1.0, 1.0)
        glLineWidth(3.0)
        glBegin(GL_LINE_LOOP)
        glVertex3f(-hl + 0.02, h + 0.002, -hw + 0.02)
        glVertex3f(hl - 0.02, h + 0.002, -hw + 0.02)
        glVertex3f(hl - 0.02, h + 0.002, hw - 0.02)
        glVertex3f(-hl + 0.02, h + 0.002, hw - 0.02)
        glEnd()
        glBegin(GL_LINES)
        glVertex3f(0, h + 0.002, -hw)
        glVertex3f(0, h + 0.002, hw)
        glVertex3f(-hl, h + 0.002, 0)
        glVertex3f(hl, h + 0.002, 0)
        glEnd()
        glEnable(GL_LIGHTING)

        # Net (YZ plane at X=0)
        glColor4f(0.95, 0.95, 0.95, 0.9)
        glBegin(GL_QUADS)
        glVertex3f(0, h, -hw - 0.15)
        glVertex3f(0, h, hw + 0.15)
        glVertex3f(0, h + nh, hw + 0.15)
        glVertex3f(0, h + nh, -hw - 0.15)
        glEnd()

        # Net posts
        glColor3f(0.2, 0.2, 0.2)
        self._draw_cylinder_yup(0, h, -hw - 0.15, 0.015, nh)
        self._draw_cylinder_yup(0, h, hw + 0.15, 0.015, nh)

        # Table legs
        glColor3f(0.25, 0.25, 0.25)
        for lx, lz in [(-hl + 0.15, -hw + 0.1), (-hl + 0.15, hw - 0.1),
                       (hl - 0.15, -hw + 0.1), (hl - 0.15, hw - 0.1)]:
            self._draw_cylinder_yup(lx, 0, lz, 0.025, h - th)

        glPopMatrix()

    def _draw_cylinder_yup(self, x, y, z, radius, height, segments=12):
        """Draw cylinder (Y-up: cylinder extends in Y direction)"""
        glBegin(GL_QUAD_STRIP)
        for i in range(segments + 1):
            angle = 2 * math.pi * i / segments
            nx, nz = math.cos(angle), math.sin(angle)
            glNormal3f(nx, 0, nz)
            glVertex3f(x + radius * nx, y, z + radius * nz)
            glVertex3f(x + radius * nx, y + height, z + radius * nz)
        glEnd()

    def update_physics(self):
        """Update entity physics"""
        if self.paused:
            return

        dt = self.params.dt * self.time_scale
        self.entity_manager.update(dt, self.params)

        # Record frame for save/replay if active
        self._record_frame()

    def handle_movement(self):
        """Handle player movement (Y-up coordinate system)"""
        if self.console_open or self.menu_open or self.data_popup_open:
            return

        # In play mode, camera is fixed
        if self.play_mode.active:
            cam_pos, cam_yaw, cam_pitch = self.play_mode.get_camera_state()
            self.camera_pos = cam_pos
            self.camera_yaw = cam_yaw
            self.camera_pitch = cam_pitch
            return

        keys = pygame.key.get_pressed()

        yaw_rad = math.radians(self.camera_yaw)
        # Y-up: forward is in XZ plane
        forward = np.array([math.cos(yaw_rad), 0, math.sin(yaw_rad)])
        # right = forward × up (cross product for right-hand coordinate system)
        right = np.array([-math.sin(yaw_rad), 0, math.cos(yaw_rad)])

        # WASD movement
        if keys[K_w]:
            self.camera_pos += forward * self.camera_speed
        if keys[K_s]:
            self.camera_pos -= forward * self.camera_speed
        if keys[K_a]:
            self.camera_pos -= right * self.camera_speed
        if keys[K_d]:
            self.camera_pos += right * self.camera_speed

        # Mouse side buttons for up/down (Y is height)
        if self.mouse_side2_held:  # Side button 2 = up
            self.camera_pos[1] += self.camera_speed
        if self.mouse_side1_held:  # Side button 1 = down
            self.camera_pos[1] -= self.camera_speed

        # Keep above ground
        self.camera_pos[1] = max(0.3, self.camera_pos[1])

        # Mouse look (always active when not in menu/console)
        rel = pygame.mouse.get_rel()
        self.camera_yaw += rel[0] * self.mouse_sensitivity  # Y-up: positive = right turn
        self.camera_pitch -= rel[1] * self.mouse_sensitivity
        self.camera_pitch = max(-80, min(80, self.camera_pitch))

        # Update player angle-axis rotation from yaw/pitch
        self._update_player_rotation()

    def _update_player_rotation(self):
        """Convert camera yaw/pitch to angle-axis rotation format"""
        yaw = math.radians(self.camera_yaw)
        pitch = math.radians(self.camera_pitch)

        # Player forward direction based on yaw/pitch
        forward = np.array([
            math.cos(pitch) * math.cos(yaw),
            math.sin(pitch),
            math.cos(pitch) * math.sin(yaw)
        ])

        # Default forward is +X (like racket default)
        default_forward = np.array([1.0, 0.0, 0.0])

        # Compute rotation axis and angle using cross product and dot product
        dot = np.dot(default_forward, forward)
        dot = np.clip(dot, -1.0, 1.0)

        if dot > 0.9999:
            # No rotation needed
            self.player_rotation_angle = 0.0
            self.player_rotation_axis = np.array([0.0, 1.0, 0.0])
        elif dot < -0.9999:
            # 180 degree rotation around Y
            self.player_rotation_angle = math.pi
            self.player_rotation_axis = np.array([0.0, 1.0, 0.0])
        else:
            self.player_rotation_angle = math.acos(dot)
            axis = np.cross(default_forward, forward)
            norm = np.linalg.norm(axis)
            if norm > 1e-6:
                self.player_rotation_axis = axis / norm
            else:
                self.player_rotation_axis = np.array([0.0, 1.0, 0.0])

    def get_player_rotation_angle_axis(self):
        """Get player rotation as (angle, axis) tuple"""
        return self.player_rotation_angle, self.player_rotation_axis.copy()

    def process_command(self, cmd):
        """Process command - supports both old and new Minecraft-style commands"""
        cmd_original = cmd.strip()
        cmd_lower = cmd_original.lower()
        parts = cmd_lower.split()
        if not parts:
            return

        command = parts[0]
        args = parts[1:]

        try:
            # New Minecraft-style commands (case-sensitive for NBT)
            if command in ['summon', 'execute', 'kill', 'gamemode', 'data', 'function', 'tp', 'rotate', 'tag', 'start', 'stop', 'play', 'played']:
                result = self.command_parser.parse(cmd_original)
                self._handle_parsed_command(result)
                return

            elif command in ['slow', 'slowmo']:
                factor = float(args[0]) if args else 0.2
                self.time_scale = factor
                self.add_output(f"Speed: {factor}x")

            elif command in ['normal', 'fast']:
                self.time_scale = 1.0
                self.add_output("Normal speed")

            elif command in ['pause', 'p']:
                self.paused = not self.paused
                self.add_output("Paused" if self.paused else "Resumed")

            elif command in ['reset', 'r']:
                self.entity_manager.kill('@e')
                self.add_output("Reset!")

            elif command in ['tp', 'teleport']:
                if len(args) >= 3:
                    self.camera_pos = np.array([float(args[0]), float(args[1]), float(args[2])])
                    self.add_output(f"TP to ({args[0]}, {args[1]}, {args[2]})")
                elif args and args[0] == 'table':
                    self.camera_pos = np.array([-2.5, 1.5, 1.5])
                    self.add_output("TP to table view")

            elif command in ['help', 'h', '?']:
                self.add_output("summon, start, stop, kill, tp, reset")
                self.add_output("function, data get/modify, slow")

            elif command in ['clear', 'cls']:
                self.console_output = []

            elif command == 'topspin':
                # Topspin demo using entity system
                self.process_command('summon ball -1 1 0 {velocity:[12,3,0],spin:[0,0,4000]}')
                self.entity_manager.start()
                self.add_output("Topspin demo started")

            elif command == 'backspin':
                # Backspin demo using entity system
                self.process_command('summon ball 1 1.2 0 {velocity:[-8,4,0],spin:[0,0,-3500]}')
                self.entity_manager.start()
                self.add_output("Backspin demo started")

            elif command == 'smash':
                # Smash demo using entity system
                self.process_command('summon ball -1 1.5 0 {velocity:[20,-2,0],spin:[0,0,2000]}')
                self.entity_manager.start()
                self.add_output("Smash demo started")

            elif command == 'save':
                # Start recording: /save <name>
                if args:
                    self._start_recording(args[0])
                else:
                    self.add_output("Usage: /save <name>")

            elif command == 'saved':
                # End recording and save: /saved <name>
                if args:
                    self._stop_recording(args[0])
                elif self.recording_name:
                    self._stop_recording(self.recording_name)
                else:
                    self.add_output("Usage: /saved <name>")

            elif command == 'replay':
                # Replay saved recording: /replay <name>
                if args:
                    self._replay_recording(args[0])
                else:
                    self.add_output("Usage: /replay <name>")

            else:
                self.add_output(f"Unknown: {command}")

        except Exception as e:
            self.add_output(f"Error: {e}")

    def _handle_parsed_command(self, result):
        """Handle parsed command from new command system"""
        cmd_type = result.get('type')

        if cmd_type == 'summon':
            args = result['args']
            nbt = args['nbt']
            entity = self.entity_manager.summon(
                args['entity'],
                args['position'],
                nbt
            )
            pos = args['position']
            # Debug log to file
            nbt_tags = nbt.get('Tags', 'none')
            vel_actual = entity.velocity if hasattr(entity, 'velocity') else 'none'
            entity_tags = getattr(entity, 'tags', [])
            self._debug_log(f"SUMMON {args['entity']} id={entity.id} pos=({pos[0]:.2f},{pos[1]:.2f},{pos[2]:.2f}) nbt_Tags={nbt_tags} -> entity_tags={entity_tags} vel={vel_actual}")
            self.add_output(f"Summoned {args['entity']} [{entity.id}]")

            # If recording, assign tag and capture exact spawn data
            if self.recording_active:
                entity_type = args['entity']
                self._recording_assign_tag(entity, entity_type)
                # Store exact spawn data for balls (position and velocity at spawn moment)
                if entity_type == 'ball':
                    tag = self.recording_entity_tags.get(id(entity))
                    if tag:
                        timestamp_ms = pygame.time.get_ticks() - self.recording_start_time
                        if not hasattr(self, '_recording_ball_spawns'):
                            self._recording_ball_spawns = {}
                        self._recording_ball_spawns[tag] = {
                            'timestamp': timestamp_ms,
                            'position': entity.position.copy(),
                            'velocity': entity.velocity.copy(),
                            'spin': entity.spin.copy() if hasattr(entity, 'spin') else np.zeros(3)
                        }
                        self._debug_log(f"BALL_SPAWN_CAPTURED tag={tag} pos={entity.position} vel={entity.velocity}")

        elif cmd_type == 'kill':
            selector = result['args']['selector']
            # Use parser's selector resolver for full tag/type support
            entities = self.command_parser._resolve_selector_multiple(selector)
            count = 0
            for entity in entities:
                # Only skip play_controlled entities if currently in play mode
                if self.play_mode.active and 'play_controlled' in getattr(entity, 'tags', []):
                    continue
                self.entity_manager._remove_entity(entity)
                count += 1
            self.add_output(f"Killed {count} entities")

        elif cmd_type == 'start':
            selector = result['args'].get('selector')
            if selector:
                # Start specific entities
                entities = self.command_parser._resolve_selector_multiple(selector)
                count = 0
                for entity in entities:
                    entity.active = True
                    count += 1
                self.entity_manager.simulation_running = True
                self._debug_log(f"START selector={selector} count={count}")
                self.add_output(f"Started {count} entities")
            else:
                # Start all
                self.entity_manager.start()
                balls_active = sum(1 for b in self.entity_manager.balls if b.active)
                balls_total = len(self.entity_manager.balls)
                rackets_total = len(self.entity_manager.rackets)
                self._debug_log(f"START all: balls={balls_active}/{balls_total} rackets={rackets_total} running={self.entity_manager.simulation_running}")
                self.add_output("Started simulation")

        elif cmd_type == 'stop':
            selector = result['args'].get('selector')
            if selector:
                # Stop specific entities
                entities = self.command_parser._resolve_selector_multiple(selector)
                count = 0
                for entity in entities:
                    entity.active = False
                    count += 1
                self.add_output(f"Stopped {count} entities")
            else:
                # Stop all
                self.entity_manager.stop()
                self.add_output("Stopped simulation")

        elif cmd_type == 'gamemode':
            args = result['args']
            setting = args['setting']
            value = args['value']
            if setting == 'gravity':
                self.params.gravity = value
                self.add_output(f"Gravity set to {value}")
            else:
                self.add_output(f"Unknown setting: {setting}")

        elif cmd_type == 'data_get':
            args = result['args']
            entity = args['entity']
            path = args['path']
            nbt_data = self._get_entity_nbt(entity)
            if path:
                # Get specific path - show in console
                if path in nbt_data:
                    self.add_output(f"{path}: {nbt_data[path]}")
                else:
                    self.add_output(f"Unknown path: {path}")
            else:
                # Show all NBT data in popup
                self._show_data_popup(entity, nbt_data)

        elif cmd_type == 'data_modify':
            args = result['args']
            entity = args['entity']
            path = args['path']
            value = args['value']
            entity_tags = getattr(entity, 'tags', [])
            # Only skip play_controlled entities if currently in play mode
            # (allows pure replay to work, but lets player control racket during play+replay)
            if self.play_mode.active and 'play_controlled' in entity_tags:
                return
            # Skip replay_freed entities (balls hit by play_controlled racket return to physics)
            if 'replay_freed' in entity_tags:
                return
            success = self._set_entity_nbt(entity, path, value)
            # Debug log to file (all data_modify commands)
            self._debug_log(f"DATA_MODIFY entity={entity.id} tags={entity_tags} path={path} success={success}")

        elif cmd_type == 'function':
            func_name = result['args']['name']
            self._run_function(func_name)

        elif cmd_type == 'tp':
            args = result['args']
            entity = args['entity']
            position = args['position']
            # Only skip play_controlled entities if currently in play mode
            if self.play_mode.active and 'play_controlled' in getattr(entity, 'tags', []):
                return
            # Check if entity is the player (pseudo-entity)
            if entity.id == "player":
                # Update actual camera position
                self.camera_pos = position.copy()
            else:
                entity.position = position.copy()
            self.add_output(f"Teleported {entity.id} to ({position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f})")

        elif cmd_type == 'rotate':
            args = result['args']
            entity = args['entity']
            mode = args.get('mode', 'angle_axis')
            # Only skip play_controlled entities if currently in play mode
            if self.play_mode.active and 'play_controlled' in getattr(entity, 'tags', []):
                return

            if mode == 'yaw_pitch':
                # Direct yaw/pitch rotation
                yaw = args['yaw']
                pitch = args['pitch']
                if entity.id == "player":
                    # Update actual camera rotation
                    self.camera_yaw = yaw
                    self.camera_pitch = pitch
                    self.add_output(f"Rotated player to yaw={yaw:.1f}, pitch={pitch:.1f}")
                elif hasattr(entity, 'rotation'):
                    entity.rotation = np.array([yaw, pitch])
                    self.add_output(f"Rotated {entity.id} to yaw={yaw:.1f}, pitch={pitch:.1f}")
                else:
                    self.add_output(f"Entity {entity.id} does not support yaw/pitch rotation")
            else:
                # angle-axis rotation (for entities)
                angle = args['angle']
                axis = args['axis']
                if entity.id == "player":
                    self.add_output(f"Use 'rotate @s <yaw> <pitch>' for player rotation")
                elif hasattr(entity, 'orientation_angle'):
                    entity.orientation_angle = angle
                    entity.orientation_axis = axis.copy()
                    self.add_output(f"Rotated {entity.id}")
                else:
                    self.add_output(f"Entity {entity.id} does not support rotation")

        elif cmd_type == 'tag_add':
            args = result['args']
            selector = args['selector']
            tagname = args['tagname']
            entities = self.command_parser._resolve_selector_multiple(selector)
            count = 0
            for entity in entities:
                if hasattr(entity, 'tags') and tagname not in entity.tags:
                    entity.tags.append(tagname)
                    count += 1
            self.add_output(f"Added tag '{tagname}' to {count} entities")

        elif cmd_type == 'tag_remove':
            args = result['args']
            selector = args['selector']
            tagname = args['tagname']
            entities = self.command_parser._resolve_selector_multiple(selector)
            count = 0
            for entity in entities:
                if hasattr(entity, 'tags') and tagname in entity.tags:
                    entity.tags.remove(tagname)
                    count += 1
            self.add_output(f"Removed tag '{tagname}' from {count} entities")

        elif cmd_type == 'tag_list':
            args = result['args']
            selector = args['selector']
            entity = self.command_parser._resolve_selector(selector)
            if entity and hasattr(entity, 'tags'):
                tags = entity.tags
                if tags:
                    self.add_output(f"Tags: {tags}")
                else:
                    self.add_output(f"Tags: []")
            else:
                self.add_output("No entity found or entity has no tags")

        elif cmd_type == 'play':
            args = result['args']
            mode = args['mode']
            racket = args['racket']
            table = args['table']
            side = args['side']

            # Enter play mode
            self.play_mode.enter(mode, racket, table, side)

            # Set camera to fixed position
            cam_pos, cam_yaw, cam_pitch = self.play_mode.get_camera_state()
            self.camera_pos = cam_pos
            self.camera_yaw = cam_yaw
            self.camera_pitch = cam_pitch

            # Show mouse cursor for play mode
            pygame.mouse.set_visible(True)
            pygame.event.set_grab(False)

            self.add_output(f"Play mode: {mode}, Side {side}")

        elif cmd_type == 'played':
            # Exit play mode
            if self.play_mode.active:
                self.play_mode.exit()
                # Restore FPS camera controls
                pygame.mouse.set_visible(False)
                pygame.event.set_grab(True)
                self.add_output("Exited play mode")
            else:
                self.add_output("Not in play mode")

        elif cmd_type == 'error':
            error_msg = result.get('message', 'Error')
            self._debug_log(f"ERROR: {error_msg}")
            # Only show first error in console to avoid flooding
            if not hasattr(self, '_error_count'):
                self._error_count = 0
            self._error_count += 1
            if self._error_count <= 3:
                self.add_output(f"Error: {error_msg}")
            elif self._error_count == 4:
                self.add_output("(more errors in debug_log.txt)")

        elif cmd_type == 'unknown':
            self._debug_log(f"UNKNOWN: {result.get('raw', '')}")
            self.add_output(f"Unknown command: {result.get('raw', '')}")

    def _run_function(self, func_name: str):
        """Run a function file from the functions folder with delay support.

        Delay syntax: 1000;command  (1000ms delay)
        If delay is omitted, uses the previous delay value (default 0).
        """
        import os

        # Try to find the function file
        base_path = os.path.dirname(os.path.abspath(__file__))
        func_path = os.path.join(base_path, 'functions', f'{func_name}.mcfunction')

        if not os.path.exists(func_path):
            # Try without .mcfunction extension (maybe they included it)
            alt_path = os.path.join(base_path, 'functions', func_name)
            if os.path.exists(alt_path):
                func_path = alt_path
            else:
                self.add_output(f"Function not found: {func_name}")
                return

        try:
            with open(func_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            # Record start time for this function
            self.function_start_time = pygame.time.get_ticks()
            self._data_modify_count = 0  # Reset debug counter
            self._error_count = 0  # Reset error counter
            self._debug_log(f"=== FUNCTION START: {func_name} ===")

            cmd_count = 0
            current_delay = 0  # Current delay in ms (inherited by subsequent lines)

            for line in lines:
                line = line.strip()

                # Skip empty lines and comments
                if not line or line.startswith('#'):
                    continue

                # Remove leading / if present
                if line.startswith('/'):
                    line = line[1:]

                # Parse delay syntax: "1000;command" or just "command"
                delay_match = re.match(r'^(\d+);(.+)$', line)
                if delay_match:
                    current_delay = int(delay_match.group(1))
                    command = delay_match.group(2).strip()
                else:
                    command = line

                if current_delay == 0:
                    # Execute immediately
                    self.process_command(command)
                else:
                    # Schedule for later
                    execute_time = self.function_start_time + current_delay
                    self.scheduled_commands.append((execute_time, command))

                cmd_count += 1

            scheduled_count = len([c for c in self.scheduled_commands if c[0] > pygame.time.get_ticks()])
            if scheduled_count > 0:
                self.add_output(f"Function {func_name}: {cmd_count} commands ({scheduled_count} scheduled)")
            else:
                self.add_output(f"Executed {func_name} ({cmd_count} commands)")

        except Exception as e:
            self.add_output(f"Error running function: {e}")

    def _process_scheduled_commands(self):
        """Process scheduled commands that are ready to execute."""
        if not self.scheduled_commands:
            return

        current_time = pygame.time.get_ticks()

        # Find and execute ready commands
        still_pending = []
        for execute_time, command in self.scheduled_commands:
            if current_time >= execute_time:
                self.process_command(command)
            else:
                still_pending.append((execute_time, command))

        self.scheduled_commands = still_pending

    def _start_recording(self, name: str):
        """Start recording world state for save/replay (memo-based approach)"""
        self.recording_active = True
        self.recording_name = name
        self.recording_start_time = pygame.time.get_ticks()
        self.recording_last_frame_time = self.recording_start_time
        self.recording_memo = []  # List of frame snapshots
        self._debug_ball_captured = set()  # Reset debug tracking
        self._recording_ball_spawns = {}  # Exact spawn data for balls

        # Track all entities with unique tags
        self.recording_entity_tags = {}  # Maps id(entity) -> tag
        self.recording_tracked_entities = set()
        self.recording_tag_counter = {'ball': 0, 'racket': 0, 'table': 0}

        # Register all existing entities and assign tags
        for ball in self.entity_manager.balls:
            self._recording_assign_tag(ball, 'ball')
        for racket in self.entity_manager.rackets:
            self._recording_assign_tag(racket, 'racket')
        for table in self.entity_manager.tables:
            self._recording_assign_tag(table, 'table')

        # Record initial frame (timestamp 0)
        self._recording_capture_frame(0)

        entity_count = len(self.recording_tracked_entities)
        self.add_output(f"Recording started: {name} ({entity_count} entities)")

    def _recording_assign_tag(self, entity, entity_type: str):
        """Assign a unique tag to an entity for tracking"""
        tag_num = self.recording_tag_counter.get(entity_type, 0)
        tag = f"rec_{entity_type}_{tag_num}"
        self.recording_tag_counter[entity_type] = tag_num + 1
        self.recording_entity_tags[id(entity)] = tag
        self.recording_tracked_entities.add(id(entity))

    def _recording_capture_frame(self, timestamp_ms: int):
        """Capture current state of all entities into memo"""
        frame = {
            'timestamp': timestamp_ms,
            'entities': []
        }

        # Capture balls
        for ball in self.entity_manager.balls:
            if id(ball) in self.recording_entity_tags:
                # Debug: log first capture of this ball
                tag = self.recording_entity_tags[id(ball)]
                if not hasattr(self, '_debug_ball_captured'):
                    self._debug_ball_captured = set()
                if tag not in self._debug_ball_captured:
                    self._debug_log(f"FIRST_CAPTURE ball tag={tag} pos={ball.position}")
                    self._debug_ball_captured.add(tag)
                frame['entities'].append({
                    'type': 'ball',
                    'tag': tag,
                    'position': ball.position.copy(),
                    'spin': ball.spin.copy(),
                    'active': ball.active,
                    'tags': getattr(ball, 'tags', []).copy()
                })

        # Capture rackets
        for racket in self.entity_manager.rackets:
            if id(racket) in self.recording_entity_tags:
                entity_data = {
                    'type': 'racket',
                    'tag': self.recording_entity_tags[id(racket)],
                    'position': racket.position.copy(),
                    'rotation_angle': racket.orientation_angle,
                    'rotation_axis': racket.orientation_axis.copy(),
                    'active': racket.active,
                    'tags': getattr(racket, 'tags', []).copy()
                }
                if hasattr(racket, 'orientation_angle2'):
                    entity_data['rotation2_angle'] = racket.orientation_angle2
                    entity_data['rotation2_axis'] = racket.orientation_axis2.copy()
                frame['entities'].append(entity_data)

        # Capture tables
        for table in self.entity_manager.tables:
            if id(table) in self.recording_entity_tags:
                frame['entities'].append({
                    'type': 'table',
                    'tag': self.recording_entity_tags[id(table)],
                    'position': table.position.copy(),
                    'tags': getattr(table, 'tags', []).copy()
                })

        self.recording_memo.append(frame)

    def _stop_recording(self, name: str):
        """Stop recording and save to file - converts memo (with positions) to commands (with velocities)"""
        if not self.recording_active:
            self.add_output("No active recording")
            return

        self.recording_active = False

        if len(self.recording_memo) < 1:
            self.add_output("No frames recorded")
            return

        # Convert memo to commands
        commands = self._memo_to_commands()

        # Create saves folder if it doesn't exist
        import os
        base_path = os.path.dirname(os.path.abspath(__file__))
        saves_path = os.path.join(base_path, 'saves')
        os.makedirs(saves_path, exist_ok=True)

        # Write to .mcfunction file
        file_path = os.path.join(saves_path, f'{name}.mcfunction')
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(f"# Save recording: {name}\n")
                f.write(f"# Recorded at: {pygame.time.get_ticks()}\n")
                f.write(f"# Frames: {len(self.recording_memo)}, Commands: {len(commands)}\n\n")

                for delay_ms, command in commands:
                    if delay_ms > 0:
                        f.write(f"{delay_ms};{command}\n")
                    else:
                        f.write(f"{command}\n")

            self.add_output(f"Saved: {name}.mcfunction ({len(commands)} commands from {len(self.recording_memo)} frames)")
        except Exception as e:
            self.add_output(f"Error saving: {e}")

        self.recording_memo = []
        self.recording_name = ""
        self.recording_entity_tags = {}
        self.recording_tracked_entities = set()
        self.recording_tag_counter = {}

    def _memo_to_commands(self):
        """Convert memo (position-based snapshots) to commands (velocity-based)"""
        commands = []

        if len(self.recording_memo) < 1:
            return commands

        # Initial kill commands
        commands.append((0, "kill @e[type=ball]"))
        commands.append((0, "kill @e[type=racket]"))

        # Get first frame for initial entity states
        first_frame = self.recording_memo[0]

        # Build entity lookup by tag for easier access
        def get_entity_by_tag(frame, tag):
            for e in frame['entities']:
                if e['tag'] == tag:
                    return e
            return None

        # Summon entities from first frame
        for entity in first_frame['entities']:
            tag = entity['tag']
            entity_type = entity['type']
            pos = entity['position']
            tags = entity.get('tags', [])
            # Filter out rec_ prefix tags and play_controlled (replay should control all entities)
            all_tags = [t for t in tags if not t.startswith('rec_') and t != 'play_controlled'] + [tag]
            tags_str = ','.join(f'"{t}"' for t in all_tags)

            # Calculate initial velocity from position change to next frame
            initial_vel = np.zeros(3)
            if len(self.recording_memo) > 1:
                next_frame = self.recording_memo[1]
                next_entity = get_entity_by_tag(next_frame, tag)
                if next_entity:
                    dt = (next_frame['timestamp'] - first_frame['timestamp']) / 1000.0  # ms to seconds
                    if dt > 0:
                        initial_vel = (next_entity['position'] - entity['position']) / dt

            vel_angle, vel_axis, vel_speed = self._velocity_to_angle_axis_speed(initial_vel)
            vel_nbt = f"velocity:{{angle:{vel_angle:.4f},axis:[{vel_axis[0]:.4f},{vel_axis[1]:.4f},{vel_axis[2]:.4f}],speed:{vel_speed:.4f}}}"

            if entity_type == 'ball':
                # Use exact spawn data if available (captures position/velocity at spawn moment)
                if hasattr(self, '_recording_ball_spawns') and tag in self._recording_ball_spawns:
                    spawn_data = self._recording_ball_spawns[tag]
                    pos = spawn_data['position']  # Override with exact spawn position
                    spawn_vel = spawn_data['velocity']
                    vel_angle, vel_axis, vel_speed = self._velocity_to_angle_axis_speed(spawn_vel)
                    vel_nbt = f"velocity:{{angle:{vel_angle:.4f},axis:[{vel_axis[0]:.4f},{vel_axis[1]:.4f},{vel_axis[2]:.4f}],speed:{vel_speed:.4f}}}"
                    spin = spawn_data.get('spin', np.zeros(3))
                    self._debug_log(f"BALL_SUMMON_CMD using spawn data: tag={tag} pos={pos} vel={spawn_vel}")
                else:
                    spin = entity.get('spin', np.zeros(3))
                    self._debug_log(f"BALL_SUMMON_CMD using frame data: tag={tag} pos={pos}")
                nbt = f"{{Tags:[{tags_str}],{vel_nbt},spin:[{spin[0]:.1f},{spin[1]:.1f},{spin[2]:.1f}]}}"
            elif entity_type == 'racket':
                rot_angle = entity.get('rotation_angle', 0)
                rot_axis = entity.get('rotation_axis', [0, 1, 0])
                rot_nbt = f"rotation:{{angle:{rot_angle:.4f},axis:[{rot_axis[0]:.4f},{rot_axis[1]:.4f},{rot_axis[2]:.4f}]}}"
                rot2_nbt = ""
                if 'rotation2_angle' in entity:
                    rot2_axis = entity.get('rotation2_axis', [0, 1, 0])
                    rot2_nbt = f",rotation2:{{angle:{entity['rotation2_angle']:.4f},axis:[{rot2_axis[0]:.4f},{rot2_axis[1]:.4f},{rot2_axis[2]:.4f}]}}"
                nbt = f"{{Tags:[{tags_str}],{vel_nbt},{rot_nbt}{rot2_nbt}}}"
            elif entity_type == 'table':
                nbt = f"{{Tags:[{tags_str}]}}"
            else:
                nbt = f"{{Tags:[{tags_str}]}}"

            cmd = f"summon {entity_type} {pos[0]:.4f} {pos[1]:.4f} {pos[2]:.4f} {nbt}"
            commands.append((0, cmd))

        # Start simulation
        commands.append((0, "start"))

        # Track entities that have been summoned
        summoned_tags = set(entity['tag'] for entity in first_frame['entities'])

        # Helper to generate summon command for an entity
        def generate_summon_cmd(entity, timestamp, next_entity=None, dt=0.016):
            tag = entity['tag']
            entity_type = entity['type']
            pos = entity['position']
            tags = entity.get('tags', [])
            # Filter out rec_ prefix tags and play_controlled (replay should control all entities)
            all_tags = [t for t in tags if not t.startswith('rec_') and t != 'play_controlled'] + [tag]
            tags_str = ','.join(f'"{t}"' for t in all_tags)

            # Calculate initial velocity from position change to next frame
            initial_vel = np.zeros(3)
            if next_entity and dt > 0:
                initial_vel = (next_entity['position'] - entity['position']) / dt

            vel_angle, vel_axis, vel_speed = self._velocity_to_angle_axis_speed(initial_vel)
            vel_nbt = f"velocity:{{angle:{vel_angle:.4f},axis:[{vel_axis[0]:.4f},{vel_axis[1]:.4f},{vel_axis[2]:.4f}],speed:{vel_speed:.4f}}}"

            if entity_type == 'ball':
                # Use exact spawn data if available (captures position/velocity at spawn moment)
                if hasattr(self, '_recording_ball_spawns') and tag in self._recording_ball_spawns:
                    spawn_data = self._recording_ball_spawns[tag]
                    pos = spawn_data['position']  # Override with exact spawn position
                    spawn_vel = spawn_data['velocity']
                    vel_angle, vel_axis, vel_speed = self._velocity_to_angle_axis_speed(spawn_vel)
                    vel_nbt = f"velocity:{{angle:{vel_angle:.4f},axis:[{vel_axis[0]:.4f},{vel_axis[1]:.4f},{vel_axis[2]:.4f}],speed:{vel_speed:.4f}}}"
                    spin = spawn_data.get('spin', np.zeros(3))
                    self._debug_log(f"BALL_SUMMON_CMD (mid-rec) using spawn data: tag={tag} pos={pos} vel={spawn_vel}")
                else:
                    spin = entity.get('spin', np.zeros(3))
                    self._debug_log(f"BALL_SUMMON_CMD (mid-rec) using frame data: tag={tag} pos={pos}")
                nbt = f"{{Tags:[{tags_str}],{vel_nbt},spin:[{spin[0]:.1f},{spin[1]:.1f},{spin[2]:.1f}]}}"
            elif entity_type == 'racket':
                rot_angle = entity.get('rotation_angle', 0)
                rot_axis = entity.get('rotation_axis', [0, 1, 0])
                rot_nbt = f"rotation:{{angle:{rot_angle:.4f},axis:[{rot_axis[0]:.4f},{rot_axis[1]:.4f},{rot_axis[2]:.4f}]}}"
                rot2_nbt = ""
                if 'rotation2_angle' in entity:
                    rot2_axis = entity.get('rotation2_axis', [0, 1, 0])
                    rot2_nbt = f",rotation2:{{angle:{entity['rotation2_angle']:.4f},axis:[{rot2_axis[0]:.4f},{rot2_axis[1]:.4f},{rot2_axis[2]:.4f}]}}"
                nbt = f"{{Tags:[{tags_str}],{vel_nbt},{rot_nbt}{rot2_nbt}}}"
            else:
                nbt = f"{{Tags:[{tags_str}]}}"

            return f"summon {entity_type} {pos[0]:.4f} {pos[1]:.4f} {pos[2]:.4f} {nbt}"

        # Process subsequent frames - calculate velocity from position changes
        for i in range(1, len(self.recording_memo)):
            frame = self.recording_memo[i]
            prev_frame = self.recording_memo[i - 1]
            timestamp = frame['timestamp']
            dt = (frame['timestamp'] - prev_frame['timestamp']) / 1000.0  # ms to seconds

            if dt <= 0:
                continue

            for entity in frame['entities']:
                tag = entity['tag']
                entity_type = entity['type']
                selector = f"@e[tag={tag}]"

                prev_entity = get_entity_by_tag(prev_frame, tag)

                # Check if this is a new entity (not seen before)
                if tag not in summoned_tags:
                    # Generate summon command for this new entity
                    # Look ahead to next frame for velocity calculation
                    next_frame = self.recording_memo[i + 1] if i + 1 < len(self.recording_memo) else None
                    next_entity = get_entity_by_tag(next_frame, tag) if next_frame else None
                    next_dt = (next_frame['timestamp'] - frame['timestamp']) / 1000.0 if next_frame else dt
                    summon_cmd = generate_summon_cmd(entity, timestamp, next_entity, next_dt)
                    commands.append((timestamp, summon_cmd))
                    summoned_tags.add(tag)
                    continue  # Skip velocity update for the first frame of this entity

                if not prev_entity:
                    continue

                # Skip velocity updates for balls - they respond to physics naturally
                # Only rackets and tables need velocity/rotation updates
                if entity_type == 'ball':
                    continue

                # Calculate velocity from position change
                vel = (entity['position'] - prev_entity['position']) / dt
                vel_angle, vel_axis, vel_speed = self._velocity_to_angle_axis_speed(vel)

                # Record velocity change
                commands.append((timestamp, f"data modify entity {selector} velocity set value {{angle:{vel_angle:.4f},axis:[{vel_axis[0]:.4f},{vel_axis[1]:.4f},{vel_axis[2]:.4f}],speed:{vel_speed:.4f}}}"))

                # Record rotation changes for rackets
                if entity_type == 'racket':
                    rot_angle = entity.get('rotation_angle', 0)
                    rot_axis = entity.get('rotation_axis', [0, 1, 0])
                    commands.append((timestamp, f"data modify entity {selector} rotation set value {{angle:{rot_angle:.4f},axis:[{rot_axis[0]:.4f},{rot_axis[1]:.4f},{rot_axis[2]:.4f}]}}"))
                    if 'rotation2_angle' in entity:
                        rot2_axis = entity.get('rotation2_axis', [0, 1, 0])
                        commands.append((timestamp, f"data modify entity {selector} rotation2 set value {{angle:{entity['rotation2_angle']:.4f},axis:[{rot2_axis[0]:.4f},{rot2_axis[1]:.4f},{rot2_axis[2]:.4f}]}}"))

        return commands

    def _replay_recording(self, name: str):
        """Replay a saved recording"""
        import os
        base_path = os.path.dirname(os.path.abspath(__file__))
        saves_path = os.path.join(base_path, 'saves')
        file_path = os.path.join(saves_path, f'{name}.mcfunction')

        if not os.path.exists(file_path):
            self.add_output(f"Save not found: {name}")
            return

        # Use existing function runner
        self._run_function_from_path(file_path, name)

    def _run_function_from_path(self, file_path: str, name: str):
        """Run a function file from a specific path"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            self.function_start_time = pygame.time.get_ticks()
            self._data_modify_count = 0  # Reset debug counter
            self._error_count = 0  # Reset error counter
            self._debug_log(f"=== REPLAY START: {name} ===")
            cmd_count = 0
            current_delay = 0

            for line in lines:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                if line.startswith('/'):
                    line = line[1:]

                delay_match = re.match(r'^(\d+);(.+)$', line)
                if delay_match:
                    current_delay = int(delay_match.group(1))
                    command = delay_match.group(2).strip()
                else:
                    command = line
                    current_delay = 0  # Reset delay for non-prefixed lines

                if current_delay == 0:
                    self.process_command(command)
                else:
                    execute_time = self.function_start_time + current_delay
                    self.scheduled_commands.append((execute_time, command))

                cmd_count += 1

            scheduled_count = len([c for c in self.scheduled_commands if c[0] > pygame.time.get_ticks()])
            self.add_output(f"Replay {name}: {cmd_count} commands ({scheduled_count} scheduled)")

        except Exception as e:
            self.add_output(f"Error replaying: {e}")

    def _record_frame(self):
        """Record current frame state if recording is active (memo-based approach)"""
        if not self.recording_active:
            return

        current_time = pygame.time.get_ticks()
        if current_time - self.recording_last_frame_time < self.recording_frame_interval:
            return

        timestamp_ms = current_time - self.recording_start_time

        # Check for new entities (spawned during recording)
        for ball in self.entity_manager.balls:
            if id(ball) not in self.recording_entity_tags:
                self._recording_assign_tag(ball, 'ball')
        for racket in self.entity_manager.rackets:
            if id(racket) not in self.recording_entity_tags:
                self._recording_assign_tag(racket, 'racket')
        for table in self.entity_manager.tables:
            if id(table) not in self.recording_entity_tags:
                self._recording_assign_tag(table, 'table')

        # Capture current frame into memo
        self._recording_capture_frame(timestamp_ms)

        self.recording_last_frame_time = current_time

    def _velocity_to_angle_axis_speed(self, vel):
        """Convert velocity vector [vx,vy,vz] to {angle, axis, speed} format"""
        speed = np.linalg.norm(vel)
        if speed < 1e-6:
            return 0.0, [0.0, 1.0, 0.0], 0.0

        # Normalize to get direction
        direction = vel / speed

        # Default direction is [1,0,0] (X axis)
        default_dir = np.array([1.0, 0.0, 0.0])

        # Calculate rotation axis (cross product)
        axis = np.cross(default_dir, direction)
        axis_norm = np.linalg.norm(axis)

        if axis_norm < 1e-6:
            # Vectors are parallel or anti-parallel
            if np.dot(default_dir, direction) > 0:
                # Same direction, no rotation needed
                return 0.0, [0.0, 1.0, 0.0], speed
            else:
                # Opposite direction, 180 degree rotation around Y
                return math.pi, [0.0, 1.0, 0.0], speed

        axis = axis / axis_norm

        # Calculate rotation angle
        angle = math.acos(np.clip(np.dot(default_dir, direction), -1.0, 1.0))

        return angle, axis.tolist(), speed

    def _get_entity_nbt(self, entity) -> dict:
        """Get NBT data from entity"""
        from src.command.objects import BallEntity, RacketEntity, TableEntity
        import math

        # Helper to convert velocity vector to angle-axis format
        def vel_to_angle_axis(vel):
            speed = np.linalg.norm(vel)
            if speed < 1e-6:
                return "{angle:0.000, axis:[0,1,0], speed:0.000}"
            direction = vel / speed
            # Calculate angle from default forward [0,0,1]
            default = np.array([0.0, 0.0, 1.0])
            dot = np.clip(np.dot(default, direction), -1.0, 1.0)
            angle = math.acos(dot)
            # Calculate axis
            if abs(angle) < 1e-6 or abs(angle - math.pi) < 1e-6:
                axis = np.array([0, 1, 0])
            else:
                axis = np.cross(default, direction)
                axis = axis / np.linalg.norm(axis)
            return f"{{angle:{angle:.3f}, axis:[{axis[0]:.2f},{axis[1]:.2f},{axis[2]:.2f}], speed:{speed:.3f}}}"

        # Helper to convert spin to RPM format
        def spin_to_rpm(spin):
            omega = np.linalg.norm(spin)
            if omega < 1e-6:
                return "{rpm:0, axis:[0,1,0]}"
            axis = spin / omega
            rpm = omega * 60 / (2 * math.pi)
            return f"{{rpm:{rpm:.0f}, axis:[{axis[0]:.2f},{axis[1]:.2f},{axis[2]:.2f}]}}"

        nbt = {
            'id': entity.id,
            'type': entity.entity_type.value,
            'position': f"[{entity.position[0]:.3f}, {entity.position[1]:.3f}, {entity.position[2]:.3f}]",
            'velocity': vel_to_angle_axis(entity.velocity),
            'active': entity.active
        }

        # Tags (common)
        if hasattr(entity, 'tags') and entity.tags:
            nbt['Tags'] = str(entity.tags)

        if isinstance(entity, BallEntity):
            nbt['mass'] = f"{entity.mass * 1000:.1f}g"
            nbt['radius'] = f"{entity.radius * 1000:.1f}mm"
            nbt['spin'] = spin_to_rpm(entity.spin)
            nbt['rotation'] = f"{{angle:{entity.orientation_angle:.3f}, axis:[{entity.orientation_axis[0]:.2f}, {entity.orientation_axis[1]:.2f}, {entity.orientation_axis[2]:.2f}]}}"
            # Acceleration
            if hasattr(entity, 'accel_speed') and entity.accel_speed != 0:
                nbt['acceleration'] = f"{{angle:{entity.accel_angle:.3f}, axis:[{entity.accel_axis[0]:.2f},{entity.accel_axis[1]:.2f},{entity.accel_axis[2]:.2f}], speed:{entity.accel_speed:.3f}}}"
            if hasattr(entity, 'circular'):
                nbt['circular'] = f"[{entity.circular[0]:.2f}, {entity.circular[1]:.2f}, {entity.circular[2]:.2f}]"

        elif isinstance(entity, RacketEntity):
            nbt['mass'] = f"{entity.mass * 1000:.1f}g"
            nbt['rubber'] = f"[{entity.rubber_red.rubber_type.value}, {entity.rubber_black.rubber_type.value}]"
            nbt['coefficient'] = f"[{entity.coefficient[0]:.2f}, {entity.coefficient[1]:.2f}]"
            nbt['restitution'] = f"[{entity.restitution[0]:.2f}, {entity.restitution[1]:.2f}]"
            nbt['rotation'] = f"{{angle:{entity.orientation_angle:.3f}, axis:[{entity.orientation_axis[0]:.2f}, {entity.orientation_axis[1]:.2f}, {entity.orientation_axis[2]:.2f}]}}"
            # Secondary rotation
            if hasattr(entity, 'orientation_angle2'):
                nbt['rotation2'] = f"{{angle:{entity.orientation_angle2:.3f}, axis:[{entity.orientation_axis2[0]:.2f}, {entity.orientation_axis2[1]:.2f}, {entity.orientation_axis2[2]:.2f}]}}"
            if hasattr(entity, 'circular'):
                nbt['circular'] = f"[{entity.circular[0]:.2f}, {entity.circular[1]:.2f}, {entity.circular[2]:.2f}]"

        elif isinstance(entity, TableEntity):
            nbt['mass'] = f"{entity.mass:.1f}kg"
            nbt['length'] = f"{entity.length:.3f}m"
            nbt['width'] = f"{entity.width:.3f}m"
            nbt['height'] = f"{entity.height:.3f}m"
            nbt['restitution'] = f"{entity.restitution:.2f}"
            nbt['coefficient'] = f"{entity.coefficient:.2f}"
            nbt['rotation'] = f"{{angle:{entity.orientation_angle:.3f}, axis:[{entity.orientation_axis[0]:.2f}, {entity.orientation_axis[1]:.2f}, {entity.orientation_axis[2]:.2f}]}}"

        return nbt

    def _set_entity_nbt(self, entity, path: str, value) -> bool:
        """Set NBT data on entity"""
        from src.command.objects import BallEntity, RacketEntity, TableEntity, RubberSideData, RubberType
        import numpy as np

        try:
            # Common properties
            if path == 'active':
                entity.active = bool(value)
                return True

            # Position
            if path == 'pos' or path == 'position':
                if isinstance(value, list):
                    entity.position = np.array(value, dtype=float)
                    return True
                return False

            # Velocity (common - supports angle-axis format)
            if path == 'velocity':
                if isinstance(value, dict):
                    # Format: {angle:X, axis:[x,y,z], speed:N}
                    # Rotate default direction [1,0,0] around axis by angle
                    # (Same as player rotation default)
                    angle = float(value.get('angle', 0))
                    axis = np.array(value.get('axis', [0, 1, 0]), dtype=float)
                    norm = np.linalg.norm(axis)
                    if norm > 0:
                        axis = axis / norm
                    speed = float(value.get('speed', 0))

                    # Rodrigues rotation: rotate [1,0,0] around axis by angle
                    default_dir = np.array([1.0, 0.0, 0.0])
                    if abs(angle) > 1e-6:
                        k = axis
                        v = default_dir
                        cos_a = np.cos(angle)
                        sin_a = np.sin(angle)
                        direction = v * cos_a + np.cross(k, v) * sin_a + k * np.dot(k, v) * (1 - cos_a)
                    else:
                        direction = default_dir
                    entity.velocity = direction * speed
                    return True
                elif isinstance(value, list):
                    entity.velocity = np.array(value, dtype=float)
                    return True
                return False

            # Acceleration (angle-axis format for balls and rackets)
            if path == 'acceleration':
                if isinstance(value, dict):
                    if hasattr(entity, 'accel_angle'):
                        entity.accel_angle = float(value.get('angle', 0))
                        axis = value.get('axis', [0, 1, 0])
                        if isinstance(axis, list):
                            entity.accel_axis = np.array(axis, dtype=float)
                            norm = np.linalg.norm(entity.accel_axis)
                            if norm > 0:
                                entity.accel_axis = entity.accel_axis / norm
                        entity.accel_speed = float(value.get('speed', 0))
                        return True
                return False

            # Circular motion (caret notation: [left, up, forward])
            if path == 'circular':
                if hasattr(entity, 'circular'):
                    if isinstance(value, list):
                        entity.circular = np.array(value, dtype=float)
                        return True
                return False

            # Spin (for balls)
            if path == 'spin':
                if hasattr(entity, 'spin'):
                    if isinstance(value, dict):
                        # {rpm:3000, axis:[0,1,0]}
                        rpm = float(value.get('rpm', 0))
                        axis = np.array(value.get('axis', [0, 1, 0]), dtype=float)
                        norm = np.linalg.norm(axis)
                        if norm > 0:
                            axis = axis / norm
                        omega = rpm * 2 * np.pi / 60  # RPM to rad/s
                        entity.spin = axis * omega
                        return True
                    elif isinstance(value, list):
                        entity.spin = np.array(value, dtype=float)
                        return True
                return False

            # Rotation (common to ball, racket, table, and player)
            if path == 'rotation':
                # Support list format [angle, axis_x, axis_y, axis_z] for recordings
                if isinstance(value, list) and len(value) == 4:
                    if hasattr(entity, 'orientation_angle'):
                        entity.orientation_angle = float(value[0])
                        entity.orientation_axis = np.array([value[1], value[2], value[3]], dtype=float)
                        norm = np.linalg.norm(entity.orientation_axis)
                        if norm > 0:
                            entity.orientation_axis = entity.orientation_axis / norm
                        return True
                    return False
                elif isinstance(value, dict):
                    # Check if this is the player entity
                    if hasattr(entity, 'id') and entity.id == "player":
                        # Player rotation: support both {yaw:X, pitch:Y} and {angle:X, axis:[...]}
                        if 'yaw' in value or 'pitch' in value:
                            # Direct yaw/pitch format
                            self.camera_yaw = float(value.get('yaw', self.camera_yaw))
                            self.camera_pitch = float(value.get('pitch', self.camera_pitch))
                            return True
                        elif 'angle' in value:
                            # angle-axis format - convert to yaw/pitch approximation
                            # For simplicity, use yaw as angle when axis is [0,1,0] (Y-up)
                            angle = float(value.get('angle', 0))
                            axis = value.get('axis', [0, 1, 0])
                            if isinstance(axis, list):
                                axis = np.array(axis, dtype=float)
                            # Convert angle from radians to degrees
                            angle_deg = np.degrees(angle)
                            # If axis is primarily Y (up), treat as yaw
                            if abs(axis[1]) > 0.9:
                                self.camera_yaw = angle_deg
                            # If axis is primarily X, treat as pitch
                            elif abs(axis[0]) > 0.9:
                                self.camera_pitch = angle_deg
                            # If axis is primarily Z, also treat as pitch (looking up/down)
                            elif abs(axis[2]) > 0.9:
                                self.camera_pitch = -angle_deg if axis[2] < 0 else angle_deg
                            else:
                                # Mixed axis - approximate
                                self.camera_yaw = angle_deg * axis[1]
                                self.camera_pitch = angle_deg * axis[0]
                            return True
                        return False
                    elif hasattr(entity, 'orientation_angle'):
                        entity.orientation_angle = float(value.get('angle', 0))
                        axis = value.get('axis', [0, 1, 0])
                        if isinstance(axis, list):
                            entity.orientation_axis = np.array(axis, dtype=float)
                            norm = np.linalg.norm(entity.orientation_axis)
                            if norm > 0:
                                entity.orientation_axis = entity.orientation_axis / norm
                        return True
                return False

            # Ball properties
            if isinstance(entity, BallEntity):
                if path == 'mass':
                    entity.mass = float(value) / 1000.0  # g -> kg
                    return True
                elif path == 'radius':
                    entity.radius = float(value) / 1000.0  # mm -> m
                    return True

            # Racket properties
            elif isinstance(entity, RacketEntity):
                if path == 'mass':
                    entity.mass = float(value) / 1000.0  # g -> kg
                    return True
                elif path == 'rubber_red':
                    rubber_type = self._parse_rubber_type(str(value))
                    entity.rubber_red = RubberSideData.from_type(rubber_type)
                    return True
                elif path == 'rubber_black':
                    rubber_type = self._parse_rubber_type(str(value))
                    entity.rubber_black = RubberSideData.from_type(rubber_type)
                    return True
                elif path == 'coefficient':
                    if isinstance(value, list) and len(value) >= 2:
                        entity.coefficient = [float(value[0]), float(value[1])]
                        return True
                elif path == 'restitution':
                    if isinstance(value, list) and len(value) >= 2:
                        entity.restitution = [float(value[0]), float(value[1])]
                        return True
                elif path == 'rotation2':
                    # Secondary rotation for rackets
                    if isinstance(value, dict):
                        entity.orientation_angle2 = float(value.get('angle', 0))
                        axis2 = value.get('axis', [0, 1, 0])
                        if isinstance(axis2, list):
                            entity.orientation_axis2 = np.array(axis2, dtype=float)
                            norm = np.linalg.norm(entity.orientation_axis2)
                            if norm > 0:
                                entity.orientation_axis2 = entity.orientation_axis2 / norm
                        return True

            # Table properties
            elif isinstance(entity, TableEntity):
                if path == 'mass':
                    entity.mass = float(value)  # kg (no conversion)
                    return True
                elif path == 'length':
                    entity.length = float(value)
                    return True
                elif path == 'width':
                    entity.width = float(value)
                    return True
                elif path == 'height':
                    entity.height = float(value)
                    return True
                elif path == 'restitution':
                    entity.restitution = float(value)
                    return True
                elif path in ('friction', 'coefficient'):
                    entity.coefficient = float(value)
                    return True

            return False
        except (ValueError, AttributeError):
            return False

    def _parse_rubber_type(self, type_str: str):
        """Parse rubber type string to RubberType enum"""
        from src.command.objects import RubberType
        type_map = {
            'inverted': RubberType.INVERTED,
            'pimples': RubberType.PIMPLES,
            'short_pimples': RubberType.PIMPLES,
            'long_pimples': RubberType.LONG_PIMPLES,
            'long': RubberType.LONG_PIMPLES,
            'anti': RubberType.ANTI,
        }
        return type_map.get(type_str.lower(), RubberType.INVERTED)

    def add_output(self, text):
        """Add console output"""
        self.console_output.append(text)
        if len(self.console_output) > self.max_output_lines:
            self.console_output.pop(0)

    def _show_data_popup(self, entity, nbt_data: dict):
        """Show data popup with entity NBT"""
        self.data_popup_title = f"Entity Data [{entity.id}]"
        self.data_popup_content = []

        for key, value in nbt_data.items():
            self.data_popup_content.append(f"{key}: {value}")

        self.data_popup_open = True
        # Show mouse cursor for popup interaction
        pygame.mouse.set_visible(True)
        pygame.event.set_grab(False)

    def _close_data_popup(self):
        """Close data popup and restore mouse state"""
        self.data_popup_open = False
        pygame.mouse.set_visible(False)
        pygame.event.set_grab(True)

    def handle_events(self):
        """Handle events"""
        for event in pygame.event.get():
            if event.type == QUIT:
                self.running = False

            elif event.type == KEYDOWN:
                # Data popup takes priority
                if self.data_popup_open:
                    if event.key == K_ESCAPE or event.key == K_RETURN:
                        self._close_data_popup()
                    continue

                if self.console_open:
                    # Chat input mode with cursor support
                    if event.key == K_RETURN:
                        if self.console_input:
                            self.console_history.append(self.console_input)
                            self.process_command(self.console_input)
                            self.console_input = ""
                            self.console_cursor = 0
                        self.console_open = False
                    elif event.key == K_ESCAPE or event.key == K_SLASH:
                        self.console_open = False
                        self.console_input = ""
                        self.console_cursor = 0
                    elif event.key == K_BACKSPACE:
                        if self.console_cursor > 0:
                            self.console_input = self.console_input[:self.console_cursor-1] + self.console_input[self.console_cursor:]
                            self.console_cursor -= 1
                    elif event.key == K_DELETE:
                        if self.console_cursor < len(self.console_input):
                            self.console_input = self.console_input[:self.console_cursor] + self.console_input[self.console_cursor+1:]
                    elif event.key == K_LEFT:
                        self.console_cursor = max(0, self.console_cursor - 1)
                    elif event.key == K_RIGHT:
                        self.console_cursor = min(len(self.console_input), self.console_cursor + 1)
                    elif event.key == K_HOME:
                        self.console_cursor = 0
                    elif event.key == K_END:
                        self.console_cursor = len(self.console_input)
                    elif event.key == K_UP and self.console_history:
                        if self.history_index < len(self.console_history) - 1:
                            self.history_index += 1
                            self.console_input = self.console_history[-(self.history_index + 1)]
                            self.console_cursor = len(self.console_input)
                    elif event.key == K_DOWN:
                        if self.history_index > 0:
                            self.history_index -= 1
                            self.console_input = self.console_history[-(self.history_index + 1)]
                            self.console_cursor = len(self.console_input)
                        else:
                            self.history_index = -1
                            self.console_input = ""
                            self.console_cursor = 0
                    else:
                        if event.unicode and event.unicode.isprintable() and event.unicode != '/':
                            self.console_input = self.console_input[:self.console_cursor] + event.unicode + self.console_input[self.console_cursor:]
                            self.console_cursor += 1

                elif self.menu_open:
                    # Menu mode
                    if event.key == K_ESCAPE:
                        self.menu_open = False
                        pygame.mouse.set_visible(False)
                        pygame.event.set_grab(True)

                elif self.play_mode.active:
                    # Play mode key handling
                    if event.key == K_SLASH:
                        self.console_open = True
                        self.history_index = -1
                        self.console_cursor = 0
                    elif event.key == K_ESCAPE:
                        # Exit play mode
                        self.play_mode.exit()
                        pygame.mouse.set_visible(False)
                        pygame.event.set_grab(True)
                        self.add_output("Exited play mode")
                    elif event.key == K_F1:
                        self.show_help = not self.show_help
                    elif event.key == K_F3:
                        self.show_debug = not self.show_debug

                else:
                    # Normal game mode
                    if event.key == K_SLASH:
                        self.console_open = True
                        self.history_index = -1
                        self.console_cursor = 0
                    elif event.key == K_ESCAPE:
                        self.menu_open = True
                        pygame.mouse.set_visible(True)
                        pygame.event.set_grab(False)
                    elif event.key == K_F1:
                        self.show_help = not self.show_help
                    elif event.key == K_F3:
                        # F3 alone toggles debug, but check if combo key is pressed
                        keys = pygame.key.get_pressed()
                        if not keys[K_b]:
                            self.show_debug = not self.show_debug
                    elif event.key == K_b:
                        # F3+B: Toggle ball orientation display
                        keys = pygame.key.get_pressed()
                        if keys[K_F3]:
                            self.show_ball_orientation = not self.show_ball_orientation
                            state = "ON" if self.show_ball_orientation else "OFF"
                            self.add_output(f"Ball orientation: {state}")

            elif event.type == MOUSEBUTTONDOWN:
                # Data popup close button
                if self.data_popup_open:
                    mx, my = event.pos
                    bx, by, bw, bh = self.data_popup_close_rect
                    if bx <= mx <= bx + bw and by <= my <= by + bh:
                        self._close_data_popup()
                    continue

                if self.menu_open:
                    # Check menu button clicks
                    mx, my = event.pos
                    if 550 <= mx <= 850:
                        if 350 <= my <= 400:  # Resume
                            self.menu_open = False
                            pygame.mouse.set_visible(False)
                            pygame.event.set_grab(True)
                        elif 420 <= my <= 470:  # Quit
                            self.running = False
                elif self.play_mode.active:
                    # Play mode mouse handling
                    self.play_mode.handle_mouse_down(event.button, event.pos)
                    # Side buttons for racket height in play mode
                    if event.button == 7:
                        self.mouse_side1_held = True  # down
                    elif event.button == 6:
                        self.mouse_side2_held = True  # up
                else:
                    # Debug: show button number for non-standard buttons
                    if event.button not in [1, 2, 3]:
                        self.add_output(f"Mouse btn {event.button} down")

                    # Mouse side buttons
                    # Button 6 = up, Button 7 = down
                    if event.button == 7:
                        self.mouse_side1_held = True  # down
                    elif event.button == 6:
                        self.mouse_side2_held = True  # up
                    # Button 4 = accelerate, Button 5 = decelerate
                    elif event.button == 4:
                        self.camera_speed = min(self.speed_max, self.camera_speed * 1.5)
                        self.add_output(f"Speed: {self.camera_speed:.2f}")
                    elif event.button == 5:
                        self.camera_speed = max(self.speed_min, self.camera_speed / 1.5)
                        self.add_output(f"Speed: {self.camera_speed:.2f}")

            elif event.type == MOUSEBUTTONUP:
                if self.play_mode.active:
                    self.play_mode.handle_mouse_up(event.button, event.pos)
                    # Side buttons for racket height in play mode
                    if event.button == 7:
                        self.mouse_side1_held = False
                    elif event.button == 6:
                        self.mouse_side2_held = False
                elif event.button == 7:
                    self.mouse_side1_held = False
                elif event.button == 6:
                    self.mouse_side2_held = False

            elif event.type == MOUSEMOTION:
                # Handle mouse motion for play mode
                if self.play_mode.active and not self.console_open and not self.menu_open:
                    self.play_mode.handle_mouse_move(event.pos, event.rel)

            elif event.type == VIDEORESIZE:
                # Handle window resize
                self.width, self.height = event.w, event.h
                self.screen = pygame.display.set_mode((self.width, self.height), DOUBLEBUF | OPENGL | RESIZABLE)
                glViewport(0, 0, self.width, self.height)
                glMatrixMode(GL_PROJECTION)
                glLoadIdentity()
                gluPerspective(60, self.width / self.height, 0.1, 100.0)
                glMatrixMode(GL_MODELVIEW)

        return self.running

    def render_hud(self):
        """Render HUD"""
        hud = pygame.Surface((self.width, self.height), pygame.SRCALPHA)

        # Console output
        y = self.height - 50 - len(self.console_output) * 24
        for line in self.console_output:
            text = self.font_small.render(line, True, (255, 255, 255))
            # Shadow
            shadow = self.font_small.render(line, True, (0, 0, 0))
            hud.blit(shadow, (17, y + 2))
            hud.blit(text, (15, y))
            y += 24

        # Console input with cursor
        if self.console_open:
            pygame.draw.rect(hud, (0, 0, 0, 220), (10, self.height - 45, self.width - 20, 40))
            pygame.draw.rect(hud, (80, 80, 80), (10, self.height - 45, self.width - 20, 40), 2)
            # Show cursor at position
            before_cursor = "/" + self.console_input[:self.console_cursor]
            after_cursor = self.console_input[self.console_cursor:]
            # Blinking cursor
            cursor_char = "|" if (pygame.time.get_ticks() // 500) % 2 == 0 else ""
            input_text = before_cursor + cursor_char + after_cursor
            text = self.font_medium.render(input_text, True, (100, 255, 100))
            hud.blit(text, (20, self.height - 40))
        else:
            hint = self.font_small.render("/ - Chat    ESC - Menu", True, (180, 180, 180))
            hud.blit(hint, (15, self.height - 28))

        # F3 Debug screen (Minecraft-style)
        if self.show_debug and not self.menu_open:
            # Normalize yaw to -180 to 180
            yaw = self.camera_yaw % 360
            if yaw > 180:
                yaw -= 360

            # Pitch is already -80 to 80
            pitch = self.camera_pitch

            # Direction facing (Y-up: XZ plane, Y is height)
            if -45 <= yaw < 45:
                facing = "East (+X)"
            elif 45 <= yaw < 135:
                facing = "North (+Z)"
            elif yaw >= 135 or yaw < -135:
                facing = "West (-X)"
            else:
                facing = "South (-Z)"

            # FPS
            fps = self.clock.get_fps()

            # Player rotation in angle-axis format
            player_angle = self.player_rotation_angle
            player_axis = self.player_rotation_axis

            debug_lines = [
                f"Table Tennis Physics Simulation",
                f"",
                f"XYZ: {self.camera_pos[0]:.3f} / {self.camera_pos[1]:.3f} / {self.camera_pos[2]:.3f}",
                f"Facing: {facing} (yaw:{yaw:.1f} pitch:{pitch:.1f})",
                f"Rotation: {{angle:{player_angle:.3f}, axis:[{player_axis[0]:.2f},{player_axis[1]:.2f},{player_axis[2]:.2f}]}}",
                f"Player Speed: {self.camera_speed:.2f}",
                f"",
                f"FPS: {fps:.0f}",
                f"Time Scale: {self.time_scale}x",
            ]

            # Show closest active ball info
            active_balls = [b for b in self.entity_manager.balls if b.active]
            if active_balls:
                # Find closest ball to player (use racket position in play mode, camera otherwise)
                if self.play_mode.active and self.play_mode.racket:
                    ref_pos = self.play_mode.racket.position
                else:
                    ref_pos = self.camera_pos

                closest_ball = None
                closest_dist = float('inf')
                for b in active_balls:
                    dist = np.linalg.norm(b.position - ref_pos)
                    if dist < closest_dist:
                        closest_dist = dist
                        closest_ball = b

                if closest_ball:
                    ball = closest_ball
                    ball_pos = ball.position
                    ball_vel = ball.velocity
                    ball_speed = np.linalg.norm(ball_vel)
                    ball_rpm = np.linalg.norm(ball.spin) * 60 / (2 * math.pi)
                    debug_lines.extend([
                        f"",
                        f"Ball XYZ: {ball_pos[0]:.3f} / {ball_pos[1]:.3f} / {ball_pos[2]:.3f}",
                        f"Ball Vel: {ball_vel[0]:.2f} / {ball_vel[1]:.2f} / {ball_vel[2]:.2f}",
                        f"Ball Speed: {ball_speed:.2f} m/s",
                        f"Ball Spin: {ball_rpm:.0f} RPM",
                    ])

            # Semi-transparent background
            debug_height = len(debug_lines) * 20 + 10
            debug_bg = pygame.Surface((320, debug_height), pygame.SRCALPHA)
            debug_bg.fill((0, 0, 0, 180))
            hud.blit(debug_bg, (5, 5))

            y = 10
            for line in debug_lines:
                if line:
                    text = self.font_small.render(line, True, (255, 255, 255))
                    hud.blit(text, (10, y))
                y += 20

        # Help
        if self.show_help and not self.menu_open and not self.show_debug:
            help_lines = [
                "WASD - Move",
                "Mouse Side 2/1 - Up/Down",
                "/ - Commands    ESC - Menu",
                "F1 - Help    F3 - Debug",
                "F3+B - Ball orientation"
            ]
            y = 15
            for line in help_lines:
                shadow = self.font_small.render(line, True, (0, 0, 0))
                text = self.font_small.render(line, True, (220, 220, 220))
                hud.blit(shadow, (17, y + 1))
                hud.blit(text, (15, y))
                y += 22

        # Paused
        if self.paused and not self.menu_open:
            text = self.font_large.render("PAUSED", True, (255, 100, 100))
            rect = text.get_rect(center=(self.width // 2, 50))
            hud.blit(text, rect)

        # Slow motion
        if self.time_scale < 1.0 and not self.menu_open:
            text = self.font_medium.render(f"SLOW {self.time_scale}x", True, (100, 200, 255))
            hud.blit(text, (self.width // 2 - 50, 80))

        # Crosshair
        if not self.menu_open:
            cx, cy = self.width // 2, self.height // 2
            pygame.draw.line(hud, (255, 255, 255, 150), (cx - 12, cy), (cx + 12, cy), 2)
            pygame.draw.line(hud, (255, 255, 255, 150), (cx, cy - 12), (cx, cy + 12), 2)

        # Menu
        if self.menu_open:
            # Darken background
            overlay = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            hud.blit(overlay, (0, 0))

            # Title
            title = self.font_large.render("MENU", True, (255, 255, 255))
            title_rect = title.get_rect(center=(self.width // 2, 280))
            hud.blit(title, title_rect)

            # Buttons
            mx, my = pygame.mouse.get_pos()

            # Resume button
            resume_color = (100, 200, 100) if 550 <= mx <= 850 and 350 <= my <= 400 else (80, 150, 80)
            pygame.draw.rect(hud, resume_color, (550, 350, 300, 50), border_radius=8)
            resume_text = self.font_medium.render("Resume", True, (255, 255, 255))
            hud.blit(resume_text, (self.width // 2 - resume_text.get_width() // 2, 360))

            # Quit button
            quit_color = (200, 100, 100) if 550 <= mx <= 850 and 420 <= my <= 470 else (150, 80, 80)
            pygame.draw.rect(hud, quit_color, (550, 420, 300, 50), border_radius=8)
            quit_text = self.font_medium.render("Quit", True, (255, 255, 255))
            hud.blit(quit_text, (self.width // 2 - quit_text.get_width() // 2, 430))

        # Data popup
        if self.data_popup_open:
            # Calculate popup dimensions
            popup_width = 500
            line_height = 22
            padding = 15
            title_height = 35
            close_btn_height = 35
            content_height = len(self.data_popup_content) * line_height + padding * 2
            popup_height = title_height + content_height + close_btn_height + padding

            # Center popup
            popup_x = (self.width - popup_width) // 2
            popup_y = (self.height - popup_height) // 2

            # Background with semi-transparent overlay
            overlay = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 120))
            hud.blit(overlay, (0, 0))

            # Popup background
            popup_bg = pygame.Surface((popup_width, popup_height), pygame.SRCALPHA)
            popup_bg.fill((30, 30, 40, 240))
            hud.blit(popup_bg, (popup_x, popup_y))

            # Border
            pygame.draw.rect(hud, (100, 100, 120), (popup_x, popup_y, popup_width, popup_height), 2, border_radius=8)

            # Title bar
            pygame.draw.rect(hud, (50, 50, 70), (popup_x, popup_y, popup_width, title_height), border_radius=8)
            pygame.draw.rect(hud, (50, 50, 70), (popup_x, popup_y + 10, popup_width, title_height - 10))
            title_text = self.font_medium.render(self.data_popup_title or "Entity Data", True, (255, 255, 255))
            hud.blit(title_text, (popup_x + padding, popup_y + 8))

            # Content
            y = popup_y + title_height + padding
            for line in self.data_popup_content:
                # Syntax highlighting for NBT
                if ":" in line and not line.strip().startswith("{"):
                    parts = line.split(":", 1)
                    key_text = self.font_small.render(parts[0] + ":", True, (150, 200, 255))
                    hud.blit(key_text, (popup_x + padding, y))
                    if len(parts) > 1:
                        value_text = self.font_small.render(parts[1], True, (255, 220, 150))
                        hud.blit(value_text, (popup_x + padding + key_text.get_width(), y))
                else:
                    text = self.font_small.render(line, True, (220, 220, 220))
                    hud.blit(text, (popup_x + padding, y))
                y += line_height

            # Close button
            mx, my = pygame.mouse.get_pos()
            close_btn_y = popup_y + popup_height - close_btn_height - padding // 2
            close_btn_x = popup_x + popup_width // 2 - 60
            close_btn_w, close_btn_h = 120, 30

            # Store button rect for click detection
            self.data_popup_close_rect = (close_btn_x, close_btn_y, close_btn_w, close_btn_h)

            btn_hover = close_btn_x <= mx <= close_btn_x + close_btn_w and close_btn_y <= my <= close_btn_y + close_btn_h
            btn_color = (100, 150, 200) if btn_hover else (70, 100, 140)
            pygame.draw.rect(hud, btn_color, (close_btn_x, close_btn_y, close_btn_w, close_btn_h), border_radius=5)
            close_text = self.font_small.render("Close [ESC]", True, (255, 255, 255))
            hud.blit(close_text, (close_btn_x + (close_btn_w - close_text.get_width()) // 2, close_btn_y + 6))

        # Play mode UI
        if self.play_mode.active:
            self.play_mode.render_ui(hud)

        return hud

    def _draw_ball_orientation(self, ball):
        """Draw orientation line from ball center (F3+B feature)"""
        pos = ball.position
        angle = ball.orientation_angle
        axis = ball.orientation_axis

        # Default direction: positive Z (forward in Y-up)
        default_dir = np.array([0.0, 0.0, 1.0])

        # Apply rotation using Rodrigues' formula: v_rot = v*cos(θ) + (k×v)*sin(θ) + k*(k·v)*(1-cos(θ))
        if abs(angle) > 1e-6:
            k = axis
            v = default_dir
            cos_a = math.cos(angle)
            sin_a = math.sin(angle)
            k_cross_v = np.cross(k, v)
            k_dot_v = np.dot(k, v)
            direction = v * cos_a + k_cross_v * sin_a + k * k_dot_v * (1 - cos_a)
        else:
            direction = default_dir

        # Line length (proportional to ball size, visible)
        line_length = ball.radius * 3

        # End point
        end_pos = pos + direction * line_length

        # Draw line
        glDisable(GL_LIGHTING)
        glLineWidth(2.0)
        glBegin(GL_LINES)
        glColor3f(1.0, 1.0, 0.0)  # Yellow
        glVertex3f(*pos)
        glColor3f(1.0, 0.0, 0.0)  # Red at tip
        glVertex3f(*end_pos)
        glEnd()

        # Draw small arrowhead
        glPointSize(6.0)
        glBegin(GL_POINTS)
        glColor3f(1.0, 0.0, 0.0)
        glVertex3f(*end_pos)
        glEnd()

        glEnable(GL_LIGHTING)

    def _draw_racket_entity(self, racket):
        """Draw racket with rubber visualization and orientation line

        Default orientation (angle=0):
        - Blade is horizontal (XZ plane)
        - Red side faces +Y (up)
        - Handle points -Z direction
        """
        from src.command.objects import RubberType

        pos = racket.position
        glPushMatrix()
        glTranslatef(*pos)

        # Apply rotations: rotation (base) first, then rotation2 (local adjustment)
        # rotation2's axis is specified in LOCAL coordinates (after rotation)
        # OpenGL does NOT automatically interpret axes in local coords,
        # so we must explicitly transform rotation2's axis by rotation

        angle = racket.orientation_angle
        axis = racket.orientation_axis

        # Apply primary rotation first
        angle_deg = math.degrees(angle)
        if abs(angle_deg) > 0.01:
            glRotatef(angle_deg, *axis)

        # Apply secondary rotation (axis2 is in local coordinates)
        if hasattr(racket, 'orientation_angle2'):
            angle2 = racket.orientation_angle2
            axis2_local = racket.orientation_axis2
            angle2_deg = math.degrees(angle2)
            if abs(angle2_deg) > 0.01:
                # Transform axis2 from local coords to world coords using rotation
                # axis2_world = R(angle, axis) * axis2_local
                if abs(angle) > 0.01:
                    cos_a = math.cos(angle)
                    sin_a = math.sin(angle)
                    k = np.array(axis)
                    v = np.array(axis2_local)
                    axis2_world = v * cos_a + np.cross(k, v) * sin_a + k * np.dot(k, v) * (1 - cos_a)
                    glRotatef(angle2_deg, *axis2_world)
                else:
                    glRotatef(angle2_deg, *axis2_local)

        # Racket dimensions
        blade_width = 0.15   # X direction
        blade_length = 0.16  # Z direction
        blade_thick = 0.006  # Y direction (thickness)
        rubber_thick = 0.002
        handle_len = 0.10
        handle_radius = 0.012

        # Draw blade core (wood) - ellipsoid in XZ plane, thin in Y
        glColor3f(0.6, 0.45, 0.25)  # Wood color
        glPushMatrix()
        glScalef(blade_width / 2, blade_thick / 2, blade_length / 2)
        quadric = gluNewQuadric()
        gluSphere(quadric, 1.0, 16, 12)
        gluDeleteQuadric(quadric)
        glPopMatrix()

        # Draw red side rubber (+Y side, facing up)
        self._draw_rubber_surface_horizontal(racket.rubber_red, blade_width, blade_length,
                                             blade_thick / 2 + rubber_thick / 2, rubber_thick, True)

        # Draw black side rubber (-Y side, facing down)
        self._draw_rubber_surface_horizontal(racket.rubber_black, blade_width, blade_length,
                                             -(blade_thick / 2 + rubber_thick / 2), rubber_thick, False)

        # Handle (attached to -Z edge of blade, extends in -Z direction)
        glColor3f(0.5, 0.35, 0.2)
        glPushMatrix()
        glTranslatef(0, 0, -blade_length / 2)  # Move to -Z edge of blade
        glRotatef(180, 1, 0, 0)  # Flip to point in -Z direction
        quadric = gluNewQuadric()
        gluCylinder(quadric, handle_radius, handle_radius * 0.9, handle_len, 8, 1)
        gluDeleteQuadric(quadric)
        glPopMatrix()

        glPopMatrix()

        # Draw orientation line (outside the rotation transform)
        self._draw_racket_orientation(racket)

    def _draw_rubber_surface(self, rubber, width, height, z_offset, thickness, is_red_side):
        """Draw rubber surface with type-specific visualization"""
        from src.command.objects import RubberType

        # Base color (red side vs black side)
        if is_red_side:
            base_color = (0.85, 0.1, 0.1)  # Red
        else:
            base_color = (0.15, 0.15, 0.15)  # Black

        rubber_type = rubber.rubber_type

        glPushMatrix()
        glTranslatef(0, 0, z_offset)

        if rubber_type == RubberType.INVERTED:
            # 裏ソフト: Smooth glossy surface
            glColor3f(*base_color)
            glPushMatrix()
            glScalef(width / 2 * 0.95, height / 2 * 0.95, thickness / 2)
            quadric = gluNewQuadric()
            gluSphere(quadric, 1.0, 16, 12)
            gluDeleteQuadric(quadric)
            glPopMatrix()

        elif rubber_type == RubberType.PIMPLES:
            # 表ソフト: Short pimples (small dots pattern)
            glColor3f(*base_color)
            # Base sheet
            glPushMatrix()
            glScalef(width / 2 * 0.95, height / 2 * 0.95, thickness / 4)
            quadric = gluNewQuadric()
            gluSphere(quadric, 1.0, 16, 12)
            gluDeleteQuadric(quadric)
            glPopMatrix()
            # Pimples pattern (small bumps)
            self._draw_pimples_pattern(width * 0.85, height * 0.85, 0.001,
                                       pimple_radius=0.001, spacing=0.0007, is_red_side=is_red_side)

        elif rubber_type == RubberType.LONG_PIMPLES:
            # 粒高: Long pimples (taller, thinner, more flexible look)
            glColor3f(*base_color)
            # Base sheet
            glPushMatrix()
            glScalef(width / 2 * 0.95, height / 2 * 0.95, thickness / 4)
            quadric = gluNewQuadric()
            gluSphere(quadric, 1.0, 16, 12)
            gluDeleteQuadric(quadric)
            glPopMatrix()
            # Long pimples (taller, more prominent)
            self._draw_pimples_pattern(width * 0.85, height * 0.85, 0.0017,
                                       pimple_radius=0.0005, spacing=0.0012, is_red_side=is_red_side,
                                       is_long=True)

        elif rubber_type == RubberType.ANTI:
            # アンチ: Very smooth, matte surface (darker, less reflective)
            # Slightly darker than normal
            dark_color = (base_color[0] * 0.7, base_color[1] * 0.7, base_color[2] * 0.7)
            glColor3f(*dark_color)
            glPushMatrix()
            glScalef(width / 2 * 0.95, height / 2 * 0.95, thickness / 2)
            quadric = gluNewQuadric()
            gluSphere(quadric, 1.0, 16, 12)
            gluDeleteQuadric(quadric)
            glPopMatrix()

        glPopMatrix()

    def _draw_pimples_pattern(self, width, height, pip_height, pimple_radius, spacing, is_red_side, is_long=False):
        """Draw pimples/pips pattern on rubber surface"""
        # Color for pimples
        if is_red_side:
            glColor3f(0.9, 0.2, 0.2)  # Lighter red for pimples
        else:
            glColor3f(0.25, 0.25, 0.25)  # Lighter black for pimples

        quadric = gluNewQuadric()

        # Calculate grid
        cols = int(width / spacing)
        rows = int(height / spacing)

        start_x = -width / 2 + spacing / 2
        start_y = -height / 2 + spacing / 2

        for row in range(rows):
            for col in range(cols):
                x = start_x + col * spacing
                y = start_y + row * spacing

                # Check if inside ellipse
                if (x / (width / 2)) ** 2 + (y / (height / 2)) ** 2 > 0.85:
                    continue

                glPushMatrix()
                glTranslatef(x, y, 0)

                if is_long:
                    # Long pimples: thin cylinders that can wobble
                    # Add slight random tilt for natural look
                    tilt_x = math.sin(x * 50 + y * 30) * 8
                    tilt_y = math.cos(x * 30 + y * 50) * 8
                    glRotatef(tilt_x, 1, 0, 0)
                    glRotatef(tilt_y, 0, 1, 0)
                    gluCylinder(quadric, pimple_radius, pimple_radius * 0.7, pip_height, 6, 1)
                else:
                    # Short pimples: small hemispheres
                    gluSphere(quadric, pimple_radius, 6, 4)

                glPopMatrix()

        gluDeleteQuadric(quadric)

    def _draw_rubber_surface_horizontal(self, rubber, width, length, y_offset, thickness, is_red_side):
        """Draw rubber surface for horizontal blade (XZ plane)"""
        from src.command.objects import RubberType

        # Base color
        if is_red_side:
            base_color = (0.85, 0.1, 0.1)  # Red
        else:
            base_color = (0.15, 0.15, 0.15)  # Black

        rubber_type = rubber.rubber_type

        glPushMatrix()
        glTranslatef(0, y_offset, 0)

        if rubber_type == RubberType.INVERTED:
            # 裏ソフト: Smooth glossy surface
            glColor3f(*base_color)
            glPushMatrix()
            glScalef(width / 2 * 0.95, thickness / 2, length / 2 * 0.95)
            quadric = gluNewQuadric()
            gluSphere(quadric, 1.0, 16, 12)
            gluDeleteQuadric(quadric)
            glPopMatrix()

        elif rubber_type == RubberType.PIMPLES:
            # 表ソフト: Short pimples
            glColor3f(*base_color)
            glPushMatrix()
            glScalef(width / 2 * 0.95, thickness / 4, length / 2 * 0.95)
            quadric = gluNewQuadric()
            gluSphere(quadric, 1.0, 16, 12)
            gluDeleteQuadric(quadric)
            glPopMatrix()
            self._draw_pimples_pattern_horizontal(width * 0.85, length * 0.85, 0.001,
                                                  pimple_radius=0.001, spacing=0.0007,
                                                  is_red_side=is_red_side, facing_up=is_red_side)

        elif rubber_type == RubberType.LONG_PIMPLES:
            # 粒高: Long pimples
            glColor3f(*base_color)
            glPushMatrix()
            glScalef(width / 2 * 0.95, thickness / 4, length / 2 * 0.95)
            quadric = gluNewQuadric()
            gluSphere(quadric, 1.0, 16, 12)
            gluDeleteQuadric(quadric)
            glPopMatrix()
            self._draw_pimples_pattern_horizontal(width * 0.85, length * 0.85, 0.0017,
                                                  pimple_radius=0.0005, spacing=0.0012,
                                                  is_red_side=is_red_side, facing_up=is_red_side,
                                                  is_long=True)

        elif rubber_type == RubberType.ANTI:
            # アンチ: Matte surface
            dark_color = (base_color[0] * 0.7, base_color[1] * 0.7, base_color[2] * 0.7)
            glColor3f(*dark_color)
            glPushMatrix()
            glScalef(width / 2 * 0.95, thickness / 2, length / 2 * 0.95)
            quadric = gluNewQuadric()
            gluSphere(quadric, 1.0, 16, 12)
            gluDeleteQuadric(quadric)
            glPopMatrix()

        glPopMatrix()

    def _draw_pimples_pattern_horizontal(self, width, length, pip_height, pimple_radius, spacing,
                                         is_red_side, facing_up=True, is_long=False):
        """Draw pimples pattern on horizontal rubber surface (XZ plane)"""
        if is_red_side:
            glColor3f(0.9, 0.2, 0.2)
        else:
            glColor3f(0.25, 0.25, 0.25)

        quadric = gluNewQuadric()

        cols = int(width / spacing)
        rows = int(length / spacing)

        start_x = -width / 2 + spacing / 2
        start_z = -length / 2 + spacing / 2

        direction = 1 if facing_up else -1

        for row in range(rows):
            for col in range(cols):
                x = start_x + col * spacing
                z = start_z + row * spacing

                # Check if inside ellipse
                if (x / (width / 2)) ** 2 + (z / (length / 2)) ** 2 > 0.85:
                    continue

                glPushMatrix()
                glTranslatef(x, 0, z)

                if is_long:
                    # Long pimples with slight tilt
                    tilt_x = math.sin(x * 50 + z * 30) * 8
                    tilt_z = math.cos(x * 30 + z * 50) * 8
                    glRotatef(90 * direction, 1, 0, 0)  # Point up or down
                    glRotatef(tilt_x, 1, 0, 0)
                    glRotatef(tilt_z, 0, 0, 1)
                    gluCylinder(quadric, pimple_radius, pimple_radius * 0.7, pip_height, 6, 1)
                else:
                    # Short pimples
                    gluSphere(quadric, pimple_radius, 6, 4)

                glPopMatrix()

        gluDeleteQuadric(quadric)

    def _draw_racket_orientation(self, racket):
        """Draw orientation line from racket center (red side direction)

        Default (angle=0): Red side faces +Y (up)
        The orientation line shows where the red side is facing.
        """
        pos = racket.position
        angle = racket.orientation_angle
        axis = racket.orientation_axis

        # Default direction: positive Y (red side faces up by default)
        default_dir = np.array([0.0, 1.0, 0.0])

        # Apply rotation using Rodrigues' formula
        if abs(angle) > 1e-6:
            k = axis
            v = default_dir
            cos_a = math.cos(angle)
            sin_a = math.sin(angle)
            k_cross_v = np.cross(k, v)
            k_dot_v = np.dot(k, v)
            direction = v * cos_a + k_cross_v * sin_a + k * k_dot_v * (1 - cos_a)
        else:
            direction = default_dir

        # Line length (visible orientation indicator)
        line_length = 0.20  # 20cm line

        # End point
        end_pos = pos + direction * line_length

        # Draw line
        glDisable(GL_LIGHTING)
        glLineWidth(3.0)
        glBegin(GL_LINES)
        glColor3f(0.9, 0.2, 0.2)  # Red (matches red rubber side)
        glVertex3f(*pos)
        glColor3f(1.0, 0.6, 0.0)  # Orange at tip
        glVertex3f(*end_pos)
        glEnd()

        # Draw arrowhead
        glPointSize(8.0)
        glBegin(GL_POINTS)
        glColor3f(1.0, 0.6, 0.0)
        glVertex3f(*end_pos)
        glEnd()

        glEnable(GL_LIGHTING)

    def _draw_entities(self):
        """Draw all entities from entity manager (Y-up coordinate system)"""
        # Draw balls (no coordinate conversion needed - all Y-up now)
        for ball in self.entity_manager.balls:
            pos = ball.position

            # Draw trail
            if len(ball.trail) > 1:
                glDisable(GL_LIGHTING)
                glLineWidth(3.0)
                glBegin(GL_LINE_STRIP)
                for i, trail_pos in enumerate(ball.trail):
                    alpha = (i / len(ball.trail)) ** 0.5
                    glColor4f(0.0, 0.8, 1.0, alpha * 0.7)
                    glVertex3f(*trail_pos)
                glEnd()
                glEnable(GL_LIGHTING)

            # Draw ball
            if ball.active:
                glColor3f(0.0, 0.8, 1.0)  # Cyan for new balls
            else:
                glColor4f(0.0, 0.8, 1.0, 0.5)
            glPushMatrix()
            glTranslatef(*pos)
            quadric = gluNewQuadric()
            gluSphere(quadric, ball.radius, 20, 20)
            gluDeleteQuadric(quadric)
            glPopMatrix()

            # Draw orientation line (F3+B)
            if self.show_ball_orientation:
                self._draw_ball_orientation(ball)

        # Draw rackets
        for racket in self.entity_manager.rackets:
            self._draw_racket_entity(racket)

        # Draw tables
        for table in self.entity_manager.tables:
            self._draw_table_entity(table)

    def render(self):
        """Main render"""
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        self._update_camera()
        self._draw_ground()
        self._draw_entities()

        # HUD overlay
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, self.width, self.height, 0, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()

        glDisable(GL_DEPTH_TEST)
        glDisable(GL_LIGHTING)

        hud = self.render_hud()
        hud_data = pygame.image.tostring(hud, 'RGBA', True)
        glEnable(GL_BLEND)
        glRasterPos2i(0, self.height)
        glDrawPixels(self.width, self.height, GL_RGBA, GL_UNSIGNED_BYTE, hud_data)

        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)

        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        glPopMatrix()

        pygame.display.flip()

    def run(self):
        """Main loop"""
        print("\n" + "=" * 50)
        print("  Table Tennis Physics - Game")
        print("=" * 50)
        print("  WASD - Move")
        print("  Mouse - Look")
        print("  Mouse Side 2/1 - Up/Down")
        print("  / - Open chat commands")
        print("  ESC - Menu")
        print("=" * 50 + "\n")

        while self.running:
            self.handle_events()
            self.handle_movement()

            # Process delayed commands from functions
            self._process_scheduled_commands()

            # Update play mode state
            if self.play_mode.active:
                self.play_mode.update(self.params.dt * self.time_scale)

            for _ in range(3):
                self.update_physics()

            self.render()
            self.clock.tick(self.fps)

        pygame.quit()
        print("\nGame ended.")


def main():
    print("Starting game...")
    game = GameWorld()
    game.run()


if __name__ == "__main__":
    main()
