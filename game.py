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

sys.path.insert(0, '.')

from src.physics.parameters import PhysicsParameters, create_offensive_setup
from src.physics.ball import Ball
from src.physics.table import Table
from src.physics.collision import CollisionHandler
from src.command.parser import CommandParser
from src.command.objects import EntityManager, BallEntity, RacketEntity


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

        self.ball = Ball(self.params)
        self.table = Table(self.params)
        self.collision = CollisionHandler(self.params)

        # New command system
        self.entity_manager = EntityManager()
        self.command_parser = None  # Initialized after self is ready

        # Ball state
        self.ball_active = False
        self.ball_trail = []
        self.max_trail = 150
        self.bounce_markers = []

        # Camera - player viewpoint (Y-up coordinate system)
        self.camera_pos = np.array([-3.0, 1.5, 2.0])  # [x, height, z]
        self.camera_yaw = -30.0
        self.camera_pitch = 15.0
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

        # Stats
        self.bounces = 0
        self.max_speed = 0
        self.flight_time = 0

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

    def _draw_table(self):
        """Draw table tennis table (Y-up coordinate system)"""
        hl = self.params.table_length / 2  # X direction
        hw = self.params.table_width / 2   # Z direction
        h = self.params.table_height       # Y direction (height)
        th = 0.03

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
        nh = self.params.table.net_height
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

    def _draw_ball(self):
        """Draw ball with trail (Y-up coordinate system)"""
        if not self.ball_active:
            # Ghost ball
            glColor4f(1.0, 0.6, 0.0, 0.3)
            glPushMatrix()
            glTranslatef(*self.ball.position)
            quadric = gluNewQuadric()
            gluSphere(quadric, self.params.ball_radius, 16, 16)
            gluDeleteQuadric(quadric)
            glPopMatrix()
            return

        # Trail
        if len(self.ball_trail) > 1:
            glDisable(GL_LIGHTING)
            glLineWidth(3.0)
            glBegin(GL_LINE_STRIP)
            for i, pos in enumerate(self.ball_trail):
                alpha = (i / len(self.ball_trail)) ** 0.5
                glColor4f(1.0, 0.5 + 0.3 * alpha, 0.0, alpha * 0.7)
                glVertex3f(*pos)
            glEnd()
            glEnable(GL_LIGHTING)

        # Bounce markers (on XZ plane, Y is height)
        glDisable(GL_LIGHTING)
        for marker in self.bounce_markers[-5:]:
            glColor4f(0.0, 1.0, 0.0, 0.5)
            glPushMatrix()
            glTranslatef(marker[0], marker[1] + 0.003, marker[2])
            glBegin(GL_LINE_LOOP)
            for i in range(16):
                angle = 2 * math.pi * i / 16
                glVertex3f(0.05 * math.cos(angle), 0, 0.05 * math.sin(angle))
            glEnd()
            glPopMatrix()
        glEnable(GL_LIGHTING)

        # Ball
        glColor3f(1.0, 0.5, 0.0)
        glPushMatrix()
        glTranslatef(*self.ball.position)
        quadric = gluNewQuadric()
        gluSphere(quadric, self.params.ball_radius, 20, 20)
        gluDeleteQuadric(quadric)
        glPopMatrix()

    def update_physics(self):
        """Update physics"""
        if not self.ball_active or self.paused:
            return

        dt = self.params.dt * self.time_scale
        self.ball.update(dt)
        self.flight_time += dt

        speed = self.ball.get_speed()
        if speed > self.max_speed:
            self.max_speed = speed

        self.ball_trail.append(self.ball.position.copy())
        if len(self.ball_trail) > self.max_trail:
            self.ball_trail.pop(0)

        if self.collision.handle_ball_table_collision(self.ball, self.table):
            self.bounces += 1
            self.bounce_markers.append(self.ball.position.copy())
            self.add_output(f"Bounce #{self.bounces}")

        if self.collision.handle_ball_net_collision(self.ball, self.table):
            self.add_output("Net!")

        if self.ball.position[1] < 0:
            self.add_output(f"Landed! Time: {self.flight_time:.2f}s")
            self.ball_active = False
        elif np.linalg.norm(self.ball.position[[0, 2]]) > 8:
            self.add_output("Out of bounds!")
            self.ball_active = False

        # Update entity manager (new system)
        if self.entity_manager.simulation_running:
            self.entity_manager.update(dt, self.params)

    def handle_movement(self):
        """Handle player movement (Y-up coordinate system)"""
        if self.console_open or self.menu_open:
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
            if command in ['summon', 'execute', 'kill', 'gamemode', 'data']:
                result = self.command_parser.parse(cmd_original)
                self._handle_parsed_command(result)
                return

            elif command == 'start':
                self.entity_manager.start()
                count = len(self.entity_manager.entities)
                self.add_output(f"Started simulation ({count} entities)")
                return

            elif command == 'stop':
                self.entity_manager.stop()
                self.add_output("Stopped simulation")
                return

            # Legacy commands (kept for compatibility)
            elif command in ['ball', 'b']:
                if len(args) >= 3:
                    pos = np.array([float(args[0]), float(args[1]), float(args[2])])
                    self.ball.reset(position=pos)
                    self.ball_active = False
                    self.ball_trail = []
                    self.bounce_markers = []
                    self.add_output(f"Ball at ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f})")
                else:
                    self.add_output("Usage: ball <x> <y> <z>")

            elif command in ['launch', 'l', 'fire', 'shoot']:
                if len(args) >= 3:
                    vel = np.array([float(args[0]), float(args[1]), float(args[2])])
                    self.ball.velocity = vel
                    self.ball_active = True
                    self.ball_trail = [self.ball.position.copy()]
                    self.bounce_markers = []
                    self.bounces = 0
                    self.max_speed = np.linalg.norm(vel)
                    self.flight_time = 0
                    self.add_output(f"Launched! {np.linalg.norm(vel):.1f} m/s")
                else:
                    self.add_output("Usage: launch <vx> <vy> <vz>")

            elif command in ['spin', 'sp']:
                if args:
                    spin_type = args[0]
                    rpm = float(args[1]) if len(args) > 1 else 3000
                    spin_rad = rpm * 2 * math.pi / 60
                    # Y-up: spin axis for topspin/backspin is Z, for sidespin is Y
                    if spin_type in ['top', 'topspin', 't']:
                        self.ball.spin = np.array([0, 0, spin_rad])
                        self.add_output(f"Topspin {rpm:.0f} RPM")
                    elif spin_type in ['back', 'backspin', 'b']:
                        self.ball.spin = np.array([0, 0, -spin_rad])
                        self.add_output(f"Backspin {rpm:.0f} RPM")
                    elif spin_type in ['side', 'sidespin', 's']:
                        self.ball.spin = np.array([0, spin_rad, 0])
                        self.add_output(f"Sidespin {rpm:.0f} RPM")
                    elif spin_type in ['none', 'no', '0']:
                        self.ball.spin = np.array([0, 0, 0])
                        self.add_output("No spin")
                else:
                    self.add_output("Usage: spin <top/back/side> [rpm]")

            elif command in ['serve', 'sv']:
                power = float(args[0]) if args else 12
                # Y-up: position is [x, height, z]
                self.ball.reset(position=np.array([-1.2, 0.9, 0.0]))
                angle_rad = math.radians(10)
                # Y-up: velocity is [vx, vy (up), vz]
                vel = np.array([power * math.cos(angle_rad), power * 0.15 + 1, 0])
                self.ball.velocity = vel
                self.ball.spin = np.array([0, 0, 200])  # Topspin on Z axis
                self.ball_active = True
                self.ball_trail = [self.ball.position.copy()]
                self.bounce_markers = []
                self.bounces = 0
                self.max_speed = np.linalg.norm(vel)
                self.flight_time = 0
                self.add_output(f"Serve! {np.linalg.norm(vel):.1f} m/s")

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
                # Y-up: [x, height, z]
                self.ball.reset(position=np.array([0, 1, 0]))
                self.ball_active = False
                self.ball_trail = []
                self.bounce_markers = []
                self.bounces = 0
                self.flight_time = 0
                self.entity_manager.kill('@e')
                self.add_output("Reset!")

            elif command in ['tp', 'teleport']:
                if len(args) >= 3:
                    self.camera_pos = np.array([float(args[0]), float(args[1]), float(args[2])])
                    self.add_output(f"TP to ({args[0]}, {args[1]}, {args[2]})")
                elif args and args[0] == 'ball':
                    # Y-up: offset is [x, height, z]
                    self.camera_pos = self.ball.position.copy() + np.array([-1, 0.5, 0])
                    self.add_output("TP to ball")
                elif args and args[0] == 'table':
                    # Y-up: [x, height, z]
                    self.camera_pos = np.array([-2.5, 1.5, 1.5])
                    self.add_output("TP to table view")

            elif command in ['help', 'h', '?']:
                self.add_output("summon, start, stop, kill, tp, reset")
                self.add_output("ball, launch, spin, serve, slow")

            elif command in ['clear', 'cls']:
                self.console_output = []

            elif command == 'topspin':
                # Y-up: ball x height z, launch vx vy vz
                self.process_command("ball -1 1 0")
                self.process_command("spin top 4000")
                self.process_command("launch 12 3 0")

            elif command == 'backspin':
                # Y-up: ball x height z, launch vx vy vz
                self.process_command("ball 1 1.2 0")
                self.process_command("spin back 3500")
                self.process_command("launch -8 4 0")

            elif command == 'smash':
                # Y-up: ball x height z, launch vx vy vz
                self.process_command("ball -1 1.5 0")
                self.process_command("spin top 2000")
                self.process_command("launch 20 -2 0")

            else:
                self.add_output(f"Unknown: {command}")

        except Exception as e:
            self.add_output(f"Error: {e}")

    def _handle_parsed_command(self, result):
        """Handle parsed command from new command system"""
        cmd_type = result.get('type')

        if cmd_type == 'summon':
            args = result['args']
            entity = self.entity_manager.summon(
                args['entity'],
                args['position'],
                args['nbt']
            )
            pos = args['position']
            self.add_output(f"Summoned {args['entity']} [{entity.id}] at ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f})")

        elif cmd_type == 'kill':
            selector = result['args']['selector']
            count = self.entity_manager.kill(selector)
            self.add_output(f"Killed {count} entities")

        elif cmd_type == 'start':
            self.entity_manager.start()
            self.add_output("Started simulation")

        elif cmd_type == 'stop':
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
            success = self._set_entity_nbt(entity, path, value)
            if success:
                self.add_output(f"Set {path} = {value}")
            else:
                self.add_output(f"Failed to set {path}")

        elif cmd_type == 'error':
            self.add_output(result.get('message', 'Error'))

        elif cmd_type == 'unknown':
            self.add_output(f"Unknown command: {result.get('raw', '')}")

    def _get_entity_nbt(self, entity) -> dict:
        """Get NBT data from entity"""
        from src.command.objects import BallEntity, RacketEntity

        nbt = {
            'id': entity.id,
            'type': entity.entity_type.value,
            'position': f"[{entity.position[0]:.3f}, {entity.position[1]:.3f}, {entity.position[2]:.3f}]",
            'velocity': f"[{entity.velocity[0]:.3f}, {entity.velocity[1]:.3f}, {entity.velocity[2]:.3f}]",
            'active': entity.active
        }

        if isinstance(entity, BallEntity):
            nbt['mass'] = f"{entity.mass * 1000:.1f}g"
            nbt['radius'] = f"{entity.radius * 1000:.1f}mm"
            nbt['spin'] = f"[{entity.spin[0]:.1f}, {entity.spin[1]:.1f}, {entity.spin[2]:.1f}]"
            nbt['rotation'] = f"{{angle:{entity.orientation_angle:.3f}, axis:[{entity.orientation_axis[0]:.2f}, {entity.orientation_axis[1]:.2f}, {entity.orientation_axis[2]:.2f}]}}"

        elif isinstance(entity, RacketEntity):
            nbt['mass'] = f"{entity.mass * 1000:.1f}g"
            nbt['rubber_red'] = entity.rubber_red.rubber_type.value
            nbt['rubber_black'] = entity.rubber_black.rubber_type.value
            nbt['coefficient'] = f"[{entity.coefficient[0]:.2f}, {entity.coefficient[1]:.2f}]"
            nbt['restitution'] = f"[{entity.restitution[0]:.2f}, {entity.restitution[1]:.2f}]"
            nbt['rotation'] = f"{{angle:{entity.orientation_angle:.3f}, axis:[{entity.orientation_axis[0]:.2f}, {entity.orientation_axis[1]:.2f}, {entity.orientation_axis[2]:.2f}]}}"

        return nbt

    def _set_entity_nbt(self, entity, path: str, value) -> bool:
        """Set NBT data on entity"""
        from src.command.objects import BallEntity, RacketEntity

        try:
            if path == 'mass':
                # Convert grams to kg
                entity.mass = float(value) / 1000.0
            elif path == 'radius' and isinstance(entity, BallEntity):
                # Convert mm to m
                entity.radius = float(value) / 1000.0
            elif path == 'active':
                entity.active = bool(value)
            else:
                return False
            return True
        except (ValueError, AttributeError):
            return False

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
                if event.button == 7:
                    self.mouse_side1_held = False
                elif event.button == 6:
                    self.mouse_side2_held = False

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

        # Ball status
        if self.ball_active:
            speed = self.ball.get_speed()
            rpm = self.ball.get_spin_rpm()
            pos = self.ball.position
            lines = [
                f"Speed: {speed:.1f} m/s",
                f"Spin: {rpm:.0f} RPM",
                f"Height: {pos[1]:.2f} m",  # Y is height in Y-up
                f"Bounces: {self.bounces}"
            ]
            y = 15
            for line in lines:
                shadow = self.font_medium.render(line, True, (0, 0, 0))
                text = self.font_medium.render(line, True, (255, 255, 100))
                hud.blit(shadow, (self.width - 178, y + 2))
                hud.blit(text, (self.width - 180, y))
                y += 28

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

            debug_lines = [
                f"Table Tennis Physics Simulation",
                f"",
                f"XYZ: {self.camera_pos[0]:.3f} / {self.camera_pos[1]:.3f} / {self.camera_pos[2]:.3f}",
                f"Facing: {facing}",
                f"Rotation: {yaw:.1f} / {pitch:.1f}",
                f"Player Speed: {self.camera_speed:.2f}",
                f"",
                f"FPS: {fps:.0f}",
                f"Time Scale: {self.time_scale}x",
            ]

            if self.ball_active:
                ball_pos = self.ball.position
                ball_vel = self.ball.velocity
                debug_lines.extend([
                    f"",
                    f"Ball XYZ: {ball_pos[0]:.3f} / {ball_pos[1]:.3f} / {ball_pos[2]:.3f}",
                    f"Ball Vel: {ball_vel[0]:.2f} / {ball_vel[1]:.2f} / {ball_vel[2]:.2f}",
                    f"Ball Speed: {self.ball.get_speed():.2f} m/s",
                    f"Ball Spin: {self.ball.get_spin_rpm():.0f} RPM",
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
        """Draw racket with rubber visualization and orientation line"""
        from src.command.objects import RubberType

        pos = racket.position
        glPushMatrix()
        glTranslatef(*pos)

        # Apply angle-axis rotation
        angle_deg = math.degrees(racket.orientation_angle)
        if abs(angle_deg) > 0.01:
            glRotatef(angle_deg, *racket.orientation_axis)

        # Racket dimensions
        blade_width = 0.15   # Width of blade (X)
        blade_height = 0.16  # Height of blade (Y)
        blade_thick = 0.006  # Thickness of blade core (Z)
        rubber_thick = 0.002  # Rubber thickness
        handle_len = 0.10    # Handle length
        handle_radius = 0.012

        # Draw blade core (wood)
        glColor3f(0.6, 0.45, 0.25)  # Wood color
        glPushMatrix()
        glScalef(blade_width / 2, blade_height / 2, blade_thick / 2)
        quadric = gluNewQuadric()
        gluSphere(quadric, 1.0, 16, 12)
        gluDeleteQuadric(quadric)
        glPopMatrix()

        # Draw red side rubber (+Z side)
        self._draw_rubber_surface(racket.rubber_red, blade_width, blade_height,
                                  blade_thick / 2 + rubber_thick / 2, rubber_thick, True)

        # Draw black side rubber (-Z side)
        self._draw_rubber_surface(racket.rubber_black, blade_width, blade_height,
                                  -(blade_thick / 2 + rubber_thick / 2), rubber_thick, False)

        # Handle (attached to bottom edge of blade, extends down in -Y)
        glColor3f(0.5, 0.35, 0.2)
        glPushMatrix()
        glTranslatef(0, -blade_height / 2, 0)  # Move to bottom of blade
        glRotatef(90, 1, 0, 0)  # Rotate cylinder to point down (-Y)
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
            self._draw_pimples_pattern(width * 0.85, height * 0.85, thickness,
                                       pimple_radius=0.004, spacing=0.015, is_red_side=is_red_side)

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
            self._draw_pimples_pattern(width * 0.85, height * 0.85, thickness * 2,
                                       pimple_radius=0.003, spacing=0.012, is_red_side=is_red_side,
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

    def _draw_racket_orientation(self, racket):
        """Draw orientation line from racket center"""
        pos = racket.position
        angle = racket.orientation_angle
        axis = racket.orientation_axis

        # Default direction: positive Z (forward, facing direction of racket)
        default_dir = np.array([0.0, 0.0, 1.0])

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
        line_length = 0.25  # 25cm line

        # End point
        end_pos = pos + direction * line_length

        # Draw line
        glDisable(GL_LIGHTING)
        glLineWidth(3.0)
        glBegin(GL_LINES)
        glColor3f(0.0, 1.0, 0.5)  # Green-cyan for racket orientation
        glVertex3f(*pos)
        glColor3f(1.0, 1.0, 0.0)  # Yellow at tip
        glVertex3f(*end_pos)
        glEnd()

        # Draw arrowhead
        glPointSize(8.0)
        glBegin(GL_POINTS)
        glColor3f(1.0, 1.0, 0.0)
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

    def render(self):
        """Main render"""
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        self._update_camera()
        self._draw_ground()
        self._draw_table()
        self._draw_ball()
        self._draw_entities()  # Draw new entity system objects

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
