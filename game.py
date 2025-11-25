"""
Table Tennis Physics Simulation - Interactive Game

A Minecraft-like experience where you're IN the game world,
can type commands, and watch the ball fly in real-time.

Press T to open command console, type command, press Enter.
Watch the ball move in real-time 3D!
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
from src.physics.racket import Racket
from src.physics.collision import CollisionHandler


class GameWorld:
    """The 3D game world with real-time physics"""

    def __init__(self, width=1400, height=900):
        self.width = width
        self.height = height

        # Initialize pygame
        pygame.init()
        pygame.font.init()

        # Create window with OpenGL
        self.screen = pygame.display.set_mode((width, height), DOUBLEBUF | OPENGL)
        pygame.display.set_caption("Table Tennis Simulation - Press T for commands")

        # Fonts
        self.font_large = pygame.font.SysFont('consolas', 28)
        self.font_medium = pygame.font.SysFont('consolas', 22)
        self.font_small = pygame.font.SysFont('consolas', 18)

        # Physics
        self.params = create_offensive_setup()
        self.params.dt = 0.002

        self.ball = Ball(self.params)
        self.table = Table(self.params)
        self.racket = Racket(self.params, side=-1)
        self.collision = CollisionHandler(self.params)

        # Ball state
        self.ball_active = False
        self.ball_trail = []
        self.max_trail = 150
        self.bounce_markers = []

        # Camera - player viewpoint
        self.camera_pos = np.array([-3.0, 2.0, 1.5])
        self.camera_yaw = -30.0  # Looking toward table
        self.camera_pitch = 15.0
        self.camera_speed = 0.1

        # Mouse look
        self.mouse_locked = False
        self.mouse_sensitivity = 0.2

        # Console state
        self.console_open = False
        self.console_input = ""
        self.console_history = []
        self.console_output = []
        self.max_output_lines = 8

        # Game state
        self.running = True
        self.paused = False
        self.show_help = True
        self.slow_motion = False
        self.time_scale = 1.0

        # Stats
        self.bounces = 0
        self.max_speed = 0
        self.flight_time = 0

        # Clock
        self.clock = pygame.time.Clock()
        self.fps = 60

        # Initialize OpenGL
        self._init_gl()

        # Welcome message
        self.add_output("Welcome! Press T to type commands")
        self.add_output("Try: ball 0 0 1 then launch 10 0 3")

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

        # Main light (sun-like)
        glLightfv(GL_LIGHT0, GL_POSITION, [5.0, 5.0, 10.0, 0.0])
        glLightfv(GL_LIGHT0, GL_AMBIENT, [0.3, 0.3, 0.35, 1.0])
        glLightfv(GL_LIGHT0, GL_DIFFUSE, [0.9, 0.9, 0.85, 1.0])

        # Fill light
        glLightfv(GL_LIGHT1, GL_POSITION, [-5.0, -3.0, 5.0, 0.0])
        glLightfv(GL_LIGHT1, GL_DIFFUSE, [0.3, 0.3, 0.4, 1.0])

        # Sky color
        glClearColor(0.4, 0.6, 0.8, 1.0)

        # Perspective
        glMatrixMode(GL_PROJECTION)
        gluPerspective(60, self.width / self.height, 0.1, 100.0)
        glMatrixMode(GL_MODELVIEW)

    def _update_camera(self):
        """First-person camera update"""
        glLoadIdentity()

        # Calculate look direction
        yaw_rad = math.radians(self.camera_yaw)
        pitch_rad = math.radians(self.camera_pitch)

        look_x = self.camera_pos[0] + math.cos(pitch_rad) * math.cos(yaw_rad)
        look_y = self.camera_pos[1] + math.cos(pitch_rad) * math.sin(yaw_rad)
        look_z = self.camera_pos[2] + math.sin(pitch_rad)

        gluLookAt(
            self.camera_pos[0], self.camera_pos[1], self.camera_pos[2],
            look_x, look_y, look_z,
            0, 0, 1
        )

    def _draw_ground(self):
        """Draw the floor/ground"""
        glDisable(GL_LIGHTING)

        # Main floor
        glColor4f(0.25, 0.35, 0.25, 1.0)
        glBegin(GL_QUADS)
        glVertex3f(-10, -10, 0)
        glVertex3f(10, -10, 0)
        glVertex3f(10, 10, 0)
        glVertex3f(-10, 10, 0)
        glEnd()

        # Grid lines
        glColor4f(0.3, 0.4, 0.3, 0.5)
        glLineWidth(1.0)
        glBegin(GL_LINES)
        for i in range(-10, 11):
            glVertex3f(i, -10, 0.001)
            glVertex3f(i, 10, 0.001)
            glVertex3f(-10, i, 0.001)
            glVertex3f(10, i, 0.001)
        glEnd()

        glEnable(GL_LIGHTING)

    def _draw_table(self):
        """Draw the table tennis table"""
        hl = self.params.table_length / 2
        hw = self.params.table_width / 2
        h = self.params.table_height
        th = 0.03  # Table thickness

        # Table top - dark blue
        glColor3f(0.0, 0.2, 0.4)
        glBegin(GL_QUADS)
        glNormal3f(0, 0, 1)
        glVertex3f(-hl, -hw, h)
        glVertex3f(hl, -hw, h)
        glVertex3f(hl, hw, h)
        glVertex3f(-hl, hw, h)
        glEnd()

        # Table sides
        glColor3f(0.0, 0.15, 0.3)
        # Front
        glBegin(GL_QUADS)
        glNormal3f(0, -1, 0)
        glVertex3f(-hl, -hw, h - th)
        glVertex3f(hl, -hw, h - th)
        glVertex3f(hl, -hw, h)
        glVertex3f(-hl, -hw, h)
        glEnd()
        # Back
        glBegin(GL_QUADS)
        glNormal3f(0, 1, 0)
        glVertex3f(-hl, hw, h - th)
        glVertex3f(hl, hw, h - th)
        glVertex3f(hl, hw, h)
        glVertex3f(-hl, hw, h)
        glEnd()
        # Left
        glBegin(GL_QUADS)
        glNormal3f(-1, 0, 0)
        glVertex3f(-hl, -hw, h - th)
        glVertex3f(-hl, hw, h - th)
        glVertex3f(-hl, hw, h)
        glVertex3f(-hl, -hw, h)
        glEnd()
        # Right
        glBegin(GL_QUADS)
        glNormal3f(1, 0, 0)
        glVertex3f(hl, -hw, h - th)
        glVertex3f(hl, hw, h - th)
        glVertex3f(hl, hw, h)
        glVertex3f(hl, -hw, h)
        glEnd()

        # White lines on table
        glDisable(GL_LIGHTING)
        glColor3f(1.0, 1.0, 1.0)
        glLineWidth(3.0)

        # Border
        glBegin(GL_LINE_LOOP)
        glVertex3f(-hl + 0.02, -hw + 0.02, h + 0.001)
        glVertex3f(hl - 0.02, -hw + 0.02, h + 0.001)
        glVertex3f(hl - 0.02, hw - 0.02, h + 0.001)
        glVertex3f(-hl + 0.02, hw - 0.02, h + 0.001)
        glEnd()

        # Center line
        glBegin(GL_LINES)
        glVertex3f(0, -hw, h + 0.001)
        glVertex3f(0, hw, h + 0.001)
        # End lines
        glVertex3f(-hl, 0, h + 0.001)
        glVertex3f(hl, 0, h + 0.001)
        glEnd()

        glEnable(GL_LIGHTING)

        # Net
        nh = self.params.table.net_height
        glColor4f(0.9, 0.9, 0.9, 0.8)
        glBegin(GL_QUADS)
        glVertex3f(0, -hw - 0.15, h)
        glVertex3f(0, hw + 0.15, h)
        glVertex3f(0, hw + 0.15, h + nh)
        glVertex3f(0, -hw - 0.15, h + nh)
        glEnd()

        # Net posts
        glColor3f(0.2, 0.2, 0.2)
        self._draw_cylinder(0, -hw - 0.15, h, 0.015, nh)
        self._draw_cylinder(0, hw + 0.15, h, 0.015, nh)

        # Table legs
        glColor3f(0.3, 0.3, 0.3)
        leg_positions = [
            (-hl + 0.15, -hw + 0.1),
            (-hl + 0.15, hw - 0.1),
            (hl - 0.15, -hw + 0.1),
            (hl - 0.15, hw - 0.1),
        ]
        for lx, ly in leg_positions:
            self._draw_cylinder(lx, ly, 0, 0.025, h - th)

    def _draw_cylinder(self, x, y, z, radius, height, segments=12):
        """Draw a cylinder"""
        glBegin(GL_QUAD_STRIP)
        for i in range(segments + 1):
            angle = 2 * math.pi * i / segments
            nx = math.cos(angle)
            ny = math.sin(angle)
            glNormal3f(nx, ny, 0)
            glVertex3f(x + radius * nx, y + radius * ny, z)
            glVertex3f(x + radius * nx, y + radius * ny, z + height)
        glEnd()

    def _draw_ball(self):
        """Draw the ball with trail and effects"""
        if not self.ball_active:
            # Draw ghost ball at set position
            glColor4f(1.0, 0.6, 0.0, 0.3)
            glPushMatrix()
            glTranslatef(*self.ball.position)
            quadric = gluNewQuadric()
            gluSphere(quadric, self.params.ball_radius, 16, 16)
            gluDeleteQuadric(quadric)
            glPopMatrix()
            return

        # Draw trail
        if len(self.ball_trail) > 1:
            glDisable(GL_LIGHTING)
            glLineWidth(3.0)
            glBegin(GL_LINE_STRIP)
            for i, pos in enumerate(self.ball_trail):
                alpha = (i / len(self.ball_trail)) ** 0.5
                # Color gradient from yellow to orange
                glColor4f(1.0, 0.5 + 0.3 * alpha, 0.0, alpha * 0.7)
                glVertex3f(*pos)
            glEnd()
            glEnable(GL_LIGHTING)

        # Draw bounce markers
        glDisable(GL_LIGHTING)
        for marker in self.bounce_markers[-5:]:  # Last 5 bounces
            glColor4f(0.0, 1.0, 0.0, 0.5)
            glPushMatrix()
            glTranslatef(marker[0], marker[1], marker[2] + 0.002)
            glBegin(GL_LINE_LOOP)
            for i in range(16):
                angle = 2 * math.pi * i / 16
                glVertex3f(0.05 * math.cos(angle), 0.05 * math.sin(angle), 0)
            glEnd()
            glPopMatrix()
        glEnable(GL_LIGHTING)

        # Draw ball - bright orange
        glColor3f(1.0, 0.5, 0.0)
        glPushMatrix()
        glTranslatef(*self.ball.position)

        quadric = gluNewQuadric()
        gluSphere(quadric, self.params.ball_radius, 20, 20)
        gluDeleteQuadric(quadric)

        # Spin indicator ring
        if self.ball.get_spin_rate() > 50:
            glDisable(GL_LIGHTING)
            spin_axis = self.ball.spin / np.linalg.norm(self.ball.spin)
            glColor4f(0.0, 1.0, 1.0, 0.6)
            glLineWidth(2.0)
            glBegin(GL_LINE_LOOP)
            # Draw ring perpendicular to spin axis
            r = self.params.ball_radius * 1.3
            for i in range(24):
                angle = 2 * math.pi * i / 24
                # Simple ring in XY plane (approximation)
                glVertex3f(r * math.cos(angle), r * math.sin(angle), 0)
            glEnd()
            glEnable(GL_LIGHTING)

        glPopMatrix()

    def _draw_racket(self):
        """Draw the racket"""
        pos = self.racket.position

        glPushMatrix()
        glTranslatef(*pos)

        # Handle
        glColor3f(0.6, 0.4, 0.2)
        glPushMatrix()
        glRotatef(90, 1, 0, 0)
        quadric = gluNewQuadric()
        gluCylinder(quadric, 0.015, 0.012, 0.15, 8, 1)
        gluDeleteQuadric(quadric)
        glPopMatrix()

        # Blade (red side)
        glColor3f(0.8, 0.1, 0.1)
        glPushMatrix()
        glScalef(0.085, 0.08, 0.008)
        quadric = gluNewQuadric()
        gluSphere(quadric, 1.0, 16, 8)
        gluDeleteQuadric(quadric)
        glPopMatrix()

        glPopMatrix()

    def _draw_sky(self):
        """Draw simple sky background elements"""
        glDisable(GL_LIGHTING)
        glDisable(GL_DEPTH_TEST)

        # Sun
        glColor3f(1.0, 0.95, 0.8)
        glPushMatrix()
        glTranslatef(10, 10, 15)
        quadric = gluNewQuadric()
        gluSphere(quadric, 1.0, 16, 16)
        gluDeleteQuadric(quadric)
        glPopMatrix()

        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)

    def update_physics(self):
        """Update physics simulation"""
        if not self.ball_active or self.paused:
            return

        # Time scaling
        dt = self.params.dt * self.time_scale

        # Update ball
        self.ball.update(dt)
        self.flight_time += dt

        # Track max speed
        speed = self.ball.get_speed()
        if speed > self.max_speed:
            self.max_speed = speed

        # Add to trail
        self.ball_trail.append(self.ball.position.copy())
        if len(self.ball_trail) > self.max_trail:
            self.ball_trail.pop(0)

        # Check collisions
        if self.collision.handle_ball_table_collision(self.ball, self.table):
            self.bounces += 1
            self.bounce_markers.append(self.ball.position.copy())
            self.add_output(f"Bounce #{self.bounces} at ({self.ball.position[0]:.2f}, {self.ball.position[1]:.2f})")

        if self.collision.handle_ball_net_collision(self.ball, self.table):
            self.add_output("Hit the net!")

        # Check out of bounds
        if self.ball.position[2] < 0:
            self.add_output(f"Ball landed! Flight time: {self.flight_time:.2f}s, Bounces: {self.bounces}")
            self.ball_active = False
        elif np.linalg.norm(self.ball.position[:2]) > 8:
            self.add_output("Ball out of bounds!")
            self.ball_active = False

    def handle_movement(self):
        """Handle camera/player movement"""
        if self.console_open:
            return

        keys = pygame.key.get_pressed()

        # Calculate forward/right vectors
        yaw_rad = math.radians(self.camera_yaw)
        forward = np.array([math.cos(yaw_rad), math.sin(yaw_rad), 0])
        right = np.array([math.sin(yaw_rad), -math.cos(yaw_rad), 0])

        # Movement
        if keys[K_w]:
            self.camera_pos += forward * self.camera_speed
        if keys[K_s]:
            self.camera_pos -= forward * self.camera_speed
        if keys[K_a]:
            self.camera_pos -= right * self.camera_speed
        if keys[K_d]:
            self.camera_pos += right * self.camera_speed
        if keys[K_SPACE]:
            self.camera_pos[2] += self.camera_speed
        if keys[K_LSHIFT]:
            self.camera_pos[2] -= self.camera_speed

        # Keep above ground
        self.camera_pos[2] = max(0.5, self.camera_pos[2])

        # Mouse look when right button held or locked
        if self.mouse_locked or pygame.mouse.get_pressed()[2]:
            rel = pygame.mouse.get_rel()
            self.camera_yaw += rel[0] * self.mouse_sensitivity
            self.camera_pitch -= rel[1] * self.mouse_sensitivity
            self.camera_pitch = max(-80, min(80, self.camera_pitch))

    def process_command(self, cmd):
        """Process a typed command"""
        cmd = cmd.strip().lower()
        parts = cmd.split()

        if not parts:
            return

        command = parts[0]
        args = parts[1:]

        try:
            # Ball position
            if command in ['ball', 'b']:
                if len(args) >= 3:
                    pos = np.array([float(args[0]), float(args[1]), float(args[2])])
                    self.ball.reset(position=pos)
                    self.ball_active = False
                    self.ball_trail = []
                    self.bounce_markers = []
                    self.add_output(f"Ball placed at ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f})")
                else:
                    self.add_output("Usage: ball <x> <y> <z>")

            # Launch ball
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
                    speed = np.linalg.norm(vel)
                    self.add_output(f"Launched at {speed:.1f} m/s!")
                else:
                    self.add_output("Usage: launch <vx> <vy> <vz>")

            # Spin
            elif command in ['spin', 'sp']:
                if len(args) >= 1:
                    spin_type = args[0]
                    rpm = float(args[1]) if len(args) > 1 else 3000
                    spin_rad = rpm * 2 * math.pi / 60

                    if spin_type in ['top', 'topspin', 't']:
                        self.ball.spin = np.array([0, spin_rad, 0])
                        self.add_output(f"Topspin: {rpm:.0f} RPM")
                    elif spin_type in ['back', 'backspin', 'b']:
                        self.ball.spin = np.array([0, -spin_rad, 0])
                        self.add_output(f"Backspin: {rpm:.0f} RPM")
                    elif spin_type in ['side', 'sidespin', 's']:
                        self.ball.spin = np.array([0, 0, spin_rad])
                        self.add_output(f"Sidespin: {rpm:.0f} RPM")
                    elif spin_type in ['none', 'no', '0']:
                        self.ball.spin = np.array([0, 0, 0])
                        self.add_output("No spin")
                    else:
                        self.add_output("Types: top, back, side, none")
                else:
                    self.add_output("Usage: spin <type> [rpm]")

            # Serve (quick ball + launch)
            elif command in ['serve', 'sv']:
                power = float(args[0]) if args else 12
                angle = float(args[1]) if len(args) > 1 else 10

                # Place ball at serve position
                self.ball.reset(position=np.array([-1.2, 0.0, 0.9]))

                # Calculate velocity
                angle_rad = math.radians(angle)
                vel = np.array([power * math.cos(angle_rad), 0, power * math.sin(angle_rad) * 0.3 + 1])
                self.ball.velocity = vel
                self.ball.spin = np.array([0, 200, 0])  # Default topspin

                self.ball_active = True
                self.ball_trail = [self.ball.position.copy()]
                self.bounce_markers = []
                self.bounces = 0
                self.max_speed = np.linalg.norm(vel)
                self.flight_time = 0
                self.add_output(f"Served at {np.linalg.norm(vel):.1f} m/s!")

            # Slow motion
            elif command in ['slow', 'slowmo']:
                factor = float(args[0]) if args else 0.2
                self.time_scale = factor
                self.add_output(f"Time scale: {factor}x")

            # Normal speed
            elif command in ['normal', 'fast']:
                self.time_scale = 1.0
                self.add_output("Normal speed")

            # Pause
            elif command in ['pause', 'p']:
                self.paused = not self.paused
                self.add_output("Paused" if self.paused else "Resumed")

            # Reset
            elif command in ['reset', 'r']:
                self.ball.reset(position=np.array([0, 0, 1]))
                self.ball_active = False
                self.ball_trail = []
                self.bounce_markers = []
                self.bounces = 0
                self.max_speed = 0
                self.flight_time = 0
                self.add_output("Reset!")

            # Teleport camera
            elif command in ['tp', 'teleport', 'goto']:
                if len(args) >= 3:
                    self.camera_pos = np.array([float(args[0]), float(args[1]), float(args[2])])
                    self.add_output(f"Teleported to ({args[0]}, {args[1]}, {args[2]})")
                elif args and args[0] == 'ball':
                    self.camera_pos = self.ball.position.copy() + np.array([-1, 0, 0.5])
                    self.add_output("Teleported near ball")
                elif args and args[0] == 'table':
                    self.camera_pos = np.array([-2.5, 1.5, 1.5])
                    self.add_output("Teleported to table view")
                else:
                    self.add_output("Usage: tp <x> <y> <z> or tp ball/table")

            # Status
            elif command in ['status', 'stat', 'info']:
                pos = self.ball.position
                vel = self.ball.velocity
                speed = np.linalg.norm(vel)
                rpm = self.ball.get_spin_rpm()
                self.add_output(f"Pos: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})")
                self.add_output(f"Speed: {speed:.1f} m/s, Spin: {rpm:.0f} RPM")

            # Help
            elif command in ['help', 'h', '?']:
                self.add_output("Commands: ball, launch, spin, serve")
                self.add_output("slow, normal, pause, reset, tp, status")

            # Clear
            elif command in ['clear', 'cls']:
                self.console_output = []

            # Example commands
            elif command == 'topspin':
                self.process_command("ball -1 0 1")
                self.process_command("spin top 4000")
                self.process_command("launch 12 0 3")

            elif command == 'backspin':
                self.process_command("ball 1 0 1.2")
                self.process_command("spin back 3500")
                self.process_command("launch -8 0 4")

            elif command == 'smash':
                self.process_command("ball -1 0 1.5")
                self.process_command("spin top 2000")
                self.process_command("launch 20 0 -2")

            else:
                self.add_output(f"Unknown command: {command}")

        except ValueError as e:
            self.add_output(f"Error: {e}")
        except Exception as e:
            self.add_output(f"Error: {e}")

    def add_output(self, text):
        """Add text to console output"""
        self.console_output.append(text)
        if len(self.console_output) > self.max_output_lines:
            self.console_output.pop(0)

    def handle_events(self):
        """Handle pygame events"""
        for event in pygame.event.get():
            if event.type == QUIT:
                self.running = False

            elif event.type == KEYDOWN:
                if self.console_open:
                    # Console input handling
                    if event.key == K_RETURN:
                        if self.console_input:
                            self.console_history.append(self.console_input)
                            self.process_command(self.console_input)
                            self.console_input = ""
                    elif event.key == K_ESCAPE:
                        self.console_open = False
                        self.console_input = ""
                    elif event.key == K_BACKSPACE:
                        self.console_input = self.console_input[:-1]
                    elif event.key == K_UP and self.console_history:
                        self.console_input = self.console_history[-1]
                    else:
                        if event.unicode and event.unicode.isprintable():
                            self.console_input += event.unicode
                else:
                    # Normal key handling
                    if event.key == K_t:
                        self.console_open = True
                        pygame.mouse.get_rel()  # Clear accumulated motion
                    elif event.key == K_ESCAPE:
                        self.running = False
                    elif event.key == K_F1:
                        self.show_help = not self.show_help
                    elif event.key == K_TAB:
                        self.mouse_locked = not self.mouse_locked
                        pygame.mouse.set_visible(not self.mouse_locked)
                        if self.mouse_locked:
                            pygame.mouse.get_rel()
                    elif event.key == K_p:
                        self.paused = not self.paused
                        self.add_output("Paused" if self.paused else "Resumed")
                    elif event.key == K_r:
                        self.process_command("reset")

            elif event.type == MOUSEBUTTONDOWN:
                if event.button == 4:  # Scroll up
                    self.camera_speed = min(0.3, self.camera_speed + 0.02)
                elif event.button == 5:  # Scroll down
                    self.camera_speed = max(0.02, self.camera_speed - 0.02)

        return self.running

    def render_hud(self):
        """Render 2D HUD elements using pygame"""
        # Create a surface for 2D rendering
        hud_surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)

        # Console output (always visible, bottom left)
        y = self.height - 40 - len(self.console_output) * 22
        for line in self.console_output:
            text = self.font_small.render(line, True, (255, 255, 255))
            hud_surface.blit(text, (15, y))
            y += 22

        # Console input bar
        if self.console_open:
            # Dark background for input
            pygame.draw.rect(hud_surface, (0, 0, 0, 200), (10, self.height - 40, self.width - 20, 35))
            pygame.draw.rect(hud_surface, (100, 100, 100), (10, self.height - 40, self.width - 20, 35), 2)

            input_text = f"> {self.console_input}_"
            text = self.font_medium.render(input_text, True, (0, 255, 0))
            hud_surface.blit(text, (20, self.height - 35))
        else:
            # Hint to open console
            hint = self.font_small.render("Press T to type commands", True, (200, 200, 200))
            hud_surface.blit(hint, (15, self.height - 30))

        # Ball status (top right)
        if self.ball_active:
            speed = self.ball.get_speed()
            rpm = self.ball.get_spin_rpm()
            pos = self.ball.position

            status_lines = [
                f"Speed: {speed:.1f} m/s",
                f"Spin: {rpm:.0f} RPM",
                f"Height: {pos[2]:.2f} m",
                f"Bounces: {self.bounces}",
                f"Time: {self.flight_time:.2f}s"
            ]

            y = 15
            for line in status_lines:
                text = self.font_medium.render(line, True, (255, 255, 0))
                hud_surface.blit(text, (self.width - 180, y))
                y += 26

        # Help text (top left)
        if self.show_help:
            help_lines = [
                "WASD - Move    Space/Shift - Up/Down",
                "Right Mouse - Look around",
                "T - Open command console",
                "P - Pause    R - Reset    F1 - Hide help",
                "",
                "Quick commands: serve, topspin, backspin, smash"
            ]
            y = 15
            for line in help_lines:
                text = self.font_small.render(line, True, (200, 200, 200))
                hud_surface.blit(text, (15, y))
                y += 20

        # Paused indicator
        if self.paused:
            text = self.font_large.render("PAUSED", True, (255, 100, 100))
            text_rect = text.get_rect(center=(self.width // 2, 50))
            hud_surface.blit(text, text_rect)

        # Slow motion indicator
        if self.time_scale < 1.0:
            text = self.font_medium.render(f"SLOW-MO {self.time_scale}x", True, (100, 200, 255))
            hud_surface.blit(text, (self.width // 2 - 60, 80))

        # Crosshair
        cx, cy = self.width // 2, self.height // 2
        pygame.draw.line(hud_surface, (255, 255, 255, 100), (cx - 10, cy), (cx + 10, cy), 1)
        pygame.draw.line(hud_surface, (255, 255, 255, 100), (cx, cy - 10), (cx, cy + 10), 1)

        return hud_surface

    def render(self):
        """Main render function"""
        # Clear and setup 3D
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        self._update_camera()

        # Draw 3D world
        self._draw_sky()
        self._draw_ground()
        self._draw_table()
        self._draw_racket()
        self._draw_ball()

        # Get OpenGL buffer
        pygame.display.flip()

        # Draw 2D HUD on top
        # Switch to 2D mode
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, self.width, self.height, 0, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()

        glDisable(GL_DEPTH_TEST)
        glDisable(GL_LIGHTING)

        # Render HUD
        hud = self.render_hud()
        hud_data = pygame.image.tostring(hud, 'RGBA', True)

        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        glRasterPos2i(0, self.height)
        glDrawPixels(self.width, self.height, GL_RGBA, GL_UNSIGNED_BYTE, hud_data)

        # Restore 3D mode
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)

        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        glPopMatrix()

        pygame.display.flip()

    def run(self):
        """Main game loop"""
        print("\n" + "=" * 60)
        print("  Table Tennis Physics - Interactive Game")
        print("=" * 60)
        print("\n  Controls:")
        print("    WASD        - Move around")
        print("    Space/Shift - Up/Down")
        print("    Right Mouse - Look around")
        print("    T           - Open command console")
        print("    P           - Pause")
        print("    R           - Reset")
        print("    ESC         - Quit")
        print("\n  Try typing: serve")
        print("              topspin")
        print("              ball 0 0 1  then  launch 10 0 3")
        print("=" * 60 + "\n")

        while self.running:
            self.handle_events()
            self.handle_movement()

            # Multiple physics steps per frame
            for _ in range(3):
                self.update_physics()

            self.render()
            self.clock.tick(self.fps)

        pygame.quit()
        print("\nGame ended. Thanks for playing!")


def main():
    print("Starting Table Tennis Game...")
    game = GameWorld()
    game.run()


if __name__ == "__main__":
    main()
