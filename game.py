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

        # Create window with OpenGL - use HWSURFACE for less flicker
        self.screen = pygame.display.set_mode((width, height), DOUBLEBUF | OPENGL | HWSURFACE)
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
        self.racket = Racket(self.params, side=-1)
        self.collision = CollisionHandler(self.params)

        # Ball state
        self.ball_active = False
        self.ball_trail = []
        self.max_trail = 150
        self.bounce_markers = []

        # Camera - player viewpoint
        self.camera_pos = np.array([-3.0, 2.0, 1.5])
        self.camera_yaw = -30.0
        self.camera_pitch = 15.0
        self.camera_speed = 0.08

        # Mouse sensitivity
        self.mouse_sensitivity = 0.15

        # Console/Chat state
        self.console_open = False
        self.console_input = ""
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
        self.time_scale = 1.0

        # Stats
        self.bounces = 0
        self.max_speed = 0
        self.flight_time = 0

        # Mouse button state for up/down
        self.mouse_side1_held = False  # Button 4 - down
        self.mouse_side2_held = False  # Button 5 - up

        # Clock
        self.clock = pygame.time.Clock()
        self.fps = 60

        # Initialize OpenGL
        self._init_gl()

        # Welcome message
        self.add_output("Welcome! Press / to type commands")
        self.add_output("Try: serve  or  topspin")

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
        """First-person camera"""
        glLoadIdentity()

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
        """Draw the floor"""
        glDisable(GL_LIGHTING)

        # Main floor
        glColor4f(0.3, 0.4, 0.3, 1.0)
        glBegin(GL_QUADS)
        glVertex3f(-10, -10, 0)
        glVertex3f(10, -10, 0)
        glVertex3f(10, 10, 0)
        glVertex3f(-10, 10, 0)
        glEnd()

        # Grid
        glColor4f(0.35, 0.45, 0.35, 0.7)
        glLineWidth(1.0)
        glBegin(GL_LINES)
        for i in range(-10, 11):
            glVertex3f(i, -10, 0.002)
            glVertex3f(i, 10, 0.002)
            glVertex3f(-10, i, 0.002)
            glVertex3f(10, i, 0.002)
        glEnd()

        glEnable(GL_LIGHTING)

    def _draw_table(self):
        """Draw table tennis table"""
        hl = self.params.table_length / 2
        hw = self.params.table_width / 2
        h = self.params.table_height
        th = 0.03

        # Table top
        glColor3f(0.0, 0.25, 0.5)
        glBegin(GL_QUADS)
        glNormal3f(0, 0, 1)
        glVertex3f(-hl, -hw, h)
        glVertex3f(hl, -hw, h)
        glVertex3f(hl, hw, h)
        glVertex3f(-hl, hw, h)
        glEnd()

        # Table sides
        glColor3f(0.0, 0.2, 0.4)
        glBegin(GL_QUADS)
        # Front
        glNormal3f(0, -1, 0)
        glVertex3f(-hl, -hw, h - th)
        glVertex3f(hl, -hw, h - th)
        glVertex3f(hl, -hw, h)
        glVertex3f(-hl, -hw, h)
        # Back
        glNormal3f(0, 1, 0)
        glVertex3f(-hl, hw, h - th)
        glVertex3f(hl, hw, h - th)
        glVertex3f(hl, hw, h)
        glVertex3f(-hl, hw, h)
        # Left
        glNormal3f(-1, 0, 0)
        glVertex3f(-hl, -hw, h - th)
        glVertex3f(-hl, hw, h - th)
        glVertex3f(-hl, hw, h)
        glVertex3f(-hl, -hw, h)
        # Right
        glNormal3f(1, 0, 0)
        glVertex3f(hl, -hw, h - th)
        glVertex3f(hl, hw, h - th)
        glVertex3f(hl, hw, h)
        glVertex3f(hl, -hw, h)
        glEnd()

        # White lines
        glDisable(GL_LIGHTING)
        glColor3f(1.0, 1.0, 1.0)
        glLineWidth(3.0)
        glBegin(GL_LINE_LOOP)
        glVertex3f(-hl + 0.02, -hw + 0.02, h + 0.002)
        glVertex3f(hl - 0.02, -hw + 0.02, h + 0.002)
        glVertex3f(hl - 0.02, hw - 0.02, h + 0.002)
        glVertex3f(-hl + 0.02, hw - 0.02, h + 0.002)
        glEnd()
        glBegin(GL_LINES)
        glVertex3f(0, -hw, h + 0.002)
        glVertex3f(0, hw, h + 0.002)
        glVertex3f(-hl, 0, h + 0.002)
        glVertex3f(hl, 0, h + 0.002)
        glEnd()
        glEnable(GL_LIGHTING)

        # Net
        nh = self.params.table.net_height
        glColor4f(0.95, 0.95, 0.95, 0.9)
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
        glColor3f(0.25, 0.25, 0.25)
        for lx, ly in [(-hl + 0.15, -hw + 0.1), (-hl + 0.15, hw - 0.1),
                       (hl - 0.15, -hw + 0.1), (hl - 0.15, hw - 0.1)]:
            self._draw_cylinder(lx, ly, 0, 0.025, h - th)

    def _draw_cylinder(self, x, y, z, radius, height, segments=12):
        """Draw cylinder"""
        glBegin(GL_QUAD_STRIP)
        for i in range(segments + 1):
            angle = 2 * math.pi * i / segments
            nx, ny = math.cos(angle), math.sin(angle)
            glNormal3f(nx, ny, 0)
            glVertex3f(x + radius * nx, y + radius * ny, z)
            glVertex3f(x + radius * nx, y + radius * ny, z + height)
        glEnd()

    def _draw_ball(self):
        """Draw ball with trail"""
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

        # Bounce markers
        glDisable(GL_LIGHTING)
        for marker in self.bounce_markers[-5:]:
            glColor4f(0.0, 1.0, 0.0, 0.5)
            glPushMatrix()
            glTranslatef(marker[0], marker[1], marker[2] + 0.003)
            glBegin(GL_LINE_LOOP)
            for i in range(16):
                angle = 2 * math.pi * i / 16
                glVertex3f(0.05 * math.cos(angle), 0.05 * math.sin(angle), 0)
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

    def _draw_racket(self):
        """Draw racket"""
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

        # Blade
        glColor3f(0.8, 0.1, 0.1)
        glPushMatrix()
        glScalef(0.085, 0.08, 0.008)
        quadric = gluNewQuadric()
        gluSphere(quadric, 1.0, 16, 8)
        gluDeleteQuadric(quadric)
        glPopMatrix()

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

        if self.ball.position[2] < 0:
            self.add_output(f"Landed! Time: {self.flight_time:.2f}s")
            self.ball_active = False
        elif np.linalg.norm(self.ball.position[:2]) > 8:
            self.add_output("Out of bounds!")
            self.ball_active = False

    def handle_movement(self):
        """Handle player movement"""
        if self.console_open or self.menu_open:
            return

        keys = pygame.key.get_pressed()

        yaw_rad = math.radians(self.camera_yaw)
        forward = np.array([math.cos(yaw_rad), math.sin(yaw_rad), 0])
        right = np.array([math.sin(yaw_rad), -math.cos(yaw_rad), 0])

        # WASD movement
        if keys[K_w]:
            self.camera_pos += forward * self.camera_speed
        if keys[K_s]:
            self.camera_pos -= forward * self.camera_speed
        if keys[K_a]:
            self.camera_pos -= right * self.camera_speed
        if keys[K_d]:
            self.camera_pos += right * self.camera_speed

        # Mouse side buttons for up/down
        if self.mouse_side2_held:  # Side button 2 = up
            self.camera_pos[2] += self.camera_speed
        if self.mouse_side1_held:  # Side button 1 = down
            self.camera_pos[2] -= self.camera_speed

        # Keep above ground
        self.camera_pos[2] = max(0.3, self.camera_pos[2])

        # Mouse look (always active when not in menu/console)
        rel = pygame.mouse.get_rel()
        self.camera_yaw -= rel[0] * self.mouse_sensitivity  # Fixed: inverted
        self.camera_pitch -= rel[1] * self.mouse_sensitivity
        self.camera_pitch = max(-80, min(80, self.camera_pitch))

    def process_command(self, cmd):
        """Process command"""
        cmd = cmd.strip().lower()
        parts = cmd.split()
        if not parts:
            return

        command = parts[0]
        args = parts[1:]

        try:
            if command in ['ball', 'b']:
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
                    if spin_type in ['top', 'topspin', 't']:
                        self.ball.spin = np.array([0, spin_rad, 0])
                        self.add_output(f"Topspin {rpm:.0f} RPM")
                    elif spin_type in ['back', 'backspin', 'b']:
                        self.ball.spin = np.array([0, -spin_rad, 0])
                        self.add_output(f"Backspin {rpm:.0f} RPM")
                    elif spin_type in ['side', 'sidespin', 's']:
                        self.ball.spin = np.array([0, 0, spin_rad])
                        self.add_output(f"Sidespin {rpm:.0f} RPM")
                    elif spin_type in ['none', 'no', '0']:
                        self.ball.spin = np.array([0, 0, 0])
                        self.add_output("No spin")
                else:
                    self.add_output("Usage: spin <top/back/side> [rpm]")

            elif command in ['serve', 'sv']:
                power = float(args[0]) if args else 12
                self.ball.reset(position=np.array([-1.2, 0.0, 0.9]))
                angle_rad = math.radians(10)
                vel = np.array([power * math.cos(angle_rad), 0, power * 0.15 + 1])
                self.ball.velocity = vel
                self.ball.spin = np.array([0, 200, 0])
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
                self.ball.reset(position=np.array([0, 0, 1]))
                self.ball_active = False
                self.ball_trail = []
                self.bounce_markers = []
                self.bounces = 0
                self.flight_time = 0
                self.add_output("Reset!")

            elif command in ['tp', 'teleport']:
                if len(args) >= 3:
                    self.camera_pos = np.array([float(args[0]), float(args[1]), float(args[2])])
                    self.add_output(f"TP to ({args[0]}, {args[1]}, {args[2]})")
                elif args and args[0] == 'ball':
                    self.camera_pos = self.ball.position.copy() + np.array([-1, 0, 0.5])
                    self.add_output("TP to ball")
                elif args and args[0] == 'table':
                    self.camera_pos = np.array([-2.5, 1.5, 1.5])
                    self.add_output("TP to table view")

            elif command in ['help', 'h', '?']:
                self.add_output("ball, launch, spin, serve, slow, tp, reset")

            elif command in ['clear', 'cls']:
                self.console_output = []

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
                self.add_output(f"Unknown: {command}")

        except Exception as e:
            self.add_output(f"Error: {e}")

    def add_output(self, text):
        """Add console output"""
        self.console_output.append(text)
        if len(self.console_output) > self.max_output_lines:
            self.console_output.pop(0)

    def handle_events(self):
        """Handle events"""
        for event in pygame.event.get():
            if event.type == QUIT:
                self.running = False

            elif event.type == KEYDOWN:
                if self.console_open:
                    # Chat input mode
                    if event.key == K_RETURN:
                        if self.console_input:
                            self.console_history.append(self.console_input)
                            self.process_command(self.console_input)
                            self.console_input = ""
                        self.console_open = False
                    elif event.key == K_ESCAPE or event.key == K_SLASH:
                        self.console_open = False
                        self.console_input = ""
                    elif event.key == K_BACKSPACE:
                        self.console_input = self.console_input[:-1]
                    elif event.key == K_UP and self.console_history:
                        if self.history_index < len(self.console_history) - 1:
                            self.history_index += 1
                            self.console_input = self.console_history[-(self.history_index + 1)]
                    elif event.key == K_DOWN:
                        if self.history_index > 0:
                            self.history_index -= 1
                            self.console_input = self.console_history[-(self.history_index + 1)]
                        else:
                            self.history_index = -1
                            self.console_input = ""
                    else:
                        if event.unicode and event.unicode.isprintable() and event.unicode != '/':
                            self.console_input += event.unicode

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
                    elif event.key == K_ESCAPE:
                        self.menu_open = True
                        pygame.mouse.set_visible(True)
                        pygame.event.set_grab(False)
                    elif event.key == K_F1:
                        self.show_help = not self.show_help
                    elif event.key == K_F3:
                        self.show_debug = not self.show_debug

            elif event.type == MOUSEBUTTONDOWN:
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

                    # Mouse side buttons - try multiple common mappings
                    # X1 (back) = 4 or 6 or 8, X2 (forward) = 5 or 7 or 9
                    if event.button in [4, 6, 8]:
                        self.mouse_side1_held = True
                    elif event.button in [5, 7, 9]:
                        self.mouse_side2_held = True

            elif event.type == MOUSEBUTTONUP:
                if event.button in [4, 6, 8]:
                    self.mouse_side1_held = False
                elif event.button in [5, 7, 9]:
                    self.mouse_side2_held = False

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

        # Console input
        if self.console_open:
            pygame.draw.rect(hud, (0, 0, 0, 220), (10, self.height - 45, self.width - 20, 40))
            pygame.draw.rect(hud, (80, 80, 80), (10, self.height - 45, self.width - 20, 40), 2)
            input_text = f"/{self.console_input}_"
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
                f"Height: {pos[2]:.2f} m",
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

            # Direction facing
            if -45 <= yaw < 45:
                facing = "East (+X)"
            elif 45 <= yaw < 135:
                facing = "North (+Y)"
            elif yaw >= 135 or yaw < -135:
                facing = "West (-X)"
            else:
                facing = "South (-Y)"

            # FPS
            fps = self.clock.get_fps()

            debug_lines = [
                f"Table Tennis Physics Simulation",
                f"",
                f"XYZ: {self.camera_pos[0]:.3f} / {self.camera_pos[1]:.3f} / {self.camera_pos[2]:.3f}",
                f"Facing: {facing}",
                f"Rotation: {yaw:.1f} / {pitch:.1f}",
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
                "F1 - Help    F3 - Debug"
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

        return hud

    def render(self):
        """Main render"""
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        self._update_camera()
        self._draw_ground()
        self._draw_table()
        self._draw_racket()
        self._draw_ball()

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
