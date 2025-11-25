"""
Interactive Table Tennis Physics Simulation

Real-time 3D visualization with keyboard/mouse control.
Control the racket and see ball physics in action.

Controls:
    WASD        - Move racket (horizontal)
    Q/E         - Move racket up/down
    Arrow Keys  - Tilt racket angle
    SPACE       - Serve ball
    R           - Reset simulation
    1/2/3       - Change spin type (topspin/backspin/sidespin)
    +/-         - Adjust swing power
    ESC         - Quit

Mouse:
    Left Click + Drag  - Swing racket
    Right Click        - Serve ball
    Scroll             - Zoom in/out
"""

import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
import sys
import math

sys.path.insert(0, '.')

from src.physics.parameters import PhysicsParameters, create_offensive_setup
from src.physics.ball import Ball
from src.physics.table import Table
from src.physics.racket import Racket
from src.physics.collision import CollisionHandler


class InteractiveSimulation:
    def __init__(self, width=1200, height=800):
        self.width = width
        self.height = height

        # Initialize pygame
        pygame.init()
        pygame.display.set_mode((width, height), DOUBLEBUF | OPENGL)
        pygame.display.set_caption("Table Tennis Physics Simulation - Interactive")

        # Physics setup
        self.params = create_offensive_setup()
        self.params.dt = 0.002  # 2ms timestep

        self.ball = Ball(self.params)
        self.table = Table(self.params)
        self.racket = Racket(self.params, side=-1)  # Player side
        self.collision_handler = CollisionHandler(self.params)

        # Camera settings
        self.camera_distance = 5.0
        self.camera_angle_h = 45.0  # Horizontal angle
        self.camera_angle_v = 30.0  # Vertical angle

        # Racket control
        self.racket_pos = np.array([-1.2, 0.0, 0.9])
        self.racket_angle = np.array([0.0, 0.0])  # pitch, yaw
        self.swing_power = 15.0
        self.spin_type = "topspin"
        self.spin_amount = 3000  # RPM

        # Simulation state
        self.ball_active = False
        self.paused = False
        self.trail = []
        self.max_trail = 200

        # Mouse state
        self.mouse_pressed = False
        self.last_mouse_pos = None
        self.swing_start = None

        # Stats
        self.hits = 0
        self.bounces = 0

        # Initialize OpenGL
        self._init_gl()

        # Clock for FPS control
        self.clock = pygame.time.Clock()
        self.fps = 60

    def _init_gl(self):
        """Initialize OpenGL settings"""
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_COLOR_MATERIAL)
        glEnable(GL_NORMALIZE)

        # Light position
        glLightfv(GL_LIGHT0, GL_POSITION, [5.0, 5.0, 10.0, 1.0])
        glLightfv(GL_LIGHT0, GL_AMBIENT, [0.3, 0.3, 0.3, 1.0])
        glLightfv(GL_LIGHT0, GL_DIFFUSE, [0.8, 0.8, 0.8, 1.0])

        # Background color
        glClearColor(0.1, 0.1, 0.15, 1.0)

        # Perspective
        glMatrixMode(GL_PROJECTION)
        gluPerspective(45, self.width / self.height, 0.1, 100.0)
        glMatrixMode(GL_MODELVIEW)

    def _update_camera(self):
        """Update camera position based on angles"""
        glLoadIdentity()

        # Calculate camera position
        rad_h = math.radians(self.camera_angle_h)
        rad_v = math.radians(self.camera_angle_v)

        cam_x = self.camera_distance * math.cos(rad_v) * math.sin(rad_h)
        cam_y = self.camera_distance * math.cos(rad_v) * math.cos(rad_h)
        cam_z = self.camera_distance * math.sin(rad_v) + 1.0

        # Look at table center
        gluLookAt(cam_x, cam_y, cam_z, 0, 0, 0.76, 0, 0, 1)

    def _draw_table(self):
        """Draw the table tennis table"""
        # Table surface (dark blue/green)
        glColor3f(0.0, 0.3, 0.5)

        half_length = self.params.table_length / 2
        half_width = self.params.table_width / 2
        height = self.params.table_height

        glBegin(GL_QUADS)
        glNormal3f(0, 0, 1)
        glVertex3f(-half_length, -half_width, height)
        glVertex3f(half_length, -half_width, height)
        glVertex3f(half_length, half_width, height)
        glVertex3f(-half_length, half_width, height)
        glEnd()

        # Table border (white lines)
        glColor3f(1.0, 1.0, 1.0)
        glLineWidth(3.0)
        glBegin(GL_LINE_LOOP)
        glVertex3f(-half_length, -half_width, height + 0.001)
        glVertex3f(half_length, -half_width, height + 0.001)
        glVertex3f(half_length, half_width, height + 0.001)
        glVertex3f(-half_length, half_width, height + 0.001)
        glEnd()

        # Center line
        glBegin(GL_LINES)
        glVertex3f(-half_length, 0, height + 0.001)
        glVertex3f(half_length, 0, height + 0.001)
        glEnd()

        # Net
        glColor3f(0.8, 0.8, 0.8)
        net_height = self.params.table.net_height
        glBegin(GL_QUADS)
        glVertex3f(0, -half_width - 0.15, height)
        glVertex3f(0, half_width + 0.15, height)
        glVertex3f(0, half_width + 0.15, height + net_height)
        glVertex3f(0, -half_width - 0.15, height + net_height)
        glEnd()

        # Net posts
        glColor3f(0.3, 0.3, 0.3)
        self._draw_cylinder(0, -half_width - 0.15, height, 0.01, net_height)
        self._draw_cylinder(0, half_width + 0.15, height, 0.01, net_height)

        # Table legs
        glColor3f(0.3, 0.3, 0.3)
        leg_positions = [
            (-half_length + 0.1, -half_width + 0.1),
            (-half_length + 0.1, half_width - 0.1),
            (half_length - 0.1, -half_width + 0.1),
            (half_length - 0.1, half_width - 0.1),
        ]
        for lx, ly in leg_positions:
            self._draw_cylinder(lx, ly, 0, 0.02, height)

    def _draw_cylinder(self, x, y, z, radius, height, segments=12):
        """Draw a simple cylinder"""
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
        """Draw the ball with trail"""
        if not self.ball_active:
            return

        # Draw trail
        if len(self.trail) > 1:
            glDisable(GL_LIGHTING)
            glBegin(GL_LINE_STRIP)
            for i, pos in enumerate(self.trail):
                alpha = i / len(self.trail)
                glColor4f(1.0, 0.6, 0.0, alpha * 0.5)
                glVertex3f(*pos)
            glEnd()
            glEnable(GL_LIGHTING)

        # Draw ball
        glColor3f(1.0, 0.6, 0.0)  # Orange
        glPushMatrix()
        glTranslatef(*self.ball.position)

        quadric = gluNewQuadric()
        gluSphere(quadric, self.params.ball_radius, 16, 16)
        gluDeleteQuadric(quadric)

        glPopMatrix()

        # Draw spin indicator
        if self.ball.get_spin_rate() > 10:
            self._draw_spin_indicator()

    def _draw_spin_indicator(self):
        """Draw an arrow showing spin direction"""
        spin = self.ball.spin
        spin_rate = np.linalg.norm(spin)
        if spin_rate < 10:
            return

        glDisable(GL_LIGHTING)
        glColor3f(0.0, 1.0, 1.0)

        pos = self.ball.position
        spin_dir = spin / spin_rate * 0.05

        glBegin(GL_LINES)
        glVertex3f(*pos)
        glVertex3f(pos[0] + spin_dir[0], pos[1] + spin_dir[1], pos[2] + spin_dir[2])
        glEnd()

        glEnable(GL_LIGHTING)

    def _draw_racket(self):
        """Draw the racket"""
        glPushMatrix()
        glTranslatef(*self.racket_pos)

        # Apply rotation
        glRotatef(self.racket_angle[1], 0, 0, 1)  # Yaw
        glRotatef(self.racket_angle[0], 0, 1, 0)  # Pitch

        # Handle
        glColor3f(0.6, 0.4, 0.2)  # Wood color
        glPushMatrix()
        glRotatef(90, 1, 0, 0)
        quadric = gluNewQuadric()
        gluCylinder(quadric, 0.015, 0.015, 0.12, 8, 1)
        gluDeleteQuadric(quadric)
        glPopMatrix()

        # Blade (elliptical)
        glColor3f(0.8, 0.1, 0.1)  # Red rubber

        # Draw as a flattened sphere
        glPushMatrix()
        glScalef(0.08, 0.075, 0.005)
        quadric = gluNewQuadric()
        gluSphere(quadric, 1.0, 16, 8)
        gluDeleteQuadric(quadric)
        glPopMatrix()

        glPopMatrix()

    def _draw_floor(self):
        """Draw floor grid"""
        glDisable(GL_LIGHTING)
        glColor3f(0.2, 0.2, 0.2)

        glBegin(GL_LINES)
        for i in range(-5, 6):
            glVertex3f(i, -5, 0)
            glVertex3f(i, 5, 0)
            glVertex3f(-5, i, 0)
            glVertex3f(5, i, 0)
        glEnd()

        glEnable(GL_LIGHTING)

    def _draw_hud(self):
        """Draw heads-up display with info"""
        # Switch to 2D rendering
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, self.width, self.height, 0, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()

        glDisable(GL_DEPTH_TEST)
        glDisable(GL_LIGHTING)

        # Create text surface with pygame
        font = pygame.font.Font(None, 24)

        texts = [
            f"Power: {self.swing_power:.0f} m/s  (+/-)",
            f"Spin: {self.spin_type} {self.spin_amount} RPM  (1/2/3)",
            f"Ball Speed: {self.ball.get_speed():.1f} m/s" if self.ball_active else "Press SPACE to serve",
            f"Ball Spin: {self.ball.get_spin_rpm():.0f} RPM" if self.ball_active else "",
            f"Bounces: {self.bounces}  Hits: {self.hits}",
            "",
            "WASD: Move  Q/E: Up/Down  Arrows: Angle",
            "SPACE: Serve  R: Reset  ESC: Quit",
        ]

        # Restore 3D rendering
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)

        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        glPopMatrix()

        # Draw text using pygame (blit after OpenGL)
        self._hud_texts = texts

    def _render_text(self):
        """Render HUD text using pygame"""
        font = pygame.font.Font(None, 28)

        texts = [
            f"Power: {self.swing_power:.0f} m/s  [+/-]",
            f"Spin: {self.spin_type} {self.spin_amount} RPM  [1/2/3]",
            f"Ball Speed: {self.ball.get_speed():.1f} m/s" if self.ball_active else "Press SPACE to serve",
            f"Ball Spin: {self.ball.get_spin_rpm():.0f} RPM" if self.ball_active else "",
            f"Bounces: {self.bounces}  |  Hits: {self.hits}",
            "",
            "Controls: WASD=Move  Q/E=Up/Down  Arrows=Angle  R=Reset",
        ]

        # Create a surface for text
        y_offset = 10
        for text in texts:
            if text:
                surface = font.render(text, True, (255, 255, 255))
                # Convert to OpenGL texture and render
                # For simplicity, we'll use pygame's overlay

    def serve_ball(self):
        """Serve the ball from racket position"""
        # Calculate serve direction based on racket angle
        pitch = math.radians(self.racket_angle[0])
        yaw = math.radians(self.racket_angle[1])

        direction = np.array([
            math.cos(pitch) * math.cos(yaw),
            math.cos(pitch) * math.sin(yaw),
            math.sin(pitch) * 0.3 + 0.2  # Slight upward
        ])
        direction = direction / np.linalg.norm(direction)

        velocity = direction * self.swing_power

        # Calculate spin based on type
        spin_rad = self.spin_amount * 2 * math.pi / 60
        if self.spin_type == "topspin":
            spin = np.array([0.0, spin_rad, 0.0])
        elif self.spin_type == "backspin":
            spin = np.array([0.0, -spin_rad, 0.0])
        else:  # sidespin
            spin = np.array([0.0, 0.0, spin_rad])

        # Set ball state
        self.ball.reset(
            position=self.racket_pos + direction * 0.1,
            velocity=velocity,
            spin=spin
        )
        self.ball_active = True
        self.trail = []
        self.bounces = 0

    def swing_racket(self, direction):
        """Swing racket in given direction"""
        if not self.ball_active:
            return

        # Check if ball is near racket
        dist = np.linalg.norm(self.ball.position - self.racket_pos)
        if dist < 0.2:
            # Hit the ball
            swing_dir = np.array(direction)
            swing_dir = swing_dir / np.linalg.norm(swing_dir)

            new_velocity = swing_dir * self.swing_power

            # Add spin
            spin_rad = self.spin_amount * 2 * math.pi / 60
            if self.spin_type == "topspin":
                spin = np.array([0.0, spin_rad, 0.0])
            elif self.spin_type == "backspin":
                spin = np.array([0.0, -spin_rad, 0.0])
            else:
                spin = np.array([0.0, 0.0, spin_rad])

            self.ball.velocity = new_velocity
            self.ball.spin = spin
            self.hits += 1

    def update_physics(self):
        """Update physics simulation"""
        if not self.ball_active or self.paused:
            return

        # Update ball
        self.ball.update(self.params.dt)

        # Add to trail
        self.trail.append(self.ball.position.copy())
        if len(self.trail) > self.max_trail:
            self.trail.pop(0)

        # Update racket object position for collision
        racket_normal = np.array([
            math.cos(math.radians(self.racket_angle[0])),
            math.sin(math.radians(self.racket_angle[1])),
            0.0
        ])
        racket_normal = racket_normal / np.linalg.norm(racket_normal) if np.linalg.norm(racket_normal) > 0.01 else np.array([1, 0, 0])
        self.racket.position = self.racket_pos.copy()
        self.racket.orientation = racket_normal

        # Check collisions
        if self.collision_handler.handle_ball_table_collision(self.ball, self.table):
            self.bounces += 1

        self.collision_handler.handle_ball_net_collision(self.ball, self.table)

        if self.collision_handler.handle_ball_racket_collision(self.ball, self.racket):
            self.hits += 1

        # Check out of bounds
        if self.ball.position[2] < -0.5 or np.linalg.norm(self.ball.position[:2]) > 5:
            self.ball_active = False

    def handle_input(self):
        """Handle keyboard and mouse input"""
        keys = pygame.key.get_pressed()

        # Racket movement
        move_speed = 0.05
        if keys[K_w]:
            self.racket_pos[0] += move_speed
        if keys[K_s]:
            self.racket_pos[0] -= move_speed
        if keys[K_a]:
            self.racket_pos[1] -= move_speed
        if keys[K_d]:
            self.racket_pos[1] += move_speed
        if keys[K_q]:
            self.racket_pos[2] -= move_speed
        if keys[K_e]:
            self.racket_pos[2] += move_speed

        # Racket angle
        angle_speed = 2.0
        if keys[K_UP]:
            self.racket_angle[0] += angle_speed
        if keys[K_DOWN]:
            self.racket_angle[0] -= angle_speed
        if keys[K_LEFT]:
            self.racket_angle[1] -= angle_speed
        if keys[K_RIGHT]:
            self.racket_angle[1] += angle_speed

        # Clamp angles
        self.racket_angle[0] = max(-60, min(60, self.racket_angle[0]))
        self.racket_angle[1] = max(-90, min(90, self.racket_angle[1]))

        # Clamp position
        self.racket_pos[0] = max(-2.5, min(2.5, self.racket_pos[0]))
        self.racket_pos[1] = max(-1.5, min(1.5, self.racket_pos[1]))
        self.racket_pos[2] = max(0.3, min(2.0, self.racket_pos[2]))

    def handle_events(self):
        """Handle pygame events"""
        for event in pygame.event.get():
            if event.type == QUIT:
                return False

            elif event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    return False
                elif event.key == K_SPACE:
                    self.serve_ball()
                elif event.key == K_r:
                    self.ball_active = False
                    self.trail = []
                    self.bounces = 0
                    self.hits = 0
                elif event.key == K_1:
                    self.spin_type = "topspin"
                elif event.key == K_2:
                    self.spin_type = "backspin"
                elif event.key == K_3:
                    self.spin_type = "sidespin"
                elif event.key == K_PLUS or event.key == K_EQUALS:
                    self.swing_power = min(30, self.swing_power + 1)
                elif event.key == K_MINUS:
                    self.swing_power = max(5, self.swing_power - 1)
                elif event.key == K_p:
                    self.paused = not self.paused

            elif event.type == MOUSEBUTTONDOWN:
                if event.button == 1:  # Left click
                    self.mouse_pressed = True
                    self.swing_start = pygame.mouse.get_pos()
                elif event.button == 3:  # Right click
                    self.serve_ball()
                elif event.button == 4:  # Scroll up
                    self.camera_distance = max(2, self.camera_distance - 0.5)
                elif event.button == 5:  # Scroll down
                    self.camera_distance = min(15, self.camera_distance + 0.5)

            elif event.type == MOUSEBUTTONUP:
                if event.button == 1 and self.mouse_pressed:
                    self.mouse_pressed = False
                    if self.swing_start:
                        end_pos = pygame.mouse.get_pos()
                        dx = end_pos[0] - self.swing_start[0]
                        dy = end_pos[1] - self.swing_start[1]
                        if abs(dx) > 20 or abs(dy) > 20:
                            self.swing_racket([dx * 0.01, 0, -dy * 0.01])
                    self.swing_start = None

            elif event.type == MOUSEMOTION:
                if pygame.mouse.get_pressed()[1]:  # Middle button for camera
                    dx, dy = event.rel
                    self.camera_angle_h += dx * 0.5
                    self.camera_angle_v = max(-10, min(80, self.camera_angle_v + dy * 0.3))

        return True

    def render(self):
        """Render the scene"""
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        self._update_camera()
        self._draw_floor()
        self._draw_table()
        self._draw_ball()
        self._draw_racket()

        pygame.display.flip()

        # Render HUD text as 2D overlay
        self._render_pygame_hud()

    def _render_pygame_hud(self):
        """Render HUD using pygame 2D"""
        # This would require switching between OpenGL and pygame 2D
        # For now, print to console occasionally
        pass

    def run(self):
        """Main game loop"""
        print("\n" + "="*60)
        print("  Table Tennis Physics Simulation - Interactive Mode")
        print("="*60)
        print("\nControls:")
        print("  WASD      - Move racket horizontally")
        print("  Q/E       - Move racket up/down")
        print("  Arrows    - Tilt racket angle")
        print("  SPACE     - Serve ball")
        print("  R         - Reset")
        print("  1/2/3     - Topspin/Backspin/Sidespin")
        print("  +/-       - Adjust power")
        print("  Mouse     - Middle drag to rotate camera")
        print("  Scroll    - Zoom in/out")
        print("  ESC       - Quit")
        print("\n" + "="*60 + "\n")

        running = True
        frame_count = 0

        while running:
            running = self.handle_events()
            self.handle_input()

            # Run multiple physics steps per frame for stability
            for _ in range(3):
                self.update_physics()

            self.render()

            # Print status occasionally
            frame_count += 1
            if frame_count % 120 == 0 and self.ball_active:
                print(f"Ball: pos=({self.ball.position[0]:.2f}, {self.ball.position[1]:.2f}, {self.ball.position[2]:.2f})  "
                      f"speed={self.ball.get_speed():.1f}m/s  spin={self.ball.get_spin_rpm():.0f}RPM  "
                      f"bounces={self.bounces}")

            self.clock.tick(self.fps)

        pygame.quit()


def main():
    print("Starting Interactive Table Tennis Simulation...")
    print("Loading...")

    sim = InteractiveSimulation()
    sim.run()

    print("\nSimulation ended.")


if __name__ == "__main__":
    main()
