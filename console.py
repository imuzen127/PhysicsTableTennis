"""
Table Tennis Physics Simulation - Console Interface

Interactive command-line interface for controlling the simulation.
Type 'help' to see available commands.

Example commands:
    set ball 0 0 1           # Place ball at position (0, 0, 1)
    set velocity 10 0 2      # Set ball velocity
    set spin topspin 3000    # Set topspin at 3000 RPM
    set racket -1 0 0.9      # Place racket at position
    set angle 15 0           # Set racket pitch and yaw
    set power 15             # Set swing power
    hit                      # Hit the ball with current settings
    serve                    # Serve from current racket position
    run 2                    # Run simulation for 2 seconds
    show                     # Show 3D visualization
    status                   # Show current state
    reset                    # Reset simulation
"""

import sys
import cmd
import numpy as np
import threading
import time

sys.path.insert(0, '.')

from src.physics.parameters import (
    PhysicsParameters,
    create_offensive_setup,
    create_defensive_setup,
    create_allround_setup
)
from src.physics.ball import Ball
from src.physics.table import Table
from src.physics.racket import Racket
from src.physics.collision import CollisionHandler


class SimulationState:
    """Holds the current simulation state"""
    def __init__(self):
        self.params = create_offensive_setup()
        self.params.dt = 0.001

        self.ball = Ball(self.params)
        self.table = Table(self.params)
        self.racket = Racket(self.params, side=-1)
        self.collision = CollisionHandler(self.params)

        # Default positions
        self.ball_pos = np.array([0.0, 0.0, 1.0])
        self.ball_vel = np.array([0.0, 0.0, 0.0])
        self.ball_spin = np.array([0.0, 0.0, 0.0])

        self.racket_pos = np.array([-1.2, 0.0, 0.9])
        self.racket_angle = np.array([10.0, 0.0])  # pitch, yaw in degrees

        self.power = 15.0
        self.spin_type = "topspin"
        self.spin_rpm = 3000.0

        self.ball_active = False
        self.trajectory = []
        self.events = []

    def reset(self):
        """Reset to initial state"""
        self.ball = Ball(self.params)
        self.ball_pos = np.array([0.0, 0.0, 1.0])
        self.ball_vel = np.array([0.0, 0.0, 0.0])
        self.ball_spin = np.array([0.0, 0.0, 0.0])
        self.ball_active = False
        self.trajectory = []
        self.events = []


class TableTennisConsole(cmd.Cmd):
    """Interactive console for table tennis simulation"""

    intro = """
============================================================
     Table Tennis Physics Simulation - Console Interface
============================================================
  Type 'help' for available commands
  Type 'tutorial' for a quick start guide
============================================================
"""
    prompt = 'TT> '

    def __init__(self):
        super().__init__()
        self.state = SimulationState()
        self.viewer = None
        self.running_sim = False

    # ===== SET Commands =====

    def do_set(self, arg):
        """
Set simulation parameters.

Usage:
    set ball <x> <y> <z>           - Set ball position
    set velocity <vx> <vy> <vz>    - Set ball velocity (m/s)
    set spin <type> <rpm>          - Set spin (topspin/backspin/sidespin/none)
    set racket <x> <y> <z>         - Set racket position
    set angle <pitch> <yaw>        - Set racket angle (degrees)
    set power <value>              - Set swing power (m/s)
    set setup <type>               - Set equipment (offensive/defensive/allround)

Examples:
    set ball 0 0 1
    set velocity 10 0 2
    set spin topspin 3000
    set racket -1.2 0 0.9
    set angle 15 0
    set power 20
        """
        args = arg.split()
        if len(args) < 2:
            print("Usage: set <parameter> <values>")
            print("Type 'help set' for details")
            return

        param = args[0].lower()
        values = args[1:]

        try:
            if param == 'ball':
                if len(values) >= 3:
                    self.state.ball_pos = np.array([float(v) for v in values[:3]])
                    print(f"Ball position set to: {self.state.ball_pos}")
                else:
                    print("Usage: set ball <x> <y> <z>")

            elif param == 'velocity' or param == 'vel':
                if len(values) >= 3:
                    self.state.ball_vel = np.array([float(v) for v in values[:3]])
                    speed = np.linalg.norm(self.state.ball_vel)
                    print(f"Ball velocity set to: {self.state.ball_vel} ({speed:.1f} m/s)")
                else:
                    print("Usage: set velocity <vx> <vy> <vz>")

            elif param == 'spin':
                if len(values) >= 1:
                    spin_type = values[0].lower()
                    rpm = float(values[1]) if len(values) > 1 else self.state.spin_rpm

                    if spin_type in ['topspin', 'top', 't']:
                        self.state.spin_type = "topspin"
                    elif spin_type in ['backspin', 'back', 'b']:
                        self.state.spin_type = "backspin"
                    elif spin_type in ['sidespin', 'side', 's']:
                        self.state.spin_type = "sidespin"
                    elif spin_type in ['none', 'no', 'n', '0']:
                        self.state.spin_type = "none"
                        rpm = 0
                    else:
                        print(f"Unknown spin type: {spin_type}")
                        return

                    self.state.spin_rpm = rpm
                    self._update_ball_spin()
                    print(f"Spin set to: {self.state.spin_type} at {rpm:.0f} RPM")
                else:
                    print("Usage: set spin <type> [rpm]")

            elif param == 'racket':
                if len(values) >= 3:
                    self.state.racket_pos = np.array([float(v) for v in values[:3]])
                    print(f"Racket position set to: {self.state.racket_pos}")
                else:
                    print("Usage: set racket <x> <y> <z>")

            elif param == 'angle':
                if len(values) >= 1:
                    pitch = float(values[0])
                    yaw = float(values[1]) if len(values) > 1 else 0.0
                    self.state.racket_angle = np.array([pitch, yaw])
                    print(f"Racket angle set to: pitch={pitch}°, yaw={yaw}°")
                else:
                    print("Usage: set angle <pitch> [yaw]")

            elif param == 'power':
                if len(values) >= 1:
                    self.state.power = float(values[0])
                    print(f"Power set to: {self.state.power} m/s")
                else:
                    print("Usage: set power <value>")

            elif param == 'setup':
                if len(values) >= 1:
                    setup_type = values[0].lower()
                    if setup_type == 'offensive':
                        self.state.params = create_offensive_setup()
                    elif setup_type == 'defensive':
                        self.state.params = create_defensive_setup()
                    elif setup_type == 'allround':
                        self.state.params = create_allround_setup()
                    else:
                        print(f"Unknown setup: {setup_type}")
                        return
                    self._reinit_physics()
                    print(f"Equipment set to: {setup_type}")
                else:
                    print("Usage: set setup <offensive/defensive/allround>")

            else:
                print(f"Unknown parameter: {param}")
                print("Available: ball, velocity, spin, racket, angle, power, setup")

        except ValueError as e:
            print(f"Error: Invalid value - {e}")

    def _update_ball_spin(self):
        """Update ball spin vector based on type and RPM"""
        spin_rad = self.state.spin_rpm * 2 * np.pi / 60
        if self.state.spin_type == "topspin":
            self.state.ball_spin = np.array([0.0, spin_rad, 0.0])
        elif self.state.spin_type == "backspin":
            self.state.ball_spin = np.array([0.0, -spin_rad, 0.0])
        elif self.state.spin_type == "sidespin":
            self.state.ball_spin = np.array([0.0, 0.0, spin_rad])
        else:
            self.state.ball_spin = np.array([0.0, 0.0, 0.0])

    def _reinit_physics(self):
        """Reinitialize physics objects with current params"""
        self.state.params.dt = 0.001
        self.state.ball = Ball(self.state.params)
        self.state.table = Table(self.state.params)
        self.state.racket = Racket(self.state.params, side=-1)
        self.state.collision = CollisionHandler(self.state.params)

    # ===== Action Commands =====

    def do_serve(self, arg):
        """
Serve the ball from current racket position.
The ball is launched in the direction the racket is facing.

Usage: serve [power]

Examples:
    serve        # Use default power
    serve 20     # Serve with 20 m/s
        """
        args = arg.split()
        power = float(args[0]) if args else self.state.power

        # Calculate direction from racket angle
        pitch = np.radians(self.state.racket_angle[0])
        yaw = np.radians(self.state.racket_angle[1])

        direction = np.array([
            np.cos(pitch) * np.cos(yaw),
            np.cos(pitch) * np.sin(yaw),
            np.sin(pitch) * 0.3 + 0.15
        ])
        direction = direction / np.linalg.norm(direction)

        # Set ball state
        self.state.ball_pos = self.state.racket_pos + direction * 0.1
        self.state.ball_vel = direction * power
        self._update_ball_spin()

        self.state.ball.reset(
            position=self.state.ball_pos.copy(),
            velocity=self.state.ball_vel.copy(),
            spin=self.state.ball_spin.copy()
        )
        self.state.ball_active = True
        self.state.trajectory = [self.state.ball_pos.copy()]
        self.state.events = []

        speed = np.linalg.norm(self.state.ball_vel)
        print(f"Served! Speed: {speed:.1f} m/s, Spin: {self.state.spin_type} {self.state.spin_rpm:.0f} RPM")

    def do_hit(self, arg):
        """
Hit the ball with the racket.
Uses current racket position, angle, power, and spin settings.

Usage: hit [power]
        """
        args = arg.split()
        power = float(args[0]) if args else self.state.power

        # Calculate hit direction
        pitch = np.radians(self.state.racket_angle[0])
        yaw = np.radians(self.state.racket_angle[1])

        direction = np.array([
            np.cos(pitch) * np.cos(yaw),
            np.cos(pitch) * np.sin(yaw),
            np.sin(pitch)
        ])
        direction = direction / np.linalg.norm(direction)

        self.state.ball_vel = direction * power
        self._update_ball_spin()

        self.state.ball.velocity = self.state.ball_vel.copy()
        self.state.ball.spin = self.state.ball_spin.copy()
        self.state.ball_active = True

        print(f"Hit! Direction: ({direction[0]:.2f}, {direction[1]:.2f}, {direction[2]:.2f}), Power: {power:.1f} m/s")

    def do_launch(self, arg):
        """
Launch the ball with specific velocity.
Places ball at current position and applies velocity.

Usage: launch <vx> <vy> <vz>

Example:
    launch 10 0 2    # Launch with velocity (10, 0, 2) m/s
        """
        args = arg.split()
        if len(args) < 3:
            print("Usage: launch <vx> <vy> <vz>")
            return

        try:
            velocity = np.array([float(v) for v in args[:3]])
            self._update_ball_spin()

            self.state.ball.reset(
                position=self.state.ball_pos.copy(),
                velocity=velocity,
                spin=self.state.ball_spin.copy()
            )
            self.state.ball_active = True
            self.state.trajectory = [self.state.ball_pos.copy()]
            self.state.events = []

            speed = np.linalg.norm(velocity)
            print(f"Launched! Velocity: {velocity}, Speed: {speed:.1f} m/s")

        except ValueError:
            print("Error: Invalid velocity values")

    def do_run(self, arg):
        """
Run the simulation for specified duration.

Usage: run [duration] [step_output]

Arguments:
    duration    - Simulation time in seconds (default: 2.0)
    step_output - Print status every N steps (default: 500)

Examples:
    run          # Run for 2 seconds
    run 3        # Run for 3 seconds
    run 2 100    # Run 2 seconds, print every 100 steps
        """
        args = arg.split()
        duration = float(args[0]) if args else 2.0
        step_output = int(args[1]) if len(args) > 1 else 500

        if not self.state.ball_active:
            print("Ball is not active. Use 'serve' or 'launch' first.")
            return

        print(f"\nRunning simulation for {duration} seconds...")
        print("-" * 60)

        dt = self.state.params.dt
        steps = int(duration / dt)
        bounces = 0
        net_hits = 0

        start_pos = self.state.ball.position.copy()
        start_speed = self.state.ball.get_speed()

        for i in range(steps):
            # Update physics
            self.state.ball.update(dt)

            # Store trajectory
            self.state.trajectory.append(self.state.ball.position.copy())

            # Check collisions
            if self.state.collision.handle_ball_table_collision(self.state.ball, self.state.table):
                bounces += 1
                self.state.events.append({
                    'step': i,
                    'time': i * dt,
                    'type': 'bounce',
                    'position': self.state.ball.position.copy()
                })

            if self.state.collision.handle_ball_net_collision(self.state.ball, self.state.table):
                net_hits += 1
                self.state.events.append({
                    'step': i,
                    'time': i * dt,
                    'type': 'net',
                    'position': self.state.ball.position.copy()
                })

            # Check out of bounds
            if self.state.ball.position[2] < -0.1:
                print(f"\n[Step {i}] Ball hit the ground at z={self.state.ball.position[2]:.3f}")
                break
            if np.linalg.norm(self.state.ball.position[:2]) > 5:
                print(f"\n[Step {i}] Ball out of bounds")
                break

            # Print status
            if step_output > 0 and i % step_output == 0:
                pos = self.state.ball.position
                speed = self.state.ball.get_speed()
                spin = self.state.ball.get_spin_rpm()
                print(f"[{i*dt:.3f}s] pos=({pos[0]:6.2f}, {pos[1]:6.2f}, {pos[2]:6.2f})  "
                      f"speed={speed:5.1f}m/s  spin={spin:5.0f}RPM")

        # Final report
        end_pos = self.state.ball.position
        end_speed = self.state.ball.get_speed()

        print("-" * 60)
        print(f"\nSimulation complete!")
        print(f"  Duration: {min(i * dt, duration):.3f} s ({i} steps)")
        print(f"  Start:    ({start_pos[0]:.2f}, {start_pos[1]:.2f}, {start_pos[2]:.2f}) at {start_speed:.1f} m/s")
        print(f"  End:      ({end_pos[0]:.2f}, {end_pos[1]:.2f}, {end_pos[2]:.2f}) at {end_speed:.1f} m/s")
        print(f"  Bounces:  {bounces}")
        print(f"  Net hits: {net_hits}")

        # Update state
        self.state.ball_pos = end_pos.copy()
        self.state.ball_vel = self.state.ball.velocity.copy()

    def do_step(self, arg):
        """
Run simulation for specified number of steps.

Usage: step [count]

Example:
    step 100    # Run 100 physics steps
        """
        args = arg.split()
        count = int(args[0]) if args else 100

        if not self.state.ball_active:
            print("Ball is not active. Use 'serve' or 'launch' first.")
            return

        dt = self.state.params.dt
        for _ in range(count):
            self.state.ball.update(dt)
            self.state.trajectory.append(self.state.ball.position.copy())
            self.state.collision.handle_ball_table_collision(self.state.ball, self.state.table)
            self.state.collision.handle_ball_net_collision(self.state.ball, self.state.table)

        pos = self.state.ball.position
        print(f"After {count} steps: pos=({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}), "
              f"speed={self.state.ball.get_speed():.2f} m/s")

    # ===== Info Commands =====

    def do_status(self, arg):
        """Show current simulation state."""
        print("\n" + "="*50)
        print("  Current Simulation State")
        print("="*50)

        print(f"\n[Ball]")
        print(f"  Position:  ({self.state.ball_pos[0]:.3f}, {self.state.ball_pos[1]:.3f}, {self.state.ball_pos[2]:.3f})")
        print(f"  Velocity:  ({self.state.ball_vel[0]:.2f}, {self.state.ball_vel[1]:.2f}, {self.state.ball_vel[2]:.2f})")
        print(f"  Speed:     {np.linalg.norm(self.state.ball_vel):.2f} m/s")
        print(f"  Spin:      {self.state.spin_type} at {self.state.spin_rpm:.0f} RPM")
        print(f"  Active:    {self.state.ball_active}")

        print(f"\n[Racket]")
        print(f"  Position:  ({self.state.racket_pos[0]:.3f}, {self.state.racket_pos[1]:.3f}, {self.state.racket_pos[2]:.3f})")
        print(f"  Angle:     pitch={self.state.racket_angle[0]:.1f}°, yaw={self.state.racket_angle[1]:.1f}°")
        print(f"  Power:     {self.state.power:.1f} m/s")

        print(f"\n[Equipment]")
        print(f"  Rubber:    {self.state.params.rubber_forehand.rubber_type.value}")
        print(f"  Friction:  {self.state.params.rubber_forehand.dynamic_friction:.2f}")
        print(f"  Spin coef: {self.state.params.rubber_forehand.spin_coefficient:.2f}")

        print(f"\n[Table]")
        print(f"  Size:      {self.state.params.table_length:.2f} x {self.state.params.table_width:.2f} m")
        print(f"  Height:    {self.state.params.table_height:.2f} m")
        print(f"  Net:       {self.state.params.table.net_height:.3f} m")

        if self.state.trajectory:
            print(f"\n[Trajectory]")
            print(f"  Points:    {len(self.state.trajectory)}")
            if len(self.state.trajectory) > 1:
                traj = np.array(self.state.trajectory)
                print(f"  Max height: {np.max(traj[:, 2]):.3f} m")
                print(f"  X range:   {np.min(traj[:, 0]):.2f} to {np.max(traj[:, 0]):.2f} m")

        print()

    def do_params(self, arg):
        """Show detailed physics parameters."""
        p = self.state.params
        print("\n" + "="*50)
        print("  Physics Parameters")
        print("="*50)

        print(f"\n[Ball]")
        print(f"  Mass:           {p.ball_mass * 1000:.2f} g")
        print(f"  Diameter:       {p.ball.diameter * 1000:.1f} mm")
        print(f"  Restitution:    {p.ball_restitution:.3f}")
        print(f"  Drag coeff:     {p.ball.drag_coefficient:.3f}")
        print(f"  Lift coeff:     {p.ball.lift_coefficient:.3f}")

        print(f"\n[Racket]")
        print(f"  Total mass:     {p.racket_mass * 1000:.1f} g")
        print(f"  Blade mass:     {p.blade.mass * 1000:.1f} g")
        print(f"  Blade stiffness:{p.blade.stiffness:.2f}")

        print(f"\n[Rubber - Forehand]")
        r = p.rubber_forehand
        print(f"  Type:           {r.rubber_type.value}")
        print(f"  Friction:       {r.dynamic_friction:.2f}")
        print(f"  Restitution:    {r.restitution:.2f}")
        print(f"  Spin coeff:     {r.spin_coefficient:.2f}")
        print(f"  Spin sens:      {r.spin_sensitivity:.2f}")

        print(f"\n[Environment]")
        print(f"  Gravity:        {p.gravity:.2f} m/s²")
        print(f"  Air density:    {p.air_density:.3f} kg/m³")
        print(f"  Temperature:    {p.environment.temperature:.1f} °C")

        print()

    def do_events(self, arg):
        """Show events from last simulation run."""
        if not self.state.events:
            print("No events recorded. Run a simulation first.")
            return

        print(f"\nEvents ({len(self.state.events)} total):")
        for e in self.state.events:
            pos = e['position']
            print(f"  [{e['time']:.3f}s] {e['type']:8s} at ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})")

    # ===== Visualization =====

    def do_show(self, arg):
        """
Show 3D visualization of the trajectory.

Usage: show [save_filename]

Examples:
    show              # Display plot
    show output.png   # Save to file
        """
        if not self.state.trajectory or len(self.state.trajectory) < 2:
            print("No trajectory to show. Run a simulation first.")
            return

        try:
            import matplotlib
            matplotlib.use('TkAgg')  # Interactive backend
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D

            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111, projection='3d')

            # Draw table
            table = self.state.table
            hl = self.state.params.table_length / 2
            hw = self.state.params.table_width / 2
            h = self.state.params.table_height

            # Table surface
            table_x = [-hl, hl, hl, -hl, -hl]
            table_y = [-hw, -hw, hw, hw, -hw]
            table_z = [h, h, h, h, h]
            ax.plot(table_x, table_y, table_z, 'b-', linewidth=2)

            # Net
            nh = self.state.params.table.net_height
            ax.plot([0, 0], [-hw-0.15, hw+0.15], [h, h], 'g-', linewidth=1)
            ax.plot([0, 0], [-hw-0.15, hw+0.15], [h+nh, h+nh], 'g-', linewidth=1)
            ax.plot([0, 0], [-hw-0.15, -hw-0.15], [h, h+nh], 'g-', linewidth=1)
            ax.plot([0, 0], [hw+0.15, hw+0.15], [h, h+nh], 'g-', linewidth=1)

            # Trajectory
            traj = np.array(self.state.trajectory)
            ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], 'r-', linewidth=2, label='Ball trajectory')

            # Start and end points
            ax.scatter(*traj[0], color='green', s=100, label='Start')
            ax.scatter(*traj[-1], color='red', s=100, label='End')

            # Bounce points
            for e in self.state.events:
                if e['type'] == 'bounce':
                    pos = e['position']
                    ax.scatter(*pos, color='blue', s=50, marker='^')

            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.set_zlabel('Z (m)')
            ax.set_title('Table Tennis Trajectory')
            ax.legend()

            # Set axis limits
            ax.set_xlim(-2, 2)
            ax.set_ylim(-1.5, 1.5)
            ax.set_zlim(0, 2)

            args = arg.split()
            if args:
                filename = args[0]
                plt.savefig(filename, dpi=150, bbox_inches='tight')
                print(f"Saved to: {filename}")

            plt.show()

        except ImportError as e:
            print(f"Error: Could not load matplotlib - {e}")
        except Exception as e:
            print(f"Error: {e}")

    def do_animate(self, arg):
        """
Open interactive 3D viewer with animation.

Usage: animate
        """
        print("Starting interactive viewer...")
        try:
            import subprocess
            subprocess.Popen([sys.executable, 'interactive.py'])
            print("Interactive viewer started in new window.")
        except Exception as e:
            print(f"Error starting viewer: {e}")

    # ===== Utility Commands =====

    def do_reset(self, arg):
        """Reset simulation to initial state."""
        self.state.reset()
        print("Simulation reset.")

    def do_clear(self, arg):
        """Clear the screen."""
        import os
        os.system('cls' if os.name == 'nt' else 'clear')

    def do_tutorial(self, arg):
        """Show a quick start tutorial."""
        print("""
============================================================
                    Quick Start Tutorial
============================================================

1. Basic serve and run:
   TT> serve
   TT> run 2

2. Custom ball placement and launch:
   TT> set ball -1 0 1         # Place ball at (-1, 0, 1)
   TT> set spin topspin 3000   # Set topspin at 3000 RPM
   TT> launch 10 0 2           # Launch with velocity (10, 0, 2)
   TT> run 2                   # Simulate for 2 seconds

3. Adjust racket and serve:
   TT> set racket -1.2 0 0.9   # Position racket
   TT> set angle 15 0          # Tilt 15 degrees up
   TT> set power 18            # Set power to 18 m/s
   TT> serve                   # Serve!
   TT> run 3                   # Watch for 3 seconds

4. View results:
   TT> status                  # Show current state
   TT> events                  # Show bounce events
   TT> show                    # 3D visualization

5. Change equipment:
   TT> set setup defensive     # Use defensive setup
   TT> params                  # View all parameters

Type 'help <command>' for detailed help on any command.
""")

    def do_example(self, arg):
        """
Run example simulations.

Usage: example <name>

Available examples:
    topspin   - Strong topspin serve
    backspin  - Backspin chop
    sidespin  - Sidespin serve
    flat      - Flat fast serve
    lob       - High defensive lob
        """
        args = arg.split()
        if not args:
            print("Usage: example <name>")
            print("Available: topspin, backspin, sidespin, flat, lob")
            return

        name = args[0].lower()

        if name == 'topspin':
            print("\n=== Topspin Serve ===")
            self.onecmd("set racket -1.2 0 0.9")
            self.onecmd("set angle 20 0")
            self.onecmd("set spin topspin 4000")
            self.onecmd("set power 15")
            self.onecmd("serve")
            self.onecmd("run 2")

        elif name == 'backspin':
            print("\n=== Backspin Chop ===")
            self.onecmd("set ball 1 0 1.2")
            self.onecmd("set spin backspin 3500")
            self.onecmd("launch -8 0 3")
            self.onecmd("run 2")

        elif name == 'sidespin':
            print("\n=== Sidespin Serve ===")
            self.onecmd("set racket -1.2 0.2 0.9")
            self.onecmd("set angle 10 15")
            self.onecmd("set spin sidespin 3000")
            self.onecmd("set power 12")
            self.onecmd("serve")
            self.onecmd("run 2")

        elif name == 'flat':
            print("\n=== Flat Fast Serve ===")
            self.onecmd("set racket -1.2 0 0.85")
            self.onecmd("set angle 5 0")
            self.onecmd("set spin none 0")
            self.onecmd("set power 22")
            self.onecmd("serve")
            self.onecmd("run 1.5")

        elif name == 'lob':
            print("\n=== Defensive Lob ===")
            self.onecmd("set ball 1 0 0.8")
            self.onecmd("set spin backspin 2000")
            self.onecmd("launch -5 0 8")
            self.onecmd("run 3")

        else:
            print(f"Unknown example: {name}")

    def do_quit(self, arg):
        """Exit the console."""
        print("Goodbye!")
        return True

    def do_exit(self, arg):
        """Exit the console."""
        return self.do_quit(arg)

    def do_EOF(self, arg):
        """Handle Ctrl+D."""
        print()
        return self.do_quit(arg)

    # Shortcuts
    do_q = do_quit
    do_s = do_status
    do_r = do_run
    do_h = cmd.Cmd.do_help


def main():
    console = TableTennisConsole()
    try:
        console.cmdloop()
    except KeyboardInterrupt:
        print("\nInterrupted. Goodbye!")


if __name__ == "__main__":
    main()
