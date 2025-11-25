"""
Table Tennis Physics Simulation - Main Entry Point

Usage:
    python main.py              # Run basic simulation
    python main.py --show       # Run and show 3D visualization window
    python main.py --help       # Show help
"""

import sys
import argparse
import numpy as np

# Add project root to path
sys.path.insert(0, '.')

from src.physics.parameters import (
    PhysicsParameters,
    create_offensive_setup,
    create_defensive_setup,
    create_allround_setup
)
from src.simulation.engine import TableTennisEngine
from src.visualization.viewer import Viewer3D


def parse_args():
    parser = argparse.ArgumentParser(description='Table Tennis Physics Simulation')
    parser.add_argument('--setup', choices=['offensive', 'defensive', 'allround'],
                        default='offensive', help='Equipment setup type')
    parser.add_argument('--speed', type=float, default=10.0,
                        help='Initial ball speed (m/s)')
    parser.add_argument('--spin', type=float, default=3000.0,
                        help='Initial spin (RPM)')
    parser.add_argument('--duration', type=float, default=3.0,
                        help='Simulation duration (seconds)')
    parser.add_argument('--show', action='store_true',
                        help='Show 3D visualization window')
    parser.add_argument('--save', type=str, default=None,
                        help='Save visualization to file')
    return parser.parse_args()


def create_params(setup_type: str) -> PhysicsParameters:
    if setup_type == 'offensive':
        return create_offensive_setup()
    elif setup_type == 'defensive':
        return create_defensive_setup()
    else:
        return create_allround_setup()


def main():
    args = parse_args()

    print("=" * 50)
    print("  Table Tennis Physics Simulation")
    print("=" * 50)
    print()

    # Create parameters
    params = create_params(args.setup)
    params.dt = 0.001
    params.max_simulation_time = args.duration + 1.0

    print(f"Setup: {args.setup}")
    print(f"Parameters:")
    print(f"  Ball: {params.ball_mass * 1000:.1f}g, {params.ball.diameter * 1000:.0f}mm")
    print(f"  Racket: {params.racket_mass * 1000:.1f}g")
    print(f"  Blade: {params.blade.stiffness:.1f} stiffness")
    print(f"  Forehand rubber: {params.rubber_forehand.rubber_type.value}")
    print(f"    - Friction: {params.rubber_forehand.dynamic_friction:.2f}")
    print(f"    - Spin coeff: {params.rubber_forehand.spin_coefficient:.2f}")
    print(f"  Backhand rubber: {params.rubber_backhand.rubber_type.value}")
    print()

    # Initialize engine
    engine = TableTennisEngine(params)

    # Initial conditions - topspin shot
    spin_rad = args.spin * 2 * np.pi / 60  # RPM to rad/s
    initial_position = np.array([-1.0, 0.0, 0.9])
    initial_velocity = np.array([args.speed, 0.0, 2.0])
    initial_spin = np.array([0.0, spin_rad, 0.0])  # Topspin

    print(f"Initial conditions:")
    print(f"  Position: {initial_position}")
    print(f"  Speed: {args.speed:.1f} m/s")
    print(f"  Spin: {args.spin:.0f} RPM (topspin)")
    print()

    # Reset and run
    engine.reset(
        ball_position=initial_position,
        ball_velocity=initial_velocity,
        ball_spin=initial_spin
    )

    print("Running simulation...")
    results = engine.run(duration=args.duration)

    # Results
    print()
    print("=" * 50)
    print("  Results")
    print("=" * 50)
    print(f"Simulation time: {results['simulation_time']:.3f} s")
    print(f"Steps: {results['step_count']}")
    print()

    print("Collisions:")
    total_collisions = 0
    for key, count in results['collision_count'].items():
        if count > 0:
            print(f"  {key}: {count}")
            total_collisions += count
    if total_collisions == 0:
        print("  (none)")
    print()

    # Trajectory analysis
    trajectory = results['ball_trajectory']
    velocity_history = results['ball_velocity_history']

    max_height_idx = np.argmax(trajectory[:, 2])
    max_height = trajectory[max_height_idx, 2]

    initial_speed = np.linalg.norm(velocity_history[0])
    final_speed = np.linalg.norm(velocity_history[-1])

    print(f"Trajectory analysis:")
    print(f"  Max height: {max_height:.3f} m")
    print(f"  Final position: x={trajectory[-1, 0]:.2f}, y={trajectory[-1, 1]:.2f}, z={trajectory[-1, 2]:.2f}")
    print(f"  Speed: {initial_speed:.1f} -> {final_speed:.1f} m/s")
    print()

    # Events
    if results['events']:
        print("Events (last 5):")
        for event in results['events'][-5:]:
            print(f"  [{event['time']:.3f}s] {event['type']}: {event['message'][:50]}")
        print()

    # Visualization
    if args.show or args.save:
        print("Creating 3D visualization...")
        viewer = Viewer3D(engine)
        viewer.plot_simulation(results)

        if args.save:
            viewer.save(args.save)
            print(f"Saved to: {args.save}")

        if args.show:
            print("Opening visualization window...")
            viewer.show()

    print("Done!")
    return results


if __name__ == "__main__":
    main()
