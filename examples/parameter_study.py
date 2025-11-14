"""
パラメータスタディの例

異なる物理パラメータでシミュレーションを実行し、結果を比較
"""

import sys
sys.path.append('..')

import numpy as np
import matplotlib.pyplot as plt
from src.physics.parameters import PhysicsParameters
from src.simulation.engine import TableTennisEngine


def run_simulation_with_params(params_dict, initial_conditions):
    """
    指定されたパラメータでシミュレーションを実行

    Args:
        params_dict: パラメータの辞書
        initial_conditions: 初期条件

    Returns:
        results: シミュレーション結果
    """
    params = PhysicsParameters(**params_dict)
    engine = TableTennisEngine(params)

    engine.reset(
        ball_position=initial_conditions['position'],
        ball_velocity=initial_conditions['velocity'],
        ball_spin=initial_conditions['spin']
    )

    results = engine.run(duration=3.0)
    return results


def main():
    print("=== パラメータスタディ ===\n")

    # 初期条件（固定）
    initial_conditions = {
        'position': np.array([-1.0, 0.0, 1.0]),
        'velocity': np.array([8.0, 0.0, 2.0]),
        'spin': np.array([0.0, 50.0, 0.0])
    }

    # 実験1: 反発係数を変化させる
    print("実験1: 反発係数の影響を調査")
    restitution_values = [0.7, 0.8, 0.89, 0.95]
    results_restitution = {}

    for restitution in restitution_values:
        print(f"  反発係数 = {restitution}")
        params_dict = {
            'ball_restitution': restitution,
            'table_restitution': restitution,
        }
        results = run_simulation_with_params(params_dict, initial_conditions)
        results_restitution[restitution] = results

    # 実験2: 摩擦係数を変化させる
    print("\n実験2: 摩擦係数の影響を調査")
    friction_values = [0.1, 0.3, 0.5, 0.7]
    results_friction = {}

    for friction in friction_values:
        print(f"  摩擦係数 = {friction}")
        params_dict = {
            'table_friction': friction,
        }
        results = run_simulation_with_params(params_dict, initial_conditions)
        results_friction[friction] = results

    # 実験3: ボール質量を変化させる
    print("\n実験3: ボール質量の影響を調査")
    mass_values = [0.0020, 0.0027, 0.0035]  # 軽い、標準、重い
    results_mass = {}

    for mass in mass_values:
        print(f"  ボール質量 = {mass * 1000:.1f}g")
        params_dict = {
            'ball_mass': mass,
        }
        results = run_simulation_with_params(params_dict, initial_conditions)
        results_mass[mass] = results

    # 結果の可視化
    print("\n結果を可視化中...")

    # 図1: 反発係数の影響
    fig1, axes1 = plt.subplots(1, 2, figsize=(14, 5))

    for restitution, results in results_restitution.items():
        trajectory = results['ball_trajectory']
        axes1[0].plot(
            trajectory[:, 0],
            trajectory[:, 2],
            label=f'e = {restitution}'
        )

        velocity_history = results['ball_velocity_history']
        speed = np.linalg.norm(velocity_history, axis=1)
        time_points = np.arange(len(speed)) * 0.001

        axes1[1].plot(
            time_points,
            speed,
            label=f'e = {restitution}'
        )

    axes1[0].set_xlabel('X Position (m)')
    axes1[0].set_ylabel('Z Position (m)')
    axes1[0].set_title('Trajectory - Restitution Coefficient')
    axes1[0].legend()
    axes1[0].grid(True, alpha=0.3)
    axes1[0].axhline(y=0.76, color='b', linestyle='--', alpha=0.5, label='Table')

    axes1[1].set_xlabel('Time (s)')
    axes1[1].set_ylabel('Speed (m/s)')
    axes1[1].set_title('Speed - Restitution Coefficient')
    axes1[1].legend()
    axes1[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('parameter_study_restitution.png')
    print("保存しました: parameter_study_restitution.png")

    # 図2: 摩擦係数の影響
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))

    for friction, results in results_friction.items():
        trajectory = results['ball_trajectory']
        axes2[0].plot(
            trajectory[:, 0],
            trajectory[:, 2],
            label=f'μ = {friction}'
        )

        spin_history = results['ball_spin_history']
        spin_magnitude = np.linalg.norm(spin_history, axis=1)
        time_points = np.arange(len(spin_magnitude)) * 0.001

        axes2[1].plot(
            time_points,
            spin_magnitude,
            label=f'μ = {friction}'
        )

    axes2[0].set_xlabel('X Position (m)')
    axes2[0].set_ylabel('Z Position (m)')
    axes2[0].set_title('Trajectory - Friction Coefficient')
    axes2[0].legend()
    axes2[0].grid(True, alpha=0.3)
    axes2[0].axhline(y=0.76, color='b', linestyle='--', alpha=0.5)

    axes2[1].set_xlabel('Time (s)')
    axes2[1].set_ylabel('Spin (rad/s)')
    axes2[1].set_title('Spin - Friction Coefficient')
    axes2[1].legend()
    axes2[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('parameter_study_friction.png')
    print("保存しました: parameter_study_friction.png")

    # 図3: ボール質量の影響
    fig3, axes3 = plt.subplots(1, 2, figsize=(14, 5))

    for mass, results in results_mass.items():
        trajectory = results['ball_trajectory']
        axes3[0].plot(
            trajectory[:, 0],
            trajectory[:, 2],
            label=f'm = {mass * 1000:.1f}g'
        )

        velocity_history = results['ball_velocity_history']
        speed = np.linalg.norm(velocity_history, axis=1)
        time_points = np.arange(len(speed)) * 0.001

        axes3[1].plot(
            time_points,
            speed,
            label=f'm = {mass * 1000:.1f}g'
        )

    axes3[0].set_xlabel('X Position (m)')
    axes3[0].set_ylabel('Z Position (m)')
    axes3[0].set_title('Trajectory - Ball Mass')
    axes3[0].legend()
    axes3[0].grid(True, alpha=0.3)
    axes3[0].axhline(y=0.76, color='b', linestyle='--', alpha=0.5)

    axes3[1].set_xlabel('Time (s)')
    axes3[1].set_ylabel('Speed (m/s)')
    axes3[1].set_title('Speed - Ball Mass')
    axes3[1].legend()
    axes3[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('parameter_study_mass.png')
    print("保存しました: parameter_study_mass.png")

    print("\nパラメータスタディ完了!")


if __name__ == "__main__":
    main()
