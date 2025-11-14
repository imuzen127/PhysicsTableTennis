"""
基本的なシミュレーションの例

ボールを打ち出して、テーブルに落ちるまでをシミュレート
"""

import sys
sys.path.append('..')

import numpy as np
from src.physics.parameters import PhysicsParameters
from src.simulation.engine import TableTennisEngine
from src.visualization.viewer import Viewer3D


def main():
    print("=== 基本的な卓球シミュレーション ===\n")

    # 物理パラメータを設定
    params = PhysicsParameters(
        ball_mass=0.0027,       # 2.7g
        ball_radius=0.020,      # 40mm直径
        ball_restitution=0.89,
        table_restitution=0.89,
        table_friction=0.5,
        dt=0.001,               # 1ms
        max_simulation_time=5.0
    )

    print("物理パラメータ:")
    print(f"  ボール質量: {params.ball_mass * 1000:.1f}g")
    print(f"  ボール半径: {params.ball_radius * 1000:.1f}mm")
    print(f"  反発係数: {params.ball_restitution}")
    print(f"  摩擦係数: {params.table_friction}\n")

    # エンジンを初期化
    engine = TableTennisEngine(params)

    # 初期条件を設定
    # ボールを相手コートに向かって打ち出す
    initial_position = np.array([-1.0, 0.0, 1.0])  # テーブルの自陣側、高さ1m
    initial_velocity = np.array([8.0, 0.0, 2.0])   # 前方8m/s、上方2m/s
    initial_spin = np.array([0.0, 50.0, 0.0])      # トップスピン 50 rad/s

    print("初期条件:")
    print(f"  位置: {initial_position}")
    print(f"  速度: {initial_velocity} (速さ: {np.linalg.norm(initial_velocity):.2f} m/s)")
    print(f"  スピン: {initial_spin} rad/s\n")

    # シミュレーションをリセット
    engine.reset(
        ball_position=initial_position,
        ball_velocity=initial_velocity,
        ball_spin=initial_spin
    )

    # シミュレーション実行
    print("シミュレーション実行中...")
    results = engine.run(duration=3.0)

    # 結果を表示
    print("\n=== シミュレーション結果 ===")
    print(f"シミュレーション時間: {results['simulation_time']:.3f}秒")
    print(f"ステップ数: {results['step_count']}")
    print(f"衝突回数:")
    for key, count in results['collision_count'].items():
        if count > 0:
            print(f"  {key}: {count}回")

    print(f"\n最終ボール位置: {results['final_ball_state'][0]}")
    print(f"最終ボール速度: {results['final_ball_state'][1]}")

    # イベントログを表示
    if len(results['events']) > 0:
        print("\n=== イベントログ ===")
        for event in results['events'][-10:]:  # 最後の10件を表示
            print(f"[{event['time']:.3f}s] {event['type']}: {event['message']}")

    # 3D可視化
    print("\n3D可視化を生成中...")
    viewer = Viewer3D(engine)
    viewer.plot_simulation(results)
    viewer.save('simulation_result.png')
    print("可視化を保存しました: simulation_result.png")

    # 速度プロファイルをプロット
    fig, ax = viewer.plot_velocity_profile(results)
    fig.savefig('velocity_profile.png')
    print("速度プロファイルを保存しました: velocity_profile.png")

    # 軌跡データを保存
    engine.export_trajectory('trajectory_data.npz')

    print("\nシミュレーション完了!")


if __name__ == "__main__":
    main()
