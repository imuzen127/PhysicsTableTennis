"""
ラケットを使ったラリーシミュレーション

動作登録システムを使用して、ボールを打ち合うシミュレーション
"""

import sys
sys.path.append('..')

import numpy as np
from src.physics.parameters import PhysicsParameters
from src.simulation.engine import TableTennisEngine
from src.simulation.motion import MotionPlayer, PredefinedMotions
from src.visualization.viewer import Viewer3D


def main():
    print("=== ラリーシミュレーション ===\n")

    # 物理パラメータを設定
    params = PhysicsParameters(
        dt=0.001,
        max_simulation_time=10.0
    )

    # エンジンを初期化
    engine = TableTennisEngine(params)

    # 初期条件（サーブ）
    # ラケット1（+X側）からサーブ
    initial_position = np.array([1.0, 0.0, 1.0])
    initial_velocity = np.array([-5.0, 0.0, 1.0])  # 相手コートに向かって
    initial_spin = np.array([0.0, 30.0, 0.0])     # トップスピン

    engine.reset(
        ball_position=initial_position,
        ball_velocity=initial_velocity,
        ball_spin=initial_spin
    )

    # 事前定義された動作を取得
    print("ラケット動作を設定中...")

    # ラケット1: フォアハンドドライブ
    motion_1 = PredefinedMotions.create_forehand_drive(
        side=1,
        table_length=params.table_length
    )
    player_1 = MotionPlayer(motion_data=motion_1)

    # ラケット2: 静的な防御姿勢
    motion_2 = PredefinedMotions.create_static_defense(
        side=-1,
        table_length=params.table_length
    )
    player_2 = MotionPlayer(motion_data=motion_2)

    print(f"ラケット1: {motion_1['name']}")
    print(f"ラケット2: {motion_2['name']}\n")

    # コントローラを作成
    controller_1 = player_1.create_controller()
    controller_2 = player_2.create_controller()

    # シミュレーション実行
    print("シミュレーション実行中...")

    # 進捗表示用のコールバック
    step_counter = [0]

    def progress_callback(engine):
        step_counter[0] += 1
        if step_counter[0] % 1000 == 0:
            print(f"  {engine.state.time:.2f}秒経過...")

    results = engine.run(
        duration=5.0,
        racket_1_controller=controller_1,
        racket_2_controller=controller_2,
        callback=progress_callback
    )

    # 結果を表示
    print("\n=== シミュレーション結果 ===")
    print(f"シミュレーション時間: {results['simulation_time']:.3f}秒")
    print(f"ステップ数: {results['step_count']}")
    print(f"\n衝突回数:")
    for key, count in results['collision_count'].items():
        print(f"  {key}: {count}回")

    # ラリーが続いたかチェック
    total_racket_hits = (results['collision_count']['racket_1'] +
                         results['collision_count']['racket_2'])
    table_hits = results['collision_count']['table']

    print(f"\nラケットヒット数: {total_racket_hits}")
    print(f"テーブルバウンド数: {table_hits}")

    if total_racket_hits > 0:
        print("✓ ラリーが成立しました！")
    else:
        print("✗ ラリーが成立しませんでした")

    # 3D可視化
    print("\n3D可視化を生成中...")
    viewer = Viewer3D(engine)
    viewer.plot_simulation(results, show_rackets=True)
    viewer.save('rally_simulation.png')
    print("可視化を保存しました: rally_simulation.png")

    # 速度プロファイル
    fig, ax = viewer.plot_velocity_profile(results)
    fig.savefig('rally_velocity_profile.png')
    print("速度プロファイルを保存しました: rally_velocity_profile.png")

    print("\nラリーシミュレーション完了!")


if __name__ == "__main__":
    main()
