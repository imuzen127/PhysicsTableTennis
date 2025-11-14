"""
動作記録の例

カスタムラケット動作を記録して保存・再生する
"""

import sys
sys.path.append('..')

import numpy as np
from src.physics.parameters import PhysicsParameters
from src.simulation.engine import TableTennisEngine
from src.simulation.motion import MotionRecorder, MotionPlayer
from src.visualization.viewer import Viewer3D


def create_custom_motion():
    """
    カスタム動作を作成

    Returns:
        recorder: 記録された動作データを持つレコーダー
    """
    print("カスタム動作を作成中...")

    recorder = MotionRecorder(name="custom_forehand")
    recorder.start_recording()

    # 時間とともに変化する動作を記録
    # 例: 円軌道を描くラケット動作
    dt = 0.01
    duration = 2.0
    num_steps = int(duration / dt)

    for i in range(num_steps):
        t = i * dt

        # 円軌道
        radius = 0.3
        angle = 2 * np.pi * t / duration
        x = 1.2
        y = radius * np.sin(angle)
        z = 0.9 + radius * (1 - np.cos(angle))

        position = np.array([x, y, z])

        # 姿勢（常に中心を向く）
        orientation = np.array([-1.0, -y / radius, 0.1])
        orientation = orientation / np.linalg.norm(orientation)

        recorder.record_frame(t, position, orientation)

    recorder.stop_recording()

    return recorder


def main():
    print("=== 動作記録の例 ===\n")

    # カスタム動作を作成
    recorder = create_custom_motion()

    # 動作データを保存
    recorder.save('custom_motion.npz')
    print(f"動作データを保存しました: custom_motion.npz\n")

    # 動作データを読み込み
    print("動作データを読み込み中...")
    player = MotionPlayer(filename='custom_motion.npz')

    # 物理パラメータを設定
    params = PhysicsParameters()
    engine = TableTennisEngine(params)

    # 初期条件
    initial_position = np.array([-1.0, 0.0, 1.0])
    initial_velocity = np.array([7.0, 0.0, 2.0])
    initial_spin = np.array([0.0, 40.0, 0.0])

    engine.reset(
        ball_position=initial_position,
        ball_velocity=initial_velocity,
        ball_spin=initial_spin
    )

    # コントローラを作成
    controller = player.create_controller()

    # シミュレーション実行
    print("\nシミュレーション実行中...")
    results = engine.run(
        duration=3.0,
        racket_1_controller=controller
    )

    # 結果を表示
    print("\n=== シミュレーション結果 ===")
    print(f"シミュレーション時間: {results['simulation_time']:.3f}秒")
    print(f"衝突回数:")
    for key, count in results['collision_count'].items():
        if count > 0:
            print(f"  {key}: {count}回")

    # 可視化
    print("\n3D可視化を生成中...")
    viewer = Viewer3D(engine)
    viewer.plot_simulation(results, show_rackets=True)
    viewer.save('motion_recording_result.png')
    print("可視化を保存しました: motion_recording_result.png")

    print("\n動作記録の例 完了!")
    print("\n次のステップ:")
    print("1. custom_motion.npz を編集して独自の動作を作成")
    print("2. 複数の動作を組み合わせて複雑なラリーを実現")
    print("3. 実験データから動作をインポート")


if __name__ == "__main__":
    main()
