"""
3D visualization for table tennis simulation

Uses matplotlib for 3D plotting
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from typing import Optional, Dict, Any
from ..simulation.engine import TableTennisEngine


class Viewer3D:
    """
    3D可視化クラス

    シミュレーション結果を3Dで表示
    """

    def __init__(
        self,
        engine: Optional[TableTennisEngine] = None,
        figsize: tuple = (12, 8)
    ):
        """
        Args:
            engine: シミュレーションエンジン
            figsize: 図のサイズ
        """
        self.engine = engine
        self.figsize = figsize

        # 図とサブプロット
        self.fig = None
        self.ax = None

    def setup_plot(self):
        """プロットの初期設定"""
        self.fig = plt.figure(figsize=self.figsize)
        self.ax = self.fig.add_subplot(111, projection='3d')

        # 軸ラベル
        self.ax.set_xlabel('X (m)')
        self.ax.set_ylabel('Y (m)')
        self.ax.set_zlabel('Z (m)')

        # タイトル
        self.ax.set_title('Table Tennis Physics Simulation')

    def draw_table(self):
        """テーブルを描画"""
        if self.engine is None:
            return

        table = self.engine.table

        # テーブル面（矩形）
        corners = table.get_corner_positions()
        corners = np.vstack([corners, corners[0]])  # 閉じる

        self.ax.plot(
            corners[:, 0],
            corners[:, 1],
            corners[:, 2],
            'b-',
            linewidth=2,
            label='Table'
        )

        # テーブル面を塗りつぶし
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        table_surface = Poly3DCollection(
            [corners[:-1]],
            alpha=0.3,
            facecolor='blue',
            edgecolor='blue'
        )
        self.ax.add_collection3d(table_surface)

        # ネット
        net_start, net_end = table.get_net_positions()
        net_points = np.array([
            [net_start[0], net_start[1], net_start[2]],
            [net_end[0], net_end[1], net_end[2]],
        ])
        self.ax.plot(
            [net_points[0, 0], net_points[1, 0]],
            [net_points[0, 1], net_points[1, 1]],
            [net_end[2], net_end[2]],
            'g-',
            linewidth=3,
            label='Net'
        )

    def draw_trajectory(
        self,
        trajectory: np.ndarray,
        color: str = 'r',
        label: str = 'Ball',
        linewidth: float = 1.5,
        marker: str = None,
        markersize: float = 3
    ):
        """
        軌跡を描画

        Args:
            trajectory: 軌跡データ (N, 3)
            color: 線の色
            label: ラベル
            linewidth: 線の太さ
            marker: マーカー
            markersize: マーカーサイズ
        """
        self.ax.plot(
            trajectory[:, 0],
            trajectory[:, 1],
            trajectory[:, 2],
            color=color,
            label=label,
            linewidth=linewidth,
            marker=marker,
            markersize=markersize
        )

    def draw_ball_at_position(
        self,
        position: np.ndarray,
        radius: float = 0.020,
        color: str = 'orange'
    ):
        """
        ボールを描画

        Args:
            position: ボールの位置
            radius: ボールの半径
            color: ボールの色
        """
        # 球を描画（簡略化として点で表示）
        self.ax.scatter(
            [position[0]],
            [position[1]],
            [position[2]],
            color=color,
            s=200,
            alpha=0.8
        )

    def draw_racket_at_position(
        self,
        position: np.ndarray,
        orientation: np.ndarray,
        radius: float = 0.08,
        color: str = 'red'
    ):
        """
        ラケットを描画

        Args:
            position: ラケットの位置
            orientation: ラケットの姿勢（法線ベクトル）
            radius: ラケットの半径
            color: ラケットの色
        """
        # ラケット面を円で近似
        self.ax.scatter(
            [position[0]],
            [position[1]],
            [position[2]],
            color=color,
            s=500,
            marker='o',
            alpha=0.6
        )

        # 法線ベクトルを矢印で表示
        self.ax.quiver(
            position[0],
            position[1],
            position[2],
            orientation[0] * 0.1,
            orientation[1] * 0.1,
            orientation[2] * 0.1,
            color=color,
            arrow_length_ratio=0.3
        )

    def plot_simulation(
        self,
        results: Optional[Dict[str, Any]] = None,
        show_rackets: bool = True,
        show_ball_start_end: bool = True
    ):
        """
        シミュレーション結果を可視化

        Args:
            results: シミュレーション結果（Noneの場合はengineから取得）
            show_rackets: ラケットの軌跡を表示するか
            show_ball_start_end: ボールの開始・終了位置を表示するか
        """
        if results is None:
            if self.engine is None:
                raise ValueError("No simulation results or engine provided")
            results = self.engine.get_results()

        # プロット設定
        self.setup_plot()

        # テーブルを描画
        self.draw_table()

        # ボールの軌跡
        ball_trajectory = results['ball_trajectory']
        self.draw_trajectory(
            ball_trajectory,
            color='orange',
            label='Ball Trajectory',
            linewidth=2
        )

        # ボールの開始・終了位置
        if show_ball_start_end and len(ball_trajectory) > 0:
            self.draw_ball_at_position(ball_trajectory[0], color='green')
            self.draw_ball_at_position(ball_trajectory[-1], color='red')

        # ラケットの軌跡
        if show_rackets:
            racket_1_traj = results['racket_1_trajectory']
            racket_2_traj = results['racket_2_trajectory']

            if len(racket_1_traj) > 1:
                self.draw_trajectory(
                    racket_1_traj,
                    color='blue',
                    label='Racket 1',
                    linewidth=1,
                    marker='.',
                    markersize=2
                )

            if len(racket_2_traj) > 1:
                self.draw_trajectory(
                    racket_2_traj,
                    color='cyan',
                    label='Racket 2',
                    linewidth=1,
                    marker='.',
                    markersize=2
                )

        # 軸の範囲を設定
        self._set_axis_limits()

        # 凡例
        self.ax.legend()

        # グリッド
        self.ax.grid(True, alpha=0.3)

        return self.fig, self.ax

    def _set_axis_limits(self):
        """軸の範囲を設定"""
        if self.engine is None:
            return

        table = self.engine.table

        # X軸（テーブルの長さ方向）
        margin = 1.0
        self.ax.set_xlim(
            table.bounds['x_min'] - margin,
            table.bounds['x_max'] + margin
        )

        # Y軸（テーブルの幅方向）
        self.ax.set_ylim(
            table.bounds['y_min'] - margin,
            table.bounds['y_max'] + margin
        )

        # Z軸（高さ方向）
        self.ax.set_zlim(0, 2.0)

    def show(self):
        """プロットを表示"""
        plt.show()

    def save(self, filename: str):
        """
        プロットを保存

        Args:
            filename: 保存先ファイル名
        """
        if self.fig is not None:
            self.fig.savefig(filename, dpi=150, bbox_inches='tight')
            print(f"Plot saved to {filename}")

    def create_animation(
        self,
        results: Optional[Dict[str, Any]] = None,
        interval: int = 20,
        repeat: bool = True
    ):
        """
        アニメーションを作成

        Args:
            results: シミュレーション結果
            interval: フレーム間隔（ミリ秒）
            repeat: リピート再生するか

        Returns:
            animation: matplotlibのアニメーションオブジェクト
        """
        if results is None:
            if self.engine is None:
                raise ValueError("No simulation results or engine provided")
            results = self.engine.get_results()

        # プロット設定
        self.setup_plot()
        self.draw_table()

        ball_trajectory = results['ball_trajectory']
        num_frames = len(ball_trajectory)

        # 初期化
        ball_point, = self.ax.plot([], [], [], 'o', color='orange', markersize=10)
        trail_line, = self.ax.plot([], [], [], '-', color='orange', linewidth=1, alpha=0.5)

        def init():
            ball_point.set_data([], [])
            ball_point.set_3d_properties([])
            trail_line.set_data([], [])
            trail_line.set_3d_properties([])
            return ball_point, trail_line

        def update(frame):
            # ボールの位置
            pos = ball_trajectory[frame]
            ball_point.set_data([pos[0]], [pos[1]])
            ball_point.set_3d_properties([pos[2]])

            # 軌跡
            trail = ball_trajectory[:frame+1]
            trail_line.set_data(trail[:, 0], trail[:, 1])
            trail_line.set_3d_properties(trail[:, 2])

            return ball_point, trail_line

        # 軸の範囲を設定
        self._set_axis_limits()

        # アニメーション作成
        anim = FuncAnimation(
            self.fig,
            update,
            frames=num_frames,
            init_func=init,
            interval=interval,
            repeat=repeat,
            blit=True
        )

        return anim

    def plot_velocity_profile(self, results: Optional[Dict[str, Any]] = None):
        """
        速度プロファイルをプロット

        Args:
            results: シミュレーション結果
        """
        if results is None:
            if self.engine is None:
                raise ValueError("No simulation results or engine provided")
            results = self.engine.get_results()

        velocity_history = results['ball_velocity_history']
        speed_history = np.linalg.norm(velocity_history, axis=1)

        time_points = np.arange(len(speed_history)) * self.engine.params.dt

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(time_points, speed_history, 'b-', linewidth=2)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Speed (m/s)')
        ax.set_title('Ball Speed over Time')
        ax.grid(True, alpha=0.3)

        return fig, ax
