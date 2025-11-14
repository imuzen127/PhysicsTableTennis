"""
Main simulation engine for table tennis physics

Integrates all physics components and manages simulation loop
"""

import numpy as np
from typing import Optional, List, Callable, Dict, Any
from ..physics.parameters import PhysicsParameters
from ..physics.ball import Ball
from ..physics.table import Table
from ..physics.racket import Racket
from ..physics.collision import CollisionHandler


class SimulationState:
    """シミュレーション状態を保持"""

    def __init__(self):
        self.time = 0.0
        self.step_count = 0
        self.is_running = False
        self.collision_count = {
            'table': 0,
            'net': 0,
            'racket_1': 0,
            'racket_2': 0
        }


class TableTennisEngine:
    """
    卓球物理エンジンのメインクラス

    全ての物理計算とシミュレーションを統合管理
    """

    def __init__(
        self,
        params: Optional[PhysicsParameters] = None,
        enable_logging: bool = True
    ):
        """
        Args:
            params: 物理パラメータ（Noneの場合はデフォルト値を使用）
            enable_logging: ログ記録を有効にするか
        """
        self.params = params if params is not None else PhysicsParameters()
        self.enable_logging = enable_logging

        # 物理オブジェクト
        self.ball = Ball(self.params)
        self.table = Table(self.params)
        self.racket_1 = Racket(self.params, side=1)  # +X側
        self.racket_2 = Racket(self.params, side=-1)  # -X側

        # 衝突ハンドラ
        self.collision_handler = CollisionHandler(self.params)

        # シミュレーション状態
        self.state = SimulationState()

        # イベントログ
        self.events = []

    def reset(
        self,
        ball_position: Optional[np.ndarray] = None,
        ball_velocity: Optional[np.ndarray] = None,
        ball_spin: Optional[np.ndarray] = None
    ):
        """
        シミュレーションをリセット

        Args:
            ball_position: ボールの初期位置
            ball_velocity: ボールの初期速度
            ball_spin: ボールの初期スピン
        """
        self.ball.reset(ball_position, ball_velocity, ball_spin)
        self.racket_1.reset()
        self.racket_2.reset()

        self.state = SimulationState()
        self.events = []

        if self.enable_logging:
            self._log_event('reset', 'Simulation reset')

    def step(self, dt: Optional[float] = None) -> bool:
        """
        シミュレーションを1ステップ進める

        Args:
            dt: タイムステップ（Noneの場合はparams.dtを使用）

        Returns:
            continue_simulation: シミュレーションを続けるかどうか
        """
        if dt is None:
            dt = self.params.dt

        # ボールの状態を更新
        self.ball.update(dt)

        # 衝突検出と応答
        self._handle_collisions()

        # 場外チェック
        if self.collision_handler.check_out_of_bounds(self.ball, self.table):
            self._log_event('out_of_bounds', f'Ball out of bounds at {self.ball.position}')
            self.state.is_running = False
            return False

        # 時間を進める
        self.state.time += dt
        self.state.step_count += 1

        # 最大シミュレーション時間チェック
        if self.state.time >= self.params.max_simulation_time:
            self._log_event('timeout', 'Simulation timeout')
            self.state.is_running = False
            return False

        return True

    def _handle_collisions(self):
        """衝突処理"""

        # ボールとテーブルの衝突
        if self.collision_handler.handle_ball_table_collision(self.ball, self.table):
            self.state.collision_count['table'] += 1
            self._log_event('collision', f'Ball-Table collision at {self.ball.position}')

        # ボールとネットの衝突
        if self.collision_handler.handle_ball_net_collision(self.ball, self.table):
            self.state.collision_count['net'] += 1
            self._log_event('collision', f'Ball-Net collision at {self.ball.position}')

        # ボールとラケット1の衝突
        if self.collision_handler.handle_ball_racket_collision(self.ball, self.racket_1):
            self.state.collision_count['racket_1'] += 1
            self._log_event('collision', f'Ball-Racket1 collision at {self.ball.position}')

        # ボールとラケット2の衝突
        if self.collision_handler.handle_ball_racket_collision(self.ball, self.racket_2):
            self.state.collision_count['racket_2'] += 1
            self._log_event('collision', f'Ball-Racket2 collision at {self.ball.position}')

    def run(
        self,
        duration: Optional[float] = None,
        racket_1_controller: Optional[Callable] = None,
        racket_2_controller: Optional[Callable] = None,
        callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        シミュレーションを実行

        Args:
            duration: シミュレーション時間（秒）
            racket_1_controller: ラケット1の制御関数 (time, ball_state) -> (position, orientation)
            racket_2_controller: ラケット2の制御関数
            callback: 各ステップ後に呼ばれるコールバック関数

        Returns:
            results: シミュレーション結果の辞書
        """
        if duration is None:
            duration = self.params.max_simulation_time

        self.state.is_running = True

        while self.state.is_running and self.state.time < duration:
            # ラケットを制御
            if racket_1_controller is not None:
                pos, ori = racket_1_controller(self.state.time, self.ball.get_state())
                self.racket_1.update_position(pos, ori, self.params.dt)

            if racket_2_controller is not None:
                pos, ori = racket_2_controller(self.state.time, self.ball.get_state())
                self.racket_2.update_position(pos, ori, self.params.dt)

            # 1ステップ実行
            if not self.step():
                break

            # コールバック実行
            if callback is not None:
                callback(self)

        # 結果をまとめる
        results = self.get_results()
        return results

    def get_results(self) -> Dict[str, Any]:
        """シミュレーション結果を取得"""
        return {
            'ball_trajectory': np.array(self.ball.trajectory),
            'ball_velocity_history': np.array(self.ball.velocity_history),
            'ball_spin_history': np.array(self.ball.spin_history),
            'racket_1_trajectory': np.array(self.racket_1.trajectory),
            'racket_2_trajectory': np.array(self.racket_2.trajectory),
            'simulation_time': self.state.time,
            'step_count': self.state.step_count,
            'collision_count': self.state.collision_count.copy(),
            'events': self.events.copy(),
            'final_ball_state': self.ball.get_state(),
        }

    def _log_event(self, event_type: str, message: str):
        """イベントをログに記録"""
        if self.enable_logging:
            event = {
                'time': self.state.time,
                'step': self.state.step_count,
                'type': event_type,
                'message': message
            }
            self.events.append(event)

    def update_parameters(self, **kwargs):
        """
        物理パラメータを更新

        例: engine.update_parameters(ball_mass=0.003, table_friction=0.6)
        """
        self.params.update(**kwargs)
        self._log_event('parameter_update', f'Parameters updated: {kwargs}')

    def export_trajectory(self, filename: str):
        """軌跡をファイルに保存"""
        results = self.get_results()
        np.savez(
            filename,
            ball_trajectory=results['ball_trajectory'],
            ball_velocity_history=results['ball_velocity_history'],
            ball_spin_history=results['ball_spin_history'],
            racket_1_trajectory=results['racket_1_trajectory'],
            racket_2_trajectory=results['racket_2_trajectory'],
            simulation_time=results['simulation_time'],
            collision_count=results['collision_count']
        )
        print(f"Trajectory saved to {filename}")

    def __repr__(self) -> str:
        return (f"TableTennisEngine(time={self.state.time:.3f}s, "
                f"steps={self.state.step_count}, "
                f"collisions={sum(self.state.collision_count.values())})")
