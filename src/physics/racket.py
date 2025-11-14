"""
Racket physics module

Implements racket dynamics and collision with ball
"""

import numpy as np
from typing import Tuple, Optional, Callable
from .parameters import PhysicsParameters


class Racket:
    """
    卓球ラケットクラス

    ラケットの位置、姿勢、速度を管理
    """

    def __init__(
        self,
        params: PhysicsParameters,
        position: np.ndarray = None,
        orientation: np.ndarray = None,
        side: int = 1
    ):
        """
        Args:
            params: 物理パラメータ
            position: 初期位置 [x, y, z] (m)
            orientation: 初期姿勢（法線ベクトル）
            side: どちら側のラケットか (1: +X側, -1: -X側)
        """
        self.params = params
        self.side = side  # 1 or -1

        # デフォルト位置（テーブルの端）
        default_x = side * (params.table_length / 2 + 0.5)
        self.position = position if position is not None else np.array([default_x, 0.0, 1.0])

        # デフォルト姿勢（相手側を向く）
        default_normal = np.array([-side, 0.0, 0.0])
        self.orientation = orientation if orientation is not None else default_normal
        self.orientation = self.orientation / np.linalg.norm(self.orientation)  # 正規化

        # 速度
        self.velocity = np.zeros(3)

        # ラケットの形状（楕円形として近似）
        self.radius_major = 0.08  # 長軸半径 (m)
        self.radius_minor = 0.075  # 短軸半径 (m)

        # 軌跡記録（動作登録用）
        self.trajectory = [self.position.copy()]
        self.orientation_history = [self.orientation.copy()]
        self.velocity_history = [self.velocity.copy()]
        self.time_stamps = [0.0]

    def update_position(
        self,
        position: np.ndarray,
        orientation: Optional[np.ndarray] = None,
        dt: Optional[float] = None
    ):
        """
        ラケットの位置と姿勢を更新

        Args:
            position: 新しい位置
            orientation: 新しい姿勢（法線ベクトル）
            dt: タイムステップ
        """
        if dt is None:
            dt = self.params.dt

        # 速度を計算
        self.velocity = (position - self.position) / dt
        self.position = position.copy()

        if orientation is not None:
            self.orientation = orientation / np.linalg.norm(orientation)

        # 軌跡を記録
        self.trajectory.append(self.position.copy())
        self.orientation_history.append(self.orientation.copy())
        self.velocity_history.append(self.velocity.copy())
        if len(self.time_stamps) > 0:
            self.time_stamps.append(self.time_stamps[-1] + dt)
        else:
            self.time_stamps.append(dt)

    def check_collision(
        self,
        ball_position: np.ndarray,
        ball_radius: float
    ) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        ボールとラケットの衝突をチェック

        Args:
            ball_position: ボールの位置
            ball_radius: ボールの半径

        Returns:
            is_collision: 衝突したかどうか
            contact_point: 接触点
            normal: 衝突面の法線ベクトル
        """
        # ボールからラケット中心への距離
        diff = ball_position - self.position
        distance = np.linalg.norm(diff)

        # 簡易的な球-円盤衝突判定
        # ラケット面への投影距離
        normal_distance = abs(np.dot(diff, self.orientation))

        # ラケット面内での距離
        projection = diff - normal_distance * self.orientation
        planar_distance = np.linalg.norm(projection)

        # 衝突判定
        if normal_distance < ball_radius and planar_distance < self.radius_major:
            contact_point = self.position + projection
            normal = self.orientation.copy()
            return True, contact_point, normal

        return False, None, None

    def compute_impact(
        self,
        ball_velocity: np.ndarray,
        ball_spin: np.ndarray,
        contact_point: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        衝突時のボールの速度とスピン変化を計算

        Args:
            ball_velocity: 衝突前のボール速度
            ball_spin: 衝突前のボールスピン
            contact_point: 接触点

        Returns:
            new_velocity: 衝突後のボール速度
            new_spin: 衝突後のボールスピン
        """
        # 相対速度
        relative_velocity = ball_velocity - self.velocity

        # 法線方向と接線方向の分解
        v_normal = np.dot(relative_velocity, self.orientation) * self.orientation
        v_tangent = relative_velocity - v_normal

        # 反発係数を適用
        v_normal_new = -self.params.racket_restitution * v_normal

        # 摩擦によるスピンの付与
        # 簡略化: 接線方向の速度からスピンを生成
        spin_axis = np.cross(self.orientation, v_tangent)
        if np.linalg.norm(spin_axis) > 1e-6:
            spin_axis = spin_axis / np.linalg.norm(spin_axis)
            spin_magnitude = np.linalg.norm(v_tangent) * self.params.racket_friction / self.params.ball_radius
            spin_change = spin_axis * spin_magnitude
        else:
            spin_change = np.zeros(3)

        # 新しい速度とスピン
        new_velocity = self.velocity + v_normal_new + v_tangent * (1 - self.params.racket_friction)
        new_spin = ball_spin + spin_change

        return new_velocity, new_spin

    def get_motion_data(self) -> dict:
        """動作データを取得（動作登録用）"""
        return {
            'trajectory': np.array(self.trajectory),
            'orientation': np.array(self.orientation_history),
            'velocity': np.array(self.velocity_history),
            'time_stamps': np.array(self.time_stamps)
        }

    def reset(
        self,
        position: Optional[np.ndarray] = None,
        orientation: Optional[np.ndarray] = None
    ):
        """ラケットをリセット"""
        if position is not None:
            self.position = position.copy()
        if orientation is not None:
            self.orientation = orientation / np.linalg.norm(orientation)

        self.velocity = np.zeros(3)
        self.trajectory = [self.position.copy()]
        self.orientation_history = [self.orientation.copy()]
        self.velocity_history = [self.velocity.copy()]
        self.time_stamps = [0.0]

    def __repr__(self) -> str:
        return (f"Racket(pos={self.position}, orientation={self.orientation}, "
                f"side={self.side})")
