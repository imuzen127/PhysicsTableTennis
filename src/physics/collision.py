"""
Collision detection and response module

Handles all collision events in the simulation
"""

import numpy as np
from typing import Tuple, Optional
from .ball import Ball
from .table import Table
from .racket import Racket
from .parameters import PhysicsParameters


class CollisionHandler:
    """
    衝突検出と応答を処理するクラス
    """

    def __init__(self, params: PhysicsParameters):
        """
        Args:
            params: 物理パラメータ
        """
        self.params = params

    def handle_ball_table_collision(
        self,
        ball: Ball,
        table: Table
    ) -> bool:
        """
        ボールとテーブルの衝突を処理

        Args:
            ball: ボールオブジェクト
            table: テーブルオブジェクト

        Returns:
            collision_occurred: 衝突が発生したかどうか
        """
        # 前フレームの位置を取得
        if len(ball.trajectory) < 2:
            prev_position = ball.position
        else:
            prev_position = ball.trajectory[-2]

        # 衝突チェック
        is_collision, normal = table.check_collision(
            ball.position,
            self.params.ball_radius
        )

        if is_collision:
            # ボールをテーブル面に配置
            ball.position[2] = table.bounds['z'] + self.params.ball_radius

            # 速度の法線成分と接線成分に分解
            velocity = ball.velocity
            v_normal = np.dot(velocity, normal) * normal
            v_tangent = velocity - v_normal

            # 反発係数を適用
            v_normal_new = -self.params.table_restitution * v_normal

            # 摩擦を適用
            friction_force = self.params.table_friction
            v_tangent_new = v_tangent * (1 - friction_force)

            # スピンの影響を考慮
            # ボールのスピンによる接線方向の速度変化
            spin_velocity = np.cross(ball.spin, normal) * self.params.ball_radius
            v_tangent_new += spin_velocity * friction_force

            # 新しい速度
            ball.velocity = v_normal_new + v_tangent_new

            # スピンの変化（テーブルとの摩擦による）
            # 簡略化: 接線方向の速度差からスピンを計算
            spin_change_axis = np.cross(normal, v_tangent)
            if np.linalg.norm(spin_change_axis) > 1e-6:
                spin_change_axis = spin_change_axis / np.linalg.norm(spin_change_axis)
                spin_change_magnitude = np.linalg.norm(v_tangent) * friction_force / self.params.ball_radius
                ball.spin += spin_change_axis * spin_change_magnitude * 0.5

            return True

        return False

    def handle_ball_net_collision(
        self,
        ball: Ball,
        table: Table
    ) -> bool:
        """
        ボールとネットの衝突を処理

        Args:
            ball: ボールオブジェクト
            table: テーブルオブジェクト

        Returns:
            collision_occurred: 衝突が発生したかどうか
        """
        # 前フレームの位置を取得
        if len(ball.trajectory) < 2:
            prev_position = ball.position
        else:
            prev_position = ball.trajectory[-2]

        # ネット衝突チェック
        is_collision, normal = table.check_net_collision(
            ball.position,
            prev_position,
            self.params.ball_radius
        )

        if is_collision:
            # 速度の法線成分を反転（単純な反射）
            velocity = ball.velocity
            v_normal = np.dot(velocity, normal) * normal
            v_tangent = velocity - v_normal

            # エネルギー損失を考慮
            v_normal_new = -0.5 * v_normal  # ネットは柔らかいので大きく減衰
            v_tangent_new = v_tangent * 0.8

            ball.velocity = v_normal_new + v_tangent_new

            # スピンも減衰
            ball.spin *= 0.7

            return True

        return False

    def handle_ball_racket_collision(
        self,
        ball: Ball,
        racket: Racket
    ) -> bool:
        """
        ボールとラケットの衝突を処理

        Args:
            ball: ボールオブジェクト
            racket: ラケットオブジェクト

        Returns:
            collision_occurred: 衝突が発生したかどうか
        """
        # 衝突チェック
        is_collision, contact_point, normal = racket.check_collision(
            ball.position,
            self.params.ball_radius
        )

        if is_collision:
            # ラケットとの衝突計算
            new_velocity, new_spin = racket.compute_impact(
                ball.velocity,
                ball.spin,
                contact_point
            )

            # ボールの状態を更新
            ball.velocity = new_velocity
            ball.spin = new_spin

            # ボールを少し離す（めり込み防止）
            ball.position += normal * (self.params.ball_radius * 0.1)

            return True

        return False

    def check_out_of_bounds(self, ball: Ball, table: Table) -> bool:
        """
        ボールが場外に出たかチェック

        Args:
            ball: ボールオブジェクト
            table: テーブルオブジェクト

        Returns:
            out_of_bounds: 場外かどうか
        """
        # 地面に落ちた
        if ball.position[2] < 0:
            return True

        # テーブルから大きく離れた
        distance_from_table = np.linalg.norm(
            ball.position[:2] - np.array([0, 0])
        )
        if distance_from_table > 5.0:  # 5m以上離れたら場外
            return True

        return False

    def compute_collision_impulse(
        self,
        v1: np.ndarray,
        v2: np.ndarray,
        m1: float,
        m2: float,
        normal: np.ndarray,
        restitution: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        2物体の衝突による力積を計算

        Args:
            v1: 物体1の速度
            v2: 物体2の速度
            m1: 物体1の質量
            m2: 物体2の質量
            normal: 衝突面の法線ベクトル
            restitution: 反発係数

        Returns:
            v1_new: 物体1の新しい速度
            v2_new: 物体2の新しい速度
        """
        # 相対速度
        relative_velocity = v1 - v2
        v_rel_normal = np.dot(relative_velocity, normal)

        # 衝突していない（離れる方向）
        if v_rel_normal > 0:
            return v1, v2

        # 力積の大きさ
        impulse_magnitude = -(1 + restitution) * v_rel_normal / (1/m1 + 1/m2)

        # 力積ベクトル
        impulse = impulse_magnitude * normal

        # 新しい速度
        v1_new = v1 + impulse / m1
        v2_new = v2 - impulse / m2

        return v1_new, v2_new


class CollisionDetector:
    """
    衝突検出専用クラス（最適化用）
    """

    @staticmethod
    def sphere_plane_collision(
        sphere_pos: np.ndarray,
        sphere_radius: float,
        plane_point: np.ndarray,
        plane_normal: np.ndarray
    ) -> Tuple[bool, float]:
        """
        球と平面の衝突判定

        Returns:
            is_collision: 衝突したか
            penetration_depth: めり込み深さ
        """
        # 球の中心から平面への距離
        distance = np.dot(sphere_pos - plane_point, plane_normal)

        # 衝突判定
        if distance < sphere_radius:
            penetration_depth = sphere_radius - distance
            return True, penetration_depth

        return False, 0.0

    @staticmethod
    def sphere_sphere_collision(
        pos1: np.ndarray,
        radius1: float,
        pos2: np.ndarray,
        radius2: float
    ) -> Tuple[bool, np.ndarray, float]:
        """
        球と球の衝突判定

        Returns:
            is_collision: 衝突したか
            normal: 衝突面の法線ベクトル
            penetration_depth: めり込み深さ
        """
        diff = pos1 - pos2
        distance = np.linalg.norm(diff)
        sum_radius = radius1 + radius2

        if distance < sum_radius:
            normal = diff / distance if distance > 1e-6 else np.array([1.0, 0.0, 0.0])
            penetration_depth = sum_radius - distance
            return True, normal, penetration_depth

        return False, np.zeros(3), 0.0
