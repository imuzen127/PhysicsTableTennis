"""
Table tennis table module

Implements table geometry and collision detection

Coordinate System: Y-up (X length, Y height, Z width)
"""

import numpy as np
from typing import Tuple, Optional
from .parameters import PhysicsParameters


class Table:
    """
    卓球テーブルクラス

    テーブルの形状と物理特性を管理
    """

    def __init__(self, params: PhysicsParameters):
        """
        Args:
            params: 物理パラメータ
        """
        self.params = params

        # テーブルの寸法
        self.length = params.table_length  # 2.74 m
        self.width = params.table_width    # 1.525 m
        self.height = params.table_height  # 0.76 m

        # テーブルの中心を原点とする
        # X軸: 長さ方向、Y軸: 高さ方向、Z軸: 幅方向
        self.bounds = {
            'x_min': -self.length / 2,
            'x_max': self.length / 2,
            'y': self.height,
            'z_min': -self.width / 2,
            'z_max': self.width / 2
        }

        # ネットの位置（テーブル中央）
        self.net_position = 0.0  # X座標
        self.net_height = 0.1525  # 15.25 cm（テーブル面から）

    def check_collision(
        self,
        position: np.ndarray,
        radius: float
    ) -> Tuple[bool, Optional[np.ndarray]]:
        """
        ボールとテーブルの衝突をチェック

        Args:
            position: ボールの位置 [x, y, z]
            radius: ボールの半径

        Returns:
            is_collision: 衝突したかどうか
            normal: 衝突面の法線ベクトル（衝突していない場合はNone）
        """
        x, y, z = position

        # テーブル面との衝突（上から）
        if (self.bounds['x_min'] <= x <= self.bounds['x_max'] and
            self.bounds['z_min'] <= z <= self.bounds['z_max'] and
            y - radius <= self.bounds['y'] and y > self.bounds['y']):

            # 法線ベクトル（上向き）
            normal = np.array([0.0, 1.0, 0.0])
            return True, normal

        return False, None

    def check_net_collision(
        self,
        position: np.ndarray,
        prev_position: np.ndarray,
        radius: float
    ) -> Tuple[bool, Optional[np.ndarray]]:
        """
        ボールとネットの衝突をチェック

        Args:
            position: 現在のボール位置
            prev_position: 前フレームのボール位置
            radius: ボールの半径

        Returns:
            is_collision: 衝突したかどうか
            normal: 衝突面の法線ベクトル
        """
        x, y, z = position
        x_prev, y_prev, z_prev = prev_position

        # ネットを横切ったかチェック
        crossed_net = (x_prev < self.net_position <= x) or (x > self.net_position >= x_prev)

        if crossed_net:
            # ネットの高さより下で、テーブル幅内
            net_top_y = self.bounds['y'] + self.net_height
            if (y <= net_top_y and
                self.bounds['z_min'] <= z <= self.bounds['z_max']):

                # 法線ベクトル（ボールの進行方向の反対）
                direction = np.sign(x - x_prev)
                normal = np.array([-direction, 0.0, 0.0])
                return True, normal

        return False, None

    def is_on_table(self, position: np.ndarray) -> bool:
        """
        ボールがテーブル上にあるかチェック

        Args:
            position: ボールの位置

        Returns:
            on_table: テーブル上にあるかどうか
        """
        x, y, z = position

        return (self.bounds['x_min'] <= x <= self.bounds['x_max'] and
                self.bounds['z_min'] <= z <= self.bounds['z_max'] and
                abs(y - self.bounds['y']) < 0.1)

    def get_corner_positions(self) -> np.ndarray:
        """テーブルの4隅の座標を取得"""
        corners = np.array([
            [self.bounds['x_min'], self.bounds['y'], self.bounds['z_min']],
            [self.bounds['x_max'], self.bounds['y'], self.bounds['z_min']],
            [self.bounds['x_max'], self.bounds['y'], self.bounds['z_max']],
            [self.bounds['x_min'], self.bounds['y'], self.bounds['z_max']],
        ])
        return corners

    def get_net_positions(self) -> Tuple[np.ndarray, np.ndarray]:
        """ネットの両端の座標を取得"""
        net_top = self.bounds['y'] + self.net_height
        start = np.array([self.net_position, self.bounds['y'], self.bounds['z_min']])
        end = np.array([self.net_position, net_top, self.bounds['z_max']])
        return start, end

    def __repr__(self) -> str:
        return (f"Table(length={self.length}m, width={self.width}m, "
                f"height={self.height}m)")
