"""
Physical parameters for table tennis simulation

All parameters can be adjusted for research purposes.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass
class PhysicsParameters:
    """
    物理パラメータクラス

    全てのパラメータは研究目的で調整可能
    """

    # ボールのパラメータ
    ball_mass: float = 0.0027  # kg (公式: 2.7g)
    ball_radius: float = 0.020  # m (公式: 40mm直径)
    ball_restitution: float = 0.89  # 反発係数 (0-1)

    # テーブルのパラメータ
    table_length: float = 2.74  # m
    table_width: float = 1.525  # m
    table_height: float = 0.76  # m
    table_restitution: float = 0.89  # 反発係数
    table_friction: float = 0.5  # 摩擦係数

    # ラケットのパラメータ
    racket_mass: float = 0.180  # kg (180g)
    racket_restitution: float = 0.85  # 反発係数
    racket_friction: float = 0.7  # 摩擦係数（ラバー表面）

    # 環境パラメータ
    gravity: float = 9.81  # m/s^2
    air_density: float = 1.225  # kg/m^3 (海面レベル、20°C)
    air_drag_coeff: float = 0.45  # 球の空気抵抗係数

    # シミュレーションパラメータ
    dt: float = 0.001  # タイムステップ (秒)
    max_simulation_time: float = 10.0  # 最大シミュレーション時間

    def to_dict(self) -> Dict[str, Any]:
        """パラメータを辞書形式で返す"""
        return {
            'ball_mass': self.ball_mass,
            'ball_radius': self.ball_radius,
            'ball_restitution': self.ball_restitution,
            'table_length': self.table_length,
            'table_width': self.table_width,
            'table_height': self.table_height,
            'table_restitution': self.table_restitution,
            'table_friction': self.table_friction,
            'racket_mass': self.racket_mass,
            'racket_restitution': self.racket_restitution,
            'racket_friction': self.racket_friction,
            'gravity': self.gravity,
            'air_density': self.air_density,
            'air_drag_coeff': self.air_drag_coeff,
            'dt': self.dt,
            'max_simulation_time': self.max_simulation_time,
        }

    @classmethod
    def from_dict(cls, params: Dict[str, Any]) -> 'PhysicsParameters':
        """辞書からパラメータを生成"""
        return cls(**params)

    def update(self, **kwargs):
        """パラメータを更新"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown parameter: {key}")

    def __repr__(self) -> str:
        return f"PhysicsParameters({self.to_dict()})"
