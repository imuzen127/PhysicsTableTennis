"""
Ball physics module

Implements ball dynamics including:
- Position and velocity
- Spin (angular velocity)
- Magnus effect
- Air resistance
"""

import numpy as np
from typing import Tuple, Optional
from .parameters import PhysicsParameters


class Ball:
    """
    卓球ボールの物理クラス

    ボールの状態（位置、速度、スピン）を管理し、
    物理法則に基づいた運動を計算する
    """

    def __init__(
        self,
        params: PhysicsParameters,
        position: np.ndarray = None,
        velocity: np.ndarray = None,
        spin: np.ndarray = None
    ):
        """
        Args:
            params: 物理パラメータ
            position: 初期位置 [x, y, z] (m)
            velocity: 初期速度 [vx, vy, vz] (m/s)
            spin: 初期角速度 [ωx, ωy, ωz] (rad/s)
        """
        self.params = params

        # 状態ベクトル
        self.position = position if position is not None else np.array([0.0, 0.0, 1.0])
        self.velocity = velocity if velocity is not None else np.array([0.0, 0.0, 0.0])
        self.spin = spin if spin is not None else np.array([0.0, 0.0, 0.0])

        # 軌跡記録
        self.trajectory = [self.position.copy()]
        self.velocity_history = [self.velocity.copy()]
        self.spin_history = [self.spin.copy()]

    def reset(
        self,
        position: np.ndarray = None,
        velocity: np.ndarray = None,
        spin: np.ndarray = None
    ):
        """ボールをリセット"""
        if position is not None:
            self.position = position.copy()
        if velocity is not None:
            self.velocity = velocity.copy()
        if spin is not None:
            self.spin = spin.copy()

        self.trajectory = [self.position.copy()]
        self.velocity_history = [self.velocity.copy()]
        self.spin_history = [self.spin.copy()]

    def compute_forces(self) -> np.ndarray:
        """
        ボールに作用する力を計算

        Returns:
            force: 合力ベクトル [Fx, Fy, Fz] (N)
        """
        # 重力
        gravity_force = np.array([0.0, 0.0, -self.params.ball_mass * self.params.gravity])

        # 空気抵抗 (Fd = -0.5 * ρ * Cd * A * v * |v|)
        speed = np.linalg.norm(self.velocity)
        if speed > 1e-6:
            cross_section = np.pi * self.params.ball_radius ** 2
            drag_force = -0.5 * self.params.air_density * self.params.air_drag_coeff * \
                         cross_section * self.velocity * speed
        else:
            drag_force = np.zeros(3)

        # マグヌス力 (Fm = 0.5 * ρ * Cl * A * (ω × v))
        # Cl は揚力係数、簡略化のため Cd を使用
        magnus_force = np.zeros(3)
        if speed > 1e-6 and np.linalg.norm(self.spin) > 1e-6:
            cross_section = np.pi * self.params.ball_radius ** 2
            # マグヌス力 = 係数 × (角速度 × 速度)
            magnus_coeff = 0.5 * self.params.air_density * self.params.air_drag_coeff * cross_section
            magnus_force = magnus_coeff * np.cross(self.spin, self.velocity)

        # 合力
        total_force = gravity_force + drag_force + magnus_force

        return total_force

    def update(self, dt: Optional[float] = None) -> None:
        """
        ボールの状態を更新（オイラー法）

        Args:
            dt: タイムステップ（Noneの場合はparams.dtを使用）
        """
        if dt is None:
            dt = self.params.dt

        # 力を計算
        force = self.compute_forces()

        # 加速度
        acceleration = force / self.params.ball_mass

        # 速度と位置を更新（オイラー法）
        self.velocity += acceleration * dt
        self.position += self.velocity * dt

        # スピンの減衰（空気抵抗による）
        spin_decay = 0.99  # 簡略化した減衰係数
        self.spin *= spin_decay

        # 軌跡を記録
        self.trajectory.append(self.position.copy())
        self.velocity_history.append(self.velocity.copy())
        self.spin_history.append(self.spin.copy())

    def apply_impulse(self, impulse: np.ndarray, spin_change: np.ndarray = None):
        """
        瞬間的な力積を適用（衝突時など）

        Args:
            impulse: 力積ベクトル [Jx, Jy, Jz] (N·s)
            spin_change: スピンの変化 [Δωx, Δωy, Δωz] (rad/s)
        """
        # 速度変化
        self.velocity += impulse / self.params.ball_mass

        # スピン変化
        if spin_change is not None:
            self.spin += spin_change

    def get_kinetic_energy(self) -> float:
        """運動エネルギーを計算"""
        translational = 0.5 * self.params.ball_mass * np.dot(self.velocity, self.velocity)

        # 回転エネルギー（球の慣性モーメント I = (2/5) * m * r^2）
        moment_of_inertia = (2.0 / 5.0) * self.params.ball_mass * self.params.ball_radius ** 2
        rotational = 0.5 * moment_of_inertia * np.dot(self.spin, self.spin)

        return translational + rotational

    def get_state(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """現在の状態を取得"""
        return self.position.copy(), self.velocity.copy(), self.spin.copy()

    def __repr__(self) -> str:
        return (f"Ball(pos={self.position}, vel={self.velocity}, "
                f"spin={self.spin}, KE={self.get_kinetic_energy():.4f}J)")
