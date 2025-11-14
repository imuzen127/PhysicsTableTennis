"""
基本的なテスト
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from src.physics.parameters import PhysicsParameters
from src.physics.ball import Ball
from src.physics.table import Table
from src.physics.racket import Racket
from src.simulation.engine import TableTennisEngine


def test_parameters():
    """パラメータのテスト"""
    params = PhysicsParameters()

    assert params.ball_mass == 0.0027
    assert params.ball_radius == 0.020
    assert params.gravity == 9.81

    # パラメータ更新
    params.update(ball_mass=0.003)
    assert params.ball_mass == 0.003

    print("[OK] パラメータテスト成功")


def test_ball():
    """ボールのテスト"""
    params = PhysicsParameters()
    ball = Ball(params)

    # 初期位置
    assert np.allclose(ball.position, [0.0, 0.0, 1.0])

    # 重力加速度
    force = ball.compute_forces()
    expected_gravity = np.array([0.0, 0.0, -params.ball_mass * params.gravity])
    assert np.allclose(force[:3], expected_gravity, atol=1e-6)

    print("[OK] ボールテスト成功")


def test_table():
    """テーブルのテスト"""
    params = PhysicsParameters()
    table = Table(params)

    # テーブルの寸法
    assert table.length == 2.74
    assert table.width == 1.525
    assert table.height == 0.76

    # 衝突検出
    ball_on_table = np.array([0.0, 0.0, 0.76 + 0.01])  # テーブル面の少し上
    is_collision, normal = table.check_collision(ball_on_table, 0.020)
    assert is_collision
    assert np.allclose(normal, [0, 0, 1])

    print("[OK] テーブルテスト成功")


def test_racket():
    """ラケットのテスト"""
    params = PhysicsParameters()
    racket = Racket(params, side=1)

    # 初期位置
    assert racket.side == 1
    assert racket.position[0] > 0  # +X側

    print("[OK] ラケットテスト成功")


def test_engine():
    """エンジンのテスト"""
    params = PhysicsParameters(max_simulation_time=1.0)
    engine = TableTennisEngine(params)

    # 初期化
    initial_pos = np.array([0.0, 0.0, 1.0])
    initial_vel = np.array([5.0, 0.0, 0.0])
    engine.reset(ball_position=initial_pos, ball_velocity=initial_vel)

    # 1ステップ実行
    engine.step()
    assert engine.state.step_count == 1
    assert engine.state.time > 0

    print("[OK] エンジンテスト成功")


def test_simulation():
    """シミュレーションのテスト"""
    params = PhysicsParameters(max_simulation_time=0.5)
    engine = TableTennisEngine(params)

    # ボールをテーブルに落とす
    initial_pos = np.array([0.0, 0.0, 1.0])
    initial_vel = np.array([0.0, 0.0, -1.0])
    engine.reset(ball_position=initial_pos, ball_velocity=initial_vel)

    results = engine.run(duration=0.5)

    # テーブルに当たったかチェック
    assert results['collision_count']['table'] > 0

    print("[OK] シミュレーションテスト成功")


if __name__ == "__main__":
    print("=== テスト実行 ===\n")

    test_parameters()
    test_ball()
    test_table()
    test_racket()
    test_engine()
    test_simulation()

    print("\n=== 全てのテストが成功しました！ ===")
