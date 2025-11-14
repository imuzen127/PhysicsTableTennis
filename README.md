# PhysicsTableTennis

卓球の3D物理エンジンシステム - 研究・教育向け

## 概要

このプロジェクトは、卓球の物理を正確にシミュレートする3D物理エンジンシステムです。
研究用途を想定し、物理パラメータの詳細な調整が可能です。

## 主な機能

- **リアルな物理シミュレーション**
  - 摩擦係数、反発係数、質量、速度、角度などを考慮
  - ボールのスピン（回転）とマグヌス効果
  - 空気抵抗の計算

- **動作登録システム**
  - ラケットの軌道を記録・再生
  - ボールの打ち合いシミュレーション

- **パラメータ調整**
  - 全ての物理パラメータを調整可能
  - シミュレーション結果の比較分析

- **3D可視化**
  - リアルタイム3D表示
  - 軌道の可視化

## インストール

```bash
pip install -r requirements.txt
```

## 使用方法

```python
from src.simulation.engine import TableTennisEngine
from src.physics.parameters import PhysicsParameters

# パラメータ設定
params = PhysicsParameters(
    ball_mass=0.0027,  # kg
    ball_radius=0.020,  # m
    restitution_coeff=0.89,
    friction_coeff=0.5
)

# エンジン初期化
engine = TableTennisEngine(params)

# シミュレーション実行
engine.simulate()
```

## プロジェクト構造

```
PhysicsTableTennis/
├── src/
│   ├── physics/          # 物理計算モジュール
│   ├── simulation/       # シミュレーションエンジン
│   └── visualization/    # 3D可視化
├── tests/                # テストコード
└── examples/             # サンプルコード
```

## ライセンス

MIT License
