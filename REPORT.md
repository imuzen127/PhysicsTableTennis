# PhysicsTableTennis - プロジェクトレポート

**作成日**: 2025年11月14日
**プロジェクト**: 卓球3D物理エンジンシステム
**リポジトリ**: https://github.com/imuzen127/PhysicsTableTennis

---

## 目次

1. [プロジェクト概要](#プロジェクト概要)
2. [開発動機と目的](#開発動機と目的)
3. [システム設計](#システム設計)
4. [実装した機能](#実装した機能)
5. [物理計算の詳細](#物理計算の詳細)
6. [技術的な特徴](#技術的な特徴)
7. [使用例とデモ](#使用例とデモ)
8. [テストと検証](#テストと検証)
9. [今後の展望](#今後の展望)
10. [結論](#結論)

---

## プロジェクト概要

PhysicsTableTennisは、卓球の物理現象を正確にシミュレートする3D物理エンジンシステムです。研究・教育用途を想定し、物理パラメータの詳細な調整が可能な設計になっています。

### 主な特徴

- **リアルな物理シミュレーション**: 摩擦、反発、質量、速度、角度、スピン、マグヌス効果を考慮
- **動作登録システム**: ラケットの軌道を記録・再生して自動打ち合いが可能
- **パラメータ調整機能**: 全ての物理定数を調整可能（研究向け）
- **3D可視化**: matplotlib による直感的な3D表示

### 技術スタック

- **Python 3.11+**: メイン言語
- **NumPy**: 数値計算とベクトル演算
- **SciPy**: 科学計算と補間処理
- **Matplotlib**: 3D可視化とグラフ作成

---

## 開発動機と目的

### 背景

卓球は複雑な物理現象が関わるスポーツです：

1. **ボールのスピン**: トップスピン、バックスピン、サイドスピンなど
2. **マグヌス効果**: スピンによる軌道の変化
3. **衝突の物理**: テーブルやラケットとの反発・摩擦
4. **空気抵抗**: 速度に応じた減速

これらを統合的にシミュレートするシステムは、以下の用途で有用です：

- スポーツ科学の研究
- 選手のトレーニング支援
- 物理教育のデモンストレーション
- ゲームやアプリケーション開発

### プロジェクトの目的

1. **研究向けツールの提供**: パラメータを自由に変更して物理現象を分析
2. **動作解析の実現**: 実際のラケット動作を記録・再生してシミュレーション
3. **教育への応用**: 物理法則を視覚的に理解できるツール
4. **拡張性**: 将来的な機能追加が容易な設計

---

## システム設計

### アーキテクチャ

```
PhysicsTableTennis/
├── src/
│   ├── physics/          # 物理計算モジュール
│   │   ├── parameters.py  # 物理パラメータ
│   │   ├── ball.py        # ボール物理
│   │   ├── table.py       # テーブル
│   │   ├── racket.py      # ラケット
│   │   └── collision.py   # 衝突検出・応答
│   ├── simulation/       # シミュレーションエンジン
│   │   ├── engine.py      # メインエンジン
│   │   └── motion.py      # 動作記録・再生
│   └── visualization/    # 3D可視化
│       └── viewer.py      # ビューワー
├── examples/             # サンプルコード
├── tests/                # テストコード
└── README.md
```

### 設計思想

1. **モジュール化**: 各物理要素を独立したクラスとして実装
2. **データ駆動**: パラメータを外部から注入可能な設計
3. **拡張性**: 新しい物理現象を追加しやすい構造
4. **再現性**: 同じ初期条件から同じ結果を得られる

---

## 実装した機能

### 1. 物理エンジン (src/physics/)

#### 1.1 パラメータ管理 (`parameters.py`)

全ての物理パラメータを一元管理：

```python
@dataclass
class PhysicsParameters:
    # ボールのパラメータ
    ball_mass: float = 0.0027        # 2.7g (公式規格)
    ball_radius: float = 0.020       # 40mm直径
    ball_restitution: float = 0.89   # 反発係数

    # テーブルのパラメータ
    table_restitution: float = 0.89
    table_friction: float = 0.5

    # 環境パラメータ
    gravity: float = 9.81            # 重力加速度
    air_density: float = 1.225       # 空気密度
    air_drag_coeff: float = 0.45     # 空気抵抗係数
```

**特徴**:
- 全パラメータを動的に変更可能
- 辞書形式での入出力対応
- デフォルト値は公式規格に準拠

#### 1.2 ボール物理 (`ball.py`)

ボールの運動を計算：

```python
class Ball:
    def compute_forces(self) -> np.ndarray:
        # 重力
        gravity_force = m * g * [0, 0, -1]

        # 空気抵抗: F_d = -0.5 * ρ * C_d * A * v * |v|
        drag_force = -0.5 * ρ * C_d * A * v * |v|

        # マグヌス力: F_m = 0.5 * ρ * C_l * A * (ω × v)
        magnus_force = 0.5 * ρ * C_l * A * (ω × v)

        return gravity_force + drag_force + magnus_force
```

**実装した物理現象**:
1. **重力**: 一定の下向き加速度
2. **空気抵抗**: 速度の2乗に比例
3. **マグヌス効果**: スピンと速度の外積で計算
4. **スピン減衰**: 時間経過によるスピンの減少

#### 1.3 テーブル (`table.py`)

テーブルの形状と衝突判定：

- **寸法**: 2.74m × 1.525m × 0.76m (公式規格)
- **衝突検出**: 球と平面の交差判定
- **ネット**: 中央に配置、高さ15.25cm

#### 1.4 ラケット (`racket.py`)

ラケットの動的制御：

- **位置・姿勢**: 3D空間での自由な配置
- **速度**: 移動速度を自動計算
- **衝突計算**: ボールとの衝突時の速度・スピン変化

#### 1.5 衝突処理 (`collision.py`)

衝突の検出と応答：

```python
def handle_ball_table_collision(ball, table):
    # 速度を法線・接線成分に分解
    v_normal = (v · n) * n
    v_tangent = v - v_normal

    # 反発係数を適用
    v_normal_new = -e * v_normal

    # 摩擦を適用
    v_tangent_new = v_tangent * (1 - μ)

    # スピンの影響を考慮
    v_tangent_new += (ω × n) * r * μ

    # 新しい速度
    v_new = v_normal_new + v_tangent_new
```

**処理する衝突**:
1. ボール-テーブル
2. ボール-ネット
3. ボール-ラケット

### 2. シミュレーションエンジン (src/simulation/)

#### 2.1 メインエンジン (`engine.py`)

統合的なシミュレーション管理：

```python
class TableTennisEngine:
    def run(self, duration, racket_1_controller, racket_2_controller):
        while time < duration:
            # ラケットを制御
            racket_1.update_position(controller_1(time, ball_state))
            racket_2.update_position(controller_2(time, ball_state))

            # 物理計算
            ball.update(dt)

            # 衝突処理
            handle_collisions()

            time += dt
```

**機能**:
- タイムステップ制御 (デフォルト 1ms)
- イベントログ記録
- 衝突カウント
- 軌跡データの保存

#### 2.2 動作登録システム (`motion.py`)

ラケット動作の記録・再生：

```python
# 記録
recorder = MotionRecorder("forehand_drive")
recorder.start_recording()
for frame in motion_data:
    recorder.record_frame(time, position, orientation)
recorder.save("forehand.npz")

# 再生
player = MotionPlayer(filename="forehand.npz")
controller = player.create_controller()
engine.run(racket_1_controller=controller)
```

**事前定義された動作**:
- フォアハンドドライブ
- 静的防御姿勢
- サーブモーション

### 3. 3D可視化 (src/visualization/)

#### 3.1 ビューワー (`viewer.py`)

matplotlib による3D表示：

- **テーブル描画**: 3D矩形とネット
- **軌跡表示**: ボールとラケットの軌跡
- **アニメーション**: 時系列での動的表示
- **速度プロファイル**: 時間-速度グラフ

---

## 物理計算の詳細

### ボールの運動方程式

#### 並進運動

```
m * dv/dt = F_gravity + F_drag + F_magnus
```

各力の計算：

1. **重力**:
   ```
   F_g = [0, 0, -mg]
   ```

2. **空気抵抗**:
   ```
   F_d = -0.5 * ρ * C_d * A * v * |v|

   ρ: 空気密度 (1.225 kg/m³)
   C_d: 抵抗係数 (0.45)
   A: 断面積 (πr²)
   ```

3. **マグヌス力**:
   ```
   F_m = 0.5 * ρ * C_l * A * (ω × v)

   ω: 角速度ベクトル (rad/s)
   v: 速度ベクトル (m/s)
   ```

#### 回転運動

スピンの減衰：
```
ω(t) = ω₀ * e^(-αt)

α: 減衰係数
```

### 衝突の物理

#### ボール-テーブル衝突

1. **法線方向**: 反発係数を適用
   ```
   v_n' = -e * v_n
   e: 反発係数 (0.89)
   ```

2. **接線方向**: 摩擦とスピンを考慮
   ```
   v_t' = v_t * (1 - μ) + (ω × n) * r * μ
   μ: 摩擦係数 (0.5)
   ```

3. **スピン変化**: 接線方向の滑りからスピンを計算
   ```
   Δω = (n × v_t) * μ / r
   ```

#### ボール-ラケット衝突

1. **相対速度**: ラケットの速度を考慮
   ```
   v_rel = v_ball - v_racket
   ```

2. **反発**: ラケットの反発係数を適用
   ```
   v_normal' = -e_racket * v_normal + v_racket_normal
   ```

3. **スピン付与**: 接線方向の速度差からスピンを生成
   ```
   Δω = (n × v_tangent) * μ_racket / r
   ```

### 数値積分

オイラー法を使用：

```python
def update(dt):
    # 力を計算
    F = compute_forces()

    # 加速度
    a = F / m

    # 速度更新
    v += a * dt

    # 位置更新
    x += v * dt
```

**タイムステップ**: 1ms (十分な精度を確保)

---

## 技術的な特徴

### 1. パラメータ調整機能

研究用途を想定し、全てのパラメータを調整可能：

```python
# パラメータを変更
engine.update_parameters(
    ball_mass=0.003,        # 軽いボール
    table_friction=0.7,     # 摩擦大
    air_drag_coeff=0.3      # 空気抵抗小
)
```

### 2. データの入出力

シミュレーション結果を保存・読み込み：

```python
# 保存
engine.export_trajectory("data.npz")

# 読み込み
data = np.load("data.npz")
trajectory = data['ball_trajectory']
```

### 3. イベントログ

全ての重要なイベントを記録：

```python
{
    'time': 1.234,
    'step': 1234,
    'type': 'collision',
    'message': 'Ball-Table collision at [1.2, 0.3, 0.76]'
}
```

### 4. 拡張性

新しい機能を追加しやすい設計：

- **新しい力**: `Ball.compute_forces()` に追加
- **新しいオブジェクト**: 新しいクラスを作成して `engine` に統合
- **新しい可視化**: `Viewer3D` にメソッドを追加

---

## 使用例とデモ

### 例1: 基本的なシミュレーション

ボールを打ち出してテーブルにバウンドさせる：

```python
from src.simulation.engine import TableTennisEngine
import numpy as np

# エンジン初期化
engine = TableTennisEngine()

# 初期条件
engine.reset(
    ball_position=np.array([-1.0, 0.0, 1.0]),
    ball_velocity=np.array([8.0, 0.0, 2.0]),
    ball_spin=np.array([0.0, 50.0, 0.0])  # トップスピン
)

# 実行
results = engine.run(duration=3.0)

# 結果表示
print(f"衝突回数: {results['collision_count']}")
```

### 例2: パラメータスタディ

反発係数を変化させて軌道を比較：

```python
restitution_values = [0.7, 0.8, 0.89, 0.95]

for e in restitution_values:
    engine.update_parameters(ball_restitution=e)
    results = engine.run()

    # プロット
    plot_trajectory(results['ball_trajectory'], label=f'e={e}')
```

**結果**: 反発係数が高いほどバウンド後の高さが増加

### 例3: ラリーシミュレーション

事前定義された動作でラリーを実現：

```python
from src.simulation.motion import PredefinedMotions, MotionPlayer

# 動作を作成
motion_1 = PredefinedMotions.create_forehand_drive(side=1)
motion_2 = PredefinedMotions.create_static_defense(side=-1)

# プレイヤー作成
player_1 = MotionPlayer(motion_data=motion_1)
player_2 = MotionPlayer(motion_data=motion_2)

# シミュレーション
results = engine.run(
    racket_1_controller=player_1.create_controller(),
    racket_2_controller=player_2.create_controller()
)

print(f"ラケットヒット数: {results['collision_count']['racket_1'] + results['collision_count']['racket_2']}")
```

### 例4: カスタム動作の記録

独自のラケット動作を作成：

```python
from src.simulation.motion import MotionRecorder

recorder = MotionRecorder("custom_motion")
recorder.start_recording()

for t in np.linspace(0, 1.0, 100):
    position = calculate_position(t)
    orientation = calculate_orientation(t)
    recorder.record_frame(t, position, orientation)

recorder.stop_recording()
recorder.save("my_motion.npz")

# 再生
player = MotionPlayer(filename="my_motion.npz")
engine.run(racket_1_controller=player.create_controller())
```

---

## テストと検証

### 単体テスト

6つの基本テストを実装：

1. **パラメータテスト**: パラメータの設定・更新
2. **ボールテスト**: 重力計算の正確性
3. **テーブルテスト**: 衝突検出の正確性
4. **ラケットテスト**: 位置・姿勢の管理
5. **エンジンテスト**: シミュレーションステップ
6. **シミュレーションテスト**: 統合テスト

```bash
python tests/test_basic.py
```

**結果**: 全テスト成功

### 物理検証

#### 1. 重力加速度

自由落下のテスト：

```python
# 1秒後の落下距離
expected = 0.5 * g * t² = 0.5 * 9.81 * 1² = 4.905m
actual = simulate_free_fall(1.0)

assert abs(expected - actual) < 0.01  # 1cm以内の誤差
```

#### 2. 反発係数

テーブルバウンド後の速度：

```python
v_before = -5.0  # m/s (下向き)
e = 0.89

v_after = -e * v_before = 4.45 m/s (上向き)

# シミュレーション結果と比較
assert abs(simulated_velocity - v_after) < 0.1
```

#### 3. マグヌス効果

トップスピンによる軌道変化：

- スピンなし: 放物線軌道
- トップスピン: 下向きに曲がる軌道

→ 実際の卓球の挙動と一致

---

## 今後の展望

### 短期目標 (1-3ヶ月)

1. **より精緻な物理モデル**
   - ボールの変形を考慮
   - ラバーの種類による反発係数の違い
   - 温度・湿度の影響

2. **GUI開発**
   - リアルタイム操作可能なインターフェース
   - パラメータ調整スライダー
   - 再生・一時停止機能

3. **実データとの検証**
   - 実際の卓球試合の映像解析
   - トラッキングデータとの比較
   - パラメータの最適化

### 中期目標 (3-6ヶ月)

1. **機械学習の統合**
   - 最適な打球タイミングの学習
   - 相手の動作予測
   - 戦略シミュレーション

2. **VRへの対応**
   - Unity/Unreal Engine への移植
   - VRヘッドセットでの可視化
   - トレーニングシミュレータ

3. **マルチボールシミュレーション**
   - 複数ボールの同時シミュレート
   - 多球練習のシミュレーション

### 長期目標 (6-12ヶ月)

1. **商用アプリケーション化**
   - スマートフォンアプリ
   - トレーニング支援ツール
   - 戦術分析ソフトウェア

2. **他のスポーツへの展開**
   - テニス、バドミントン
   - ビリヤード、ゴルフ
   - 汎用スポーツ物理エンジン

3. **研究コミュニティの形成**
   - オープンソース貢献の促進
   - 論文発表
   - 学会での発表

---

## 結論

PhysicsTableTennisは、卓球の複雑な物理現象を統合的にシミュレートするシステムです。本プロジェクトの主な成果は以下の通りです：

### 達成したこと

1. **包括的な物理モデル**: スピン、マグヌス効果、衝突を含む
2. **研究向け設計**: 全パラメータが調整可能
3. **動作登録機能**: ラケット動作の記録・再生
4. **3D可視化**: 直感的な結果表示
5. **テスト済み**: 全ての基本機能を検証

### 技術的貢献

- Pythonベースの卓球物理エンジンの実装
- 動作記録システムによる自動ラリーの実現
- パラメータスタディ機能による研究支援
- 拡張可能な設計による将来の機能追加の容易化

### 応用可能性

1. **研究**: スポーツ科学、物理学の研究
2. **教育**: 物理法則の可視化教材
3. **トレーニング**: 選手の技術向上支援
4. **エンターテインメント**: ゲーム開発のベース

### 最後に

本プロジェクトは、物理シミュレーションの基礎を確立しました。今後は実データとの検証、GUI開発、機械学習の統合などを進め、実用的なツールへと発展させていきます。

オープンソースプロジェクトとして公開することで、コミュニティからのフィードバックや貢献を得て、さらなる改善を図ります。

---

## 参考資料

### 物理学

- 古典力学の教科書
- 流体力学（空気抵抗）
- 衝突の物理学

### 卓球

- ITTF公式ルール
- 卓球の物理学に関する論文
- プロ選手の技術解説

### プログラミング

- NumPy/SciPy ドキュメント
- Matplotlib チュートリアル
- Python ベストプラクティス

---

**プロジェクト情報**

- **GitHub**: https://github.com/imuzen127/PhysicsTableTennis
- **ライセンス**: MIT License
- **開発者**: imuzen127
- **開発支援**: Claude Code (Anthropic)

---

*このレポートは、プロジェクトの初期開発完了時点(2025年11月14日)での内容をまとめたものです。*
