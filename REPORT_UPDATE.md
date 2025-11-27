# Table Tennis Physics Simulation - Progress Report

## Project Status: Active Development

**Date:** 2025-11-27
**Repository:** https://github.com/imuzen127/PhysicsTableTennis

---

## Latest Additions

### 1. Y-up Coordinate System Unification (NEW!)

座標系を統一しました。全システムでY-up（XZ平面が水平、Yが高さ）を使用。

**座標系:**
| 軸 | 意味 |
|----|------|
| X | 水平（テーブル長さ方向） |
| Y | 高さ（上方向） |
| Z | 水平（テーブル幅方向） |

**コマンド例:**
```
ball -1 1 0           # x=-1, height=1, z=0
launch 12 3 0         # vx=12, vy(上)=3, vz=0
summon ball ~ ~1 ~    # プレイヤー位置の1m上
```

---

### 2. Ball Orientation Display - F3+B (NEW!)

ボールの向き（回転状態）を可視化する機能。

**使用方法:**
1. `F3`キーを押しながら`B`キーを押してトグル
2. 黄色→赤のグラデーション線がボールの向きを表示
3. 再度F3+Bで非表示

**コマンド例:**
```
summon ball ~ ~1 ~ {rotation:{angle:1.570, axis:[1,0,0]}}
```
- X軸周りに90度（π/2ラジアン）回転したボールを召喚
- デフォルトの正面方向は+Z方向

---

### 3. Minecraft-Style Command System

Full NBT data support with Minecraft-like syntax.

**NBT Parameters:**
| Parameter | Description | Default | Unit |
|-----------|-------------|---------|------|
| velocity | Velocity | 0 | m/s |
| acceleration | Acceleration | 0 | m/s^2 |
| mass | Mass | 2.7 (ball), 180 (racket) | g |
| radius | Radius | 20 | mm |
| coefficient | Friction | [0.8, 0.8] | [red, black] |
| rotation | Orientation | {angle:0, axis:[0,1,0]} | rad |
| spin | Spin | {rpm:0, axis:[0,1,0]} | RPM |

**Coordinate Systems:**
- `~ ~ ~` - Relative to player position (default if omitted)
- `^ ^ ^` - Local coordinates (based on player facing direction)

**Commands:**
```
summon ball                         - Spawn ball at player position
summon ball ~ ~1 ~                  - Spawn ball 1m above player
summon ball {velocity:{rotation:@s, speed:15}}  - With velocity
summon ball {rotation:{angle:1.57, axis:[1,0,0]}} - With orientation
summon racket {mass:180, coefficient:[0.9,0.8]}

execute rotate as @s run summon ball ^0 ^0 ^2
execute at @n[type=ball] run summon ball ~ ~1 ~

start                               - Start simulation
stop                                - Stop simulation
kill @e                             - Remove all entities
kill @e[type=ball]                  - Remove all balls

gamemode gravity 9.8                - Change gravity
```

---

### 4. Interactive 3D Game (`game.py`) - MAIN

Minecraft-style immersive experience.

**Controls:**
| Key | Action |
|-----|--------|
| WASD | Move around |
| Mouse | Look around |
| Mouse Button 6 | Up (ascend) |
| Mouse Button 7 | Down (descend) |
| Mouse Button 4 | Accelerate |
| Mouse Button 5 | Decelerate |
| / | Open command console |
| Arrow Keys | Move cursor in command |
| ESC | Menu |
| F1 | Toggle help |
| F3 | Debug screen |
| F3+B | Ball orientation display |

---

## File Structure

```
PhysicsTableTennis/
├── game.py              # Interactive 3D game (MAIN)
├── src/
│   ├── command/         # Command system
│   │   ├── parser.py    # NBT & coordinate parser
│   │   └── objects.py   # Entity management
│   ├── physics/
│   │   ├── parameters.py
│   │   ├── ball.py      # Y-up coordinate system
│   │   ├── table.py     # Y-up coordinate system
│   │   ├── racket.py    # Y-up coordinate system
│   │   └── collision.py # Y-up coordinate system
│   └── simulation/
│       └── engine.py
```

---

## Quick Start

```bash
cd PhysicsTableTennis
pip install numpy matplotlib pygame PyOpenGL
python game.py
```

Press `/` and type:
- `summon ball` - Spawn ball
- `start` - Begin simulation
- `gamemode gravity 1.62` - Moon gravity!
- Press `F3+B` to see ball orientation

---

*Report updated: 2025-11-27*
