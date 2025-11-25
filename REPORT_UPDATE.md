# Table Tennis Physics Simulation - Progress Report

## Project Status: Active Development

**Date:** 2025-11-25
**Repository:** https://github.com/imuzen127/PhysicsTableTennis

---

## Latest Additions

### 1. Interactive 3D Game (`game.py`) - MAIN

Minecraft-style immersive experience where you ARE inside the game world.

**Features:**
- First-person camera with WASD movement
- Mouse always controls view (FPS-style)
- Real-time ball physics visualization
- In-game command console (press /)
- F3 debug screen with coordinates & rotation
- Ball trajectory trail effect
- Bounce markers on table
- Slow motion mode
- ESC menu with Resume/Quit buttons

**Run:** `python game.py`

**Controls:**
| Key | Action |
|-----|--------|
| WASD | Move around |
| Mouse | Look around (always active) |
| Mouse Side 2 | Up (ascend) |
| Mouse Side 1 | Down (descend) |
| / (slash) | Open command console |
| ESC | Toggle menu / Close chat |
| F1 | Toggle help |
| F3 | Toggle debug screen |

**F3 Debug Screen (Minecraft-style):**
```
Table Tennis Physics Simulation

XYZ: -3.000 / 2.000 / 1.500
Facing: South (-Y)
Rotation: -30.0 / 15.0

FPS: 60
Time Scale: 1.0x

Ball XYZ: 0.000 / 0.000 / 1.000
Ball Vel: 10.00 / 0.00 / 3.00
Ball Speed: 10.44 m/s
Ball Spin: 3000 RPM
```

**In-Game Commands (press / to open):**
```
ball 0 0 1          - Place ball at position
launch 10 0 3       - Launch ball with velocity
spin top 3000       - Set topspin at 3000 RPM
spin back 2500      - Set backspin
serve 15            - Serve at 15 m/s
slow 0.2            - Slow motion (0.2x speed)
tp ball             - Teleport to ball
reset               - Reset simulation
```

**Quick Demo Commands:** `topspin`, `backspin`, `smash`

---

### 2. Console Interface (`console.py`)

Text-based interactive command interface for detailed simulation control.

**Run:** `python console.py`

---

### 3. Simple Runner (`main.py`)

Command-line interface for quick simulations.

**Run:** `python main.py --help`

---

## Physics Engine Features

### Ball Physics
- Accurate Magnus effect (lift coefficient: Cl = 1/(2 + 1/S))
- Air drag with proper coefficient (Cd = 0.45)
- Exponential spin decay
- ITTF standard ball (2.7g, 40mm)

### Racket Physics
- Multiple rubber types (inverted, pimples, anti-spin)
- Rubber-specific friction and spin coefficients
- Blade stiffness modeling
- Forehand/backhand rubber distinction

### Equipment Presets
- Offensive setup (fast blade, high-spin rubbers)
- Defensive setup (soft blade, long pimples)
- All-round setup (balanced)

---

## File Structure

```
PhysicsTableTennis/
├── game.py              # Interactive 3D game (MAIN)
├── console.py           # Command-line interface
├── main.py              # Simple runner
├── interactive.py       # OpenGL viewer (legacy)
├── src/
│   ├── physics/
│   │   ├── parameters.py   # Physics parameters
│   │   ├── ball.py         # Ball dynamics + Magnus
│   │   ├── racket.py       # Racket + rubber physics
│   │   ├── table.py        # Table collision
│   │   └── collision.py    # Collision handler
│   ├── simulation/
│   │   └── engine.py       # Main simulation engine
│   └── visualization/
│       └── viewer.py       # Matplotlib 3D viewer
└── examples/               # Example scripts
```

---

## Git Commits (Latest First)

| Commit | Description |
|--------|-------------|
| 1d40c87 | Add F3 debug screen with coordinates and rotation |
| 63bf5ea | Fix mouse controls: invert look, detect side buttons |
| 0405030 | Fix flicker and update controls |
| f84a6de | Add immersive 3D game with in-game commands |
| 44085e4 | Add console.py command interface |
| f219e9d | Add interactive.py OpenGL viewer |
| 2c31b71 | Add main.py entry point |
| 865fd1d | Enhance physics engine with Magnus effect |

---

## Quick Start

```bash
cd PhysicsTableTennis
pip install numpy matplotlib pygame PyOpenGL
python game.py
```

Press `/` and type `serve` or `topspin`

---

## Requirements

```
numpy
matplotlib
pygame
PyOpenGL
```

Install: `pip install numpy matplotlib pygame PyOpenGL`

---

*Report updated: 2025-11-25*
