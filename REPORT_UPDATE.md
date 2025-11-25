# Table Tennis Physics Simulation - Progress Report

## Project Status: Active Development

**Date:** 2025-11-25  
**Repository:** https://github.com/imuzen127/PhysicsTableTennis

---

## Latest Additions

### 1. Interactive 3D Game (`game.py`) - NEW!

Minecraft-style immersive experience where you ARE inside the game world.

**Features:**
- First-person camera with WASD movement
- Real-time ball physics visualization
- In-game command console (press T)
- Ball trajectory trail effect
- Bounce markers on table
- Slow motion mode
- Live stats display (speed, spin, height)

**Run:** `python game.py`

**Controls:**
| Key | Action |
|-----|--------|
| WASD | Move around |
| Space/Shift | Up/Down |
| Right Mouse + Drag | Look around |
| T | Open command console |
| P | Pause |
| R | Reset |
| ESC | Quit |

**In-Game Commands:**
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

**Features:**
- Full parameter control
- Step-by-step simulation
- Event logging
- 3D visualization export
- Tutorial mode

---

### 3. Simple Runner (`main.py`)

Command-line interface for quick simulations.

**Run:** `python main.py --help`

**Options:**
- `--setup`: offensive/defensive/allround
- `--speed`: Initial speed (m/s)
- `--spin`: Spin rate (RPM)
- `--show`: Show 3D plot

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
├── game.py              # Interactive 3D game (NEW!)
├── console.py           # Command-line interface (NEW!)
├── main.py              # Simple runner (NEW!)
├── interactive.py       # OpenGL real-time viewer
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

## Quick Start

### Option 1: Play the Game (Recommended)
```bash
cd PhysicsTableTennis
python game.py
```
Then press T and type `serve` or `topspin`

### Option 2: Console Mode
```bash
python console.py
```
Type `tutorial` for guide

### Option 3: Simple Simulation
```bash
python main.py --show
```

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

## Git Commits

1. `865fd1d` - Enhanced physics engine with Magnus effect
2. `2c31b71` - Added main.py entry point
3. `f219e9d` - Added interactive.py OpenGL viewer
4. `44085e4` - Added console.py command interface
5. `f84a6de` - Added game.py immersive 3D game

---

*Report generated: 2025-11-25*
