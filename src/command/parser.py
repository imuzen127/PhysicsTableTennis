"""
Minecraft-style command parser with NBT data support.

Coordinate Systems:
- World: X (horizontal), Y (height/up), Z (horizontal)
- ~ ~ ~ : Relative to player position (default if omitted)
- ^ ^ ^ : Local coordinates (left/up/forward based on player facing)

NBT Parameters:
| Parameter    | Description      | Default              | Unit    |
|-------------|------------------|----------------------|---------|
| velocity    | Velocity         | 0                    | m/s     |
| acceleration| Acceleration     | 0                    | m/s^2   |
| mass        | Mass             | 2.7 (ball), 180 (racket) | g   |
| radius      | Radius           | 20                   | mm      |
| coefficient | Friction coeff   | [0.8, 0.8]           | [red, black] |
| rotation    | Orientation      | {angle:0, axis:[0,1,0]} | rad  |
| spin        | Spin             | {rpm:0, axis:[0,1,0]} | RPM    |

Commands:
- summon ball [x y z] [nbt]  (default: ~ ~ ~)
- summon racket [x y z] [nbt]
- execute rotate as @s run <cmd>
- execute at @n[type=ball] run <cmd>
- start / stop
- kill <selector>
- gamemode gravity <value>
"""

import re
import math
import numpy as np
from typing import Dict, List, Any, Tuple, Optional, Union


class NBTParser:
    """Parse Minecraft-style NBT data: {key:value, ...}"""

    def __init__(self):
        self.pos = 0
        self.text = ""

    def parse(self, text: str) -> Dict[str, Any]:
        """Parse NBT string to dictionary"""
        self.text = text.strip()
        self.pos = 0

        if not self.text or self.text[0] != '{':
            return {}

        return self._parse_compound()

    def _skip_whitespace(self):
        while self.pos < len(self.text) and self.text[self.pos] in ' \t\n':
            self.pos += 1

    def _parse_compound(self) -> Dict[str, Any]:
        """Parse {key:value, ...}"""
        result = {}

        if self.text[self.pos] != '{':
            raise ValueError(f"Expected '{{' at position {self.pos}")
        self.pos += 1
        self._skip_whitespace()

        while self.pos < len(self.text) and self.text[self.pos] != '}':
            # Parse key
            key = self._parse_key()
            self._skip_whitespace()

            if self.pos >= len(self.text) or self.text[self.pos] != ':':
                raise ValueError(f"Expected ':' after key '{key}'")
            self.pos += 1
            self._skip_whitespace()

            # Parse value
            value = self._parse_value()
            result[key] = value

            self._skip_whitespace()
            if self.pos < len(self.text) and self.text[self.pos] == ',':
                self.pos += 1
                self._skip_whitespace()

        if self.pos >= len(self.text) or self.text[self.pos] != '}':
            raise ValueError("Expected '}' at end of compound")
        self.pos += 1

        return result

    def _parse_key(self) -> str:
        """Parse key name (alphanumeric + underscore)"""
        start = self.pos
        while self.pos < len(self.text) and (self.text[self.pos].isalnum() or self.text[self.pos] == '_'):
            self.pos += 1
        return self.text[start:self.pos]

    def _parse_value(self) -> Any:
        """Parse any value type"""
        self._skip_whitespace()

        if self.pos >= len(self.text):
            raise ValueError("Unexpected end of input")

        char = self.text[self.pos]

        if char == '{':
            return self._parse_compound()
        elif char == '[':
            return self._parse_array()
        elif char == '"' or char == "'":
            return self._parse_string()
        elif char == '@':
            return self._parse_selector()
        else:
            return self._parse_primitive()

    def _parse_array(self) -> List[Any]:
        """Parse [value, value, ...]"""
        result = []

        self.pos += 1  # skip '['
        self._skip_whitespace()

        while self.pos < len(self.text) and self.text[self.pos] != ']':
            value = self._parse_value()
            result.append(value)

            self._skip_whitespace()
            if self.pos < len(self.text) and self.text[self.pos] == ',':
                self.pos += 1
                self._skip_whitespace()

        if self.pos >= len(self.text) or self.text[self.pos] != ']':
            raise ValueError("Expected ']' at end of array")
        self.pos += 1

        return result

    def _parse_string(self) -> str:
        """Parse "string" or 'string'"""
        quote = self.text[self.pos]
        self.pos += 1
        start = self.pos

        while self.pos < len(self.text) and self.text[self.pos] != quote:
            if self.text[self.pos] == '\\':
                self.pos += 1
            self.pos += 1

        result = self.text[start:self.pos]
        self.pos += 1  # skip closing quote
        return result

    def _parse_selector(self) -> str:
        """Parse @s, @e, @n[type=ball], etc."""
        start = self.pos
        self.pos += 1  # skip '@'

        while self.pos < len(self.text) and (self.text[self.pos].isalnum() or self.text[self.pos] in '[].,=!'):
            if self.text[self.pos] == '[':
                # Skip selector arguments
                while self.pos < len(self.text) and self.text[self.pos] != ']':
                    self.pos += 1
                if self.pos < len(self.text):
                    self.pos += 1
            else:
                self.pos += 1

        return self.text[start:self.pos]

    def _parse_primitive(self) -> Union[float, int, bool, str]:
        """Parse number, boolean, or unquoted string"""
        start = self.pos

        # Check for negative number
        if self.text[self.pos] == '-':
            self.pos += 1

        # Collect characters
        while self.pos < len(self.text) and self.text[self.pos] not in ' \t\n,}]:':
            self.pos += 1

        value = self.text[start:self.pos]

        # Try to parse as number
        if value.lower() == 'true':
            return True
        elif value.lower() == 'false':
            return False
        else:
            try:
                if '.' in value:
                    return float(value)
                else:
                    return int(value)
            except ValueError:
                return value


class CoordinateParser:
    """Parse Minecraft-style coordinates with ~ and ^ notation"""

    def __init__(self, player_pos: np.ndarray, player_yaw: float, player_pitch: float):
        """
        Args:
            player_pos: Player position [x, y, z] (Y is up)
            player_yaw: Player yaw in degrees (0 = +X, 90 = +Z)
            player_pitch: Player pitch in degrees (positive = looking up)
        """
        self.player_pos = player_pos
        self.player_yaw = player_yaw
        self.player_pitch = player_pitch

    def parse(self, x_str: str, y_str: str, z_str: str) -> np.ndarray:
        """
        Parse coordinate strings to world position.

        Args:
            x_str, y_str, z_str: Coordinate strings (number, ~, ~N, ^, ^N)

        Returns:
            World position [x, y, z]
        """
        # Check coordinate type
        if x_str.startswith('^') or y_str.startswith('^') or z_str.startswith('^'):
            # Local coordinates (all must be ^)
            return self._parse_local(x_str, y_str, z_str)
        elif x_str.startswith('~') or y_str.startswith('~') or z_str.startswith('~'):
            # Relative coordinates
            return self._parse_relative(x_str, y_str, z_str)
        else:
            # Absolute coordinates
            return np.array([float(x_str), float(y_str), float(z_str)])

    def _parse_relative(self, x_str: str, y_str: str, z_str: str) -> np.ndarray:
        """Parse ~ coordinates (relative to player)"""
        def parse_one(s: str, base: float) -> float:
            if s == '~':
                return base
            elif s.startswith('~'):
                return base + float(s[1:])
            else:
                return float(s)

        return np.array([
            parse_one(x_str, self.player_pos[0]),
            parse_one(y_str, self.player_pos[1]),
            parse_one(z_str, self.player_pos[2])
        ])

    def _parse_local(self, x_str: str, y_str: str, z_str: str) -> np.ndarray:
        """Parse ^ coordinates (local to player facing)"""
        def parse_one(s: str) -> float:
            if s == '^':
                return 0.0
            elif s.startswith('^'):
                return float(s[1:])
            else:
                return float(s)

        # Local offsets: left, up, forward
        left = parse_one(x_str)
        up = parse_one(y_str)
        forward = parse_one(z_str)

        # Convert player angles to radians
        yaw_rad = math.radians(self.player_yaw)
        pitch_rad = math.radians(self.player_pitch)

        # Calculate direction vectors
        # Forward vector (in XZ plane, Y is up)
        forward_vec = np.array([
            math.cos(yaw_rad) * math.cos(pitch_rad),
            math.sin(pitch_rad),
            math.sin(yaw_rad) * math.cos(pitch_rad)
        ])

        # Up vector (world up adjusted for pitch)
        up_vec = np.array([
            -math.cos(yaw_rad) * math.sin(pitch_rad),
            math.cos(pitch_rad),
            -math.sin(yaw_rad) * math.sin(pitch_rad)
        ])

        # Left vector (perpendicular to forward and up)
        left_vec = np.cross(up_vec, forward_vec)
        if np.linalg.norm(left_vec) > 0:
            left_vec = left_vec / np.linalg.norm(left_vec)

        # Calculate world position
        return self.player_pos + left * left_vec + up * up_vec + forward * forward_vec


class VelocityParser:
    """Parse velocity/acceleration with rotation + speed format"""

    def __init__(self, player_yaw: float, player_pitch: float):
        self.player_yaw = player_yaw
        self.player_pitch = player_pitch

    def parse(self, data: Any) -> np.ndarray:
        """
        Parse velocity data to vector.

        Formats:
        - {rotation:[yaw, pitch], speed:N}
        - {rotation:@s, speed:N}
        - [vx, vy, vz] (direct vector)
        - N (scalar, uses player direction)
        """
        if isinstance(data, (int, float)):
            # Scalar value - use player direction
            speed = float(data)
            yaw_rad = math.radians(self.player_yaw)
            pitch_rad = math.radians(self.player_pitch)
            return np.array([
                speed * math.cos(yaw_rad) * math.cos(pitch_rad),
                speed * math.sin(pitch_rad),
                speed * math.sin(yaw_rad) * math.cos(pitch_rad)
            ])

        if isinstance(data, list):
            return np.array(data, dtype=float)

        if not isinstance(data, dict):
            raise ValueError(f"Invalid velocity format: {data}")

        speed = float(data.get('speed', 0))
        rotation = data.get('rotation', [0, 0])

        # Handle @s selector
        if rotation == '@s':
            yaw = self.player_yaw
            pitch = self.player_pitch
        elif isinstance(rotation, list) and len(rotation) >= 2:
            # Check if elements are @s
            yaw = self.player_yaw if rotation[0] == '@s' else float(rotation[0])
            pitch = self.player_pitch if rotation[1] == '@s' else float(rotation[1])
        else:
            yaw, pitch = 0, 0

        # Convert to vector (Y is up)
        yaw_rad = math.radians(yaw)
        pitch_rad = math.radians(pitch)

        vx = speed * math.cos(yaw_rad) * math.cos(pitch_rad)
        vy = speed * math.sin(pitch_rad)
        vz = speed * math.sin(yaw_rad) * math.cos(pitch_rad)

        return np.array([vx, vy, vz])


class RotationParser:
    """Parse rotation with angle+axis format"""

    @staticmethod
    def parse(data: Any) -> Tuple[float, np.ndarray]:
        """
        Parse rotation data to (angle, axis).

        Formats:
        - {angle:1.57, axis:[1,0,0]}
        - [angle, ax, ay, az]

        Returns:
            (angle in radians, normalized axis vector)
        """
        if isinstance(data, list) and len(data) >= 4:
            angle = float(data[0])
            axis = np.array(data[1:4], dtype=float)
        elif isinstance(data, dict):
            angle = float(data.get('angle', 0))
            axis = np.array(data.get('axis', [0, 1, 0]), dtype=float)
        else:
            return 0.0, np.array([0, 1, 0])

        # Normalize axis
        norm = np.linalg.norm(axis)
        if norm > 0:
            axis = axis / norm

        return angle, axis


class SpinParser:
    """Parse spin with rpm+axis format"""

    @staticmethod
    def parse(data: Any) -> np.ndarray:
        """
        Parse spin data to angular velocity vector (rad/s).

        Formats:
        - {rpm:3000, axis:[0,1,0]}
        - [wx, wy, wz] (rad/s direct)

        Returns:
            Angular velocity vector [wx, wy, wz] in rad/s
        """
        if isinstance(data, list):
            return np.array(data, dtype=float)

        if isinstance(data, dict):
            rpm = float(data.get('rpm', 0))
            axis = np.array(data.get('axis', [0, 1, 0]), dtype=float)

            # Normalize axis
            norm = np.linalg.norm(axis)
            if norm > 0:
                axis = axis / norm

            # Convert RPM to rad/s
            omega = rpm * 2 * math.pi / 60

            return axis * omega

        return np.zeros(3)


class CommandParser:
    """Main command parser with execute support"""

    def __init__(self, game_world):
        """
        Args:
            game_world: Reference to GameWorld for player position/rotation
        """
        self.game = game_world
        self.nbt_parser = NBTParser()

        # Execute context
        self.execute_rotation = None  # Override rotation from 'execute rotate'
        self.execute_position = None  # Override position from 'execute at'

    def get_player_pos_yup(self) -> np.ndarray:
        """Get player position in Y-up coordinate system"""
        if self.execute_position is not None:
            return self.execute_position
        # Camera is now directly in Y-up coordinates
        return self.game.camera_pos.copy()

    def get_player_yaw(self) -> float:
        """Get player yaw (handle execute rotate override)"""
        if self.execute_rotation is not None:
            return self.execute_rotation[0]
        return self.game.camera_yaw

    def get_player_pitch(self) -> float:
        """Get player pitch (handle execute rotate override)"""
        if self.execute_rotation is not None:
            return self.execute_rotation[1]
        return self.game.camera_pitch

    def _find_nearest_entity(self, entity_type: str) -> Optional[np.ndarray]:
        """Find nearest entity of type and return its position"""
        player_pos = self.get_player_pos_yup()
        nearest_dist = float('inf')
        nearest_pos = None

        if entity_type == 'ball':
            entities = self.game.entity_manager.balls
        elif entity_type == 'racket':
            entities = self.game.entity_manager.rackets
        else:
            return None

        for entity in entities:
            dist = np.linalg.norm(entity.position - player_pos)
            if dist < nearest_dist:
                nearest_dist = dist
                nearest_pos = entity.position.copy()

        return nearest_pos

    def parse(self, command: str) -> Dict[str, Any]:
        """Parse command string."""
        command = command.strip()

        # Handle execute command
        if command.startswith('execute '):
            return self._parse_execute(command[8:])

        # Handle summon command
        if command.startswith('summon '):
            return self._parse_summon(command[7:])

        # Handle start command
        if command == 'start' or command.startswith('start '):
            return {'type': 'start', 'args': {}}

        # Handle stop command
        if command == 'stop' or command.startswith('stop '):
            return {'type': 'stop', 'args': {}}

        # Handle kill command
        if command.startswith('kill '):
            return self._parse_kill(command[5:])

        # Handle gamemode command
        if command.startswith('gamemode '):
            return self._parse_gamemode(command[9:])

        # Handle other commands
        return {'type': 'unknown', 'raw': command}

    def _parse_execute(self, rest: str) -> Dict[str, Any]:
        """Parse execute subcommand"""
        # execute rotate as @s run <command>
        match = re.match(r'rotate\s+as\s+@s\s+run\s+(.+)', rest)
        if match:
            # Set rotation override
            self.execute_rotation = (self.game.camera_yaw, self.game.camera_pitch)

            # Parse inner command
            inner_result = self.parse(match.group(1))

            # Clear rotation override
            self.execute_rotation = None

            return inner_result

        # execute at @n[type=ball] run <command>
        match = re.match(r'at\s+@n\[type=(\w+)\]\s+run\s+(.+)', rest)
        if match:
            entity_type = match.group(1)
            inner_cmd = match.group(2)

            # Find nearest entity
            nearest_pos = self._find_nearest_entity(entity_type)
            if nearest_pos is None:
                return {'type': 'error', 'message': f'No {entity_type} found'}

            # Set position override
            self.execute_position = nearest_pos

            # Parse inner command
            inner_result = self.parse(inner_cmd)

            # Clear position override
            self.execute_position = None

            return inner_result

        return {'type': 'unknown', 'raw': f'execute {rest}'}

    def _parse_summon(self, rest: str) -> Dict[str, Any]:
        """Parse summon command: summon <entity> [x y z] [nbt]"""
        # Find NBT part (starts with {)
        nbt_start = rest.find('{')
        if nbt_start != -1:
            coords_part = rest[:nbt_start].strip()
            nbt_str = rest[nbt_start:]
        else:
            coords_part = rest.strip()
            nbt_str = '{}'

        parts = coords_part.split()

        if len(parts) < 1:
            return {'type': 'error', 'message': 'Usage: summon <ball|racket> [x y z] [nbt]'}

        entity_type = parts[0]

        # Default coordinates: ~ ~ ~
        if len(parts) >= 4:
            x_str, y_str, z_str = parts[1], parts[2], parts[3]
        else:
            x_str, y_str, z_str = '~', '~', '~'

        # Parse coordinates
        coord_parser = CoordinateParser(
            self.get_player_pos_yup(),
            self.get_player_yaw(),
            self.get_player_pitch()
        )
        position = coord_parser.parse(x_str, y_str, z_str)

        # Parse NBT
        nbt = self.nbt_parser.parse(nbt_str)

        # Process NBT values
        vel_parser = VelocityParser(self.get_player_yaw(), self.get_player_pitch())

        # Parse velocity if present
        if 'velocity' in nbt:
            nbt['velocity'] = vel_parser.parse(nbt['velocity'])

        # Parse acceleration if present
        if 'acceleration' in nbt:
            nbt['acceleration'] = vel_parser.parse(nbt['acceleration'])

        # Parse spin if present
        if 'spin' in nbt:
            nbt['spin'] = SpinParser.parse(nbt['spin'])

        # Parse rotation if present
        if 'rotation' in nbt:
            angle, axis = RotationParser.parse(nbt['rotation'])
            nbt['rotation'] = {'angle': angle, 'axis': axis}

        # Convert units
        # mass: g -> kg
        if 'mass' in nbt:
            nbt['mass'] = float(nbt['mass']) / 1000.0

        # radius: mm -> m
        if 'radius' in nbt:
            nbt['radius'] = float(nbt['radius']) / 1000.0

        return {
            'type': 'summon',
            'args': {
                'entity': entity_type,
                'position': position,
                'nbt': nbt
            }
        }

    def _parse_kill(self, rest: str) -> Dict[str, Any]:
        """Parse kill command: kill <selector>"""
        selector = rest.strip()
        return {
            'type': 'kill',
            'args': {
                'selector': selector
            }
        }

    def _parse_gamemode(self, rest: str) -> Dict[str, Any]:
        """Parse gamemode command: gamemode gravity <value>"""
        parts = rest.strip().split()

        if len(parts) < 2:
            return {'type': 'error', 'message': 'Usage: gamemode gravity <value>'}

        setting = parts[0]
        value = float(parts[1])

        return {
            'type': 'gamemode',
            'args': {
                'setting': setting,
                'value': value
            }
        }


# Utility function to convert Y-up to Z-up (for rendering)
def yup_to_zup(pos: np.ndarray) -> np.ndarray:
    """Convert [x, y, z] from Y-up to Z-up: [x, z, y]"""
    return np.array([pos[0], pos[2], pos[1]])


# Utility function to convert Z-up to Y-up (for commands)
def zup_to_yup(pos: np.ndarray) -> np.ndarray:
    """Convert [x, y, z] from Z-up to Y-up: [x, z, y]"""
    return np.array([pos[0], pos[2], pos[1]])
