"""
Minecraft-style command parser with NBT data support.

Coordinate Systems:
- World: X (horizontal), Y (height/up), Z (horizontal)
- ~ ~ ~ : Relative to player position
- ^ ^ ^ : Local coordinates (left/up/forward based on player facing)

NBT Syntax:
{key:value, key2:[array], key3:{nested}}

Example Commands:
summon ball ~ ~1 ~ {velocity:{rotation:[@s], speed:15}, spin:[0,100,0]}
execute rotate as @s run summon racket ^0 ^0 ^1 {velocity:{rotation:@s, speed:10}}
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
        """Parse @s, @e, etc."""
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
        left_vec = left_vec / np.linalg.norm(left_vec)

        # Calculate world position
        return self.player_pos + left * left_vec + up * up_vec + forward * forward_vec


class VelocityParser:
    """Parse velocity/acceleration with rotation + speed format"""

    def __init__(self, player_yaw: float, player_pitch: float):
        self.player_yaw = player_yaw
        self.player_pitch = player_pitch

    def parse(self, data: Dict[str, Any]) -> np.ndarray:
        """
        Parse velocity data to vector.

        Formats:
        - {rotation:[yaw, pitch], speed:N}
        - {rotation:@s, speed:N}
        - [vx, vy, vz] (direct vector)
        """
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

    def get_player_pos_yup(self) -> np.ndarray:
        """Get player position in Y-up coordinate system"""
        # Convert from Z-up to Y-up: [x, y, z] -> [x, z, y]
        pos = self.game.camera_pos
        return np.array([pos[0], pos[2], pos[1]])

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

    def parse(self, command: str) -> Dict[str, Any]:
        """
        Parse command string.

        Returns:
            {
                'type': 'summon'|'execute'|'start'|'kill'|...,
                'args': {...}
            }
        """
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

        # Handle kill command
        if command.startswith('kill '):
            return self._parse_kill(command[5:])

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

        return {'type': 'unknown', 'raw': f'execute {rest}'}

    def _parse_summon(self, rest: str) -> Dict[str, Any]:
        """Parse summon command: summon <entity> <x> <y> <z> [nbt]"""
        parts = rest.split(None, 4)  # Split into at most 5 parts

        if len(parts) < 4:
            return {'type': 'error', 'message': 'Usage: summon <ball|racket> <x> <y> <z> [nbt]'}

        entity_type = parts[0]
        x_str, y_str, z_str = parts[1], parts[2], parts[3]
        nbt_str = parts[4] if len(parts) > 4 else '{}'

        # Parse coordinates
        coord_parser = CoordinateParser(
            self.get_player_pos_yup(),
            self.get_player_yaw(),
            self.get_player_pitch()
        )
        position = coord_parser.parse(x_str, y_str, z_str)

        # Parse NBT
        nbt = self.nbt_parser.parse(nbt_str)

        # Parse velocity if present
        if 'velocity' in nbt:
            vel_parser = VelocityParser(self.get_player_yaw(), self.get_player_pitch())
            nbt['velocity'] = vel_parser.parse(nbt['velocity'])

        # Parse acceleration if present
        if 'acceleration' in nbt:
            vel_parser = VelocityParser(self.get_player_yaw(), self.get_player_pitch())
            nbt['acceleration'] = vel_parser.parse(nbt['acceleration'])

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


# Utility function to convert Y-up to Z-up (for rendering)
def yup_to_zup(pos: np.ndarray) -> np.ndarray:
    """Convert [x, y, z] from Y-up to Z-up: [x, z, y]"""
    return np.array([pos[0], pos[2], pos[1]])


# Utility function to convert Z-up to Y-up (for commands)
def zup_to_yup(pos: np.ndarray) -> np.ndarray:
    """Convert [x, y, z] from Z-up to Y-up: [x, z, y]"""
    return np.array([pos[0], pos[2], pos[1]])
