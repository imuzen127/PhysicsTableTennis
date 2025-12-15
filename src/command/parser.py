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
        self.execute_entity = None    # Target entity from 'execute as'

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

    def get_player_rotation_angle_axis(self) -> Tuple[float, np.ndarray]:
        """
        Get player rotation as angle-axis format.
        Converts yaw/pitch to a combined rotation.
        """
        yaw = math.radians(self.get_player_yaw())
        pitch = math.radians(self.get_player_pitch())

        # Create rotation: first pitch around X, then yaw around Y
        # For simplicity, we'll compute the forward direction and derive angle-axis
        # Forward direction after rotation
        forward = np.array([
            math.cos(pitch) * math.cos(yaw),
            math.sin(pitch),
            math.cos(pitch) * math.sin(yaw)
        ])

        # Default forward is +Z
        default_forward = np.array([0.0, 0.0, 1.0])

        # Compute rotation axis and angle using cross product and dot product
        dot = np.dot(default_forward, forward)
        dot = np.clip(dot, -1.0, 1.0)  # Clamp for numerical stability

        if dot > 0.9999:
            # No rotation needed
            return 0.0, np.array([0.0, 1.0, 0.0])
        elif dot < -0.9999:
            # 180 degree rotation
            return math.pi, np.array([0.0, 1.0, 0.0])

        angle = math.acos(dot)
        axis = np.cross(default_forward, forward)
        axis = axis / np.linalg.norm(axis)

        return angle, axis

    def _find_nearest_entity(self, entity_type: str) -> Optional[np.ndarray]:
        """Find nearest entity of type and return its position"""
        entity = self._get_nearest_entity(entity_type)
        return entity.position.copy() if entity else None

    def _get_nearest_entity(self, entity_type: str):
        """Find nearest entity of type and return the entity itself"""
        player_pos = self.get_player_pos_yup()
        nearest_dist = float('inf')
        nearest_entity = None

        if entity_type == 'ball':
            entities = self.game.entity_manager.balls
        elif entity_type == 'racket':
            entities = self.game.entity_manager.rackets
        elif entity_type == 'table':
            entities = self.game.entity_manager.tables
        else:
            return None

        for entity in entities:
            dist = np.linalg.norm(entity.position - player_pos)
            if dist < nearest_dist:
                nearest_dist = dist
                nearest_entity = entity

        return nearest_entity

    def _parse_selector_args(self, args_str: str) -> Dict[str, str]:
        """Parse selector arguments like type=ball,tag=mytag"""
        args = {}
        if not args_str:
            return args
        for part in args_str.split(','):
            if '=' in part:
                key, value = part.split('=', 1)
                args[key.strip()] = value.strip()
        return args

    def _filter_entities_by_args(self, entities: list, args: Dict[str, str]) -> list:
        """Filter entity list by selector arguments (type, tag)"""
        result = entities
        if 'type' in args:
            target_type = args['type']
            result = [e for e in result if e.entity_type.value == target_type]
        if 'tag' in args:
            target_tag = args['tag']
            result = [e for e in result if target_tag in getattr(e, 'tags', [])]
        return result

    def _get_all_entities(self) -> list:
        """Get all entities (balls, rackets, tables)"""
        all_entities = []
        all_entities.extend(self.game.entity_manager.balls)
        all_entities.extend(self.game.entity_manager.rackets)
        all_entities.extend(self.game.entity_manager.tables)
        return all_entities

    def _resolve_selector(self, selector: str):
        """Resolve entity selector (@s, @n[...], @e[...]) to entity or player"""
        if selector == '@s':
            if self.execute_entity is not None:
                return self.execute_entity
            else:
                return self._get_player_entity()
        match = re.match(r'@([ne])(?:\[(.*?)\])?$', selector)
        if match:
            selector_type = match.group(1)
            args_str = match.group(2) or ''
            args = self._parse_selector_args(args_str)
            all_entities = self._get_all_entities()
            filtered = self._filter_entities_by_args(all_entities, args)
            if selector_type == 'n':
                player_pos = self.get_player_pos_yup()
                nearest_dist = float('inf')
                nearest_entity = None
                for entity in filtered:
                    dist = np.linalg.norm(entity.position - player_pos)
                    if dist < nearest_dist:
                        nearest_dist = dist
                        nearest_entity = entity
                return nearest_entity
            elif selector_type == 'e':
                return filtered[0] if filtered else None
        return None

    def _resolve_selector_multiple(self, selector: str) -> list:
        """Resolve selector to multiple entities (for @e)"""
        if selector == '@s':
            entity = self._resolve_selector(selector)
            return [entity] if entity else []
        match = re.match(r'@([ne])(?:\[(.*?)\])?$', selector)
        if match:
            selector_type = match.group(1)
            args_str = match.group(2) or ''
            args = self._parse_selector_args(args_str)
            all_entities = self._get_all_entities()
            filtered = self._filter_entities_by_args(all_entities, args)
            if selector_type == 'n':
                entity = self._resolve_selector(selector)
                return [entity] if entity else []
            elif selector_type == 'e':
                return filtered
        return []

    def _get_player_entity(self):
        """Return player as a pseudo-entity with position and rotation"""
        class PlayerEntity:
            def __init__(self, game):
                self.id = "player"
                self.entity_type = type('EntityType', (), {'value': 'player'})()
                self.position = game.camera_pos.copy()
                self.velocity = np.zeros(3)
                self.active = True
                # Get angle-axis rotation from game
                self.orientation_angle, self.orientation_axis = game.get_player_rotation_angle_axis()

        return PlayerEntity(self.game)

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
            selector = None
            if command.startswith('start '):
                selector = command[6:].strip()
                if not selector:
                    selector = None
            return {'type': 'start', 'args': {'selector': selector}}

        # Handle stop command
        if command == 'stop' or command.startswith('stop '):
            selector = None
            if command.startswith('stop '):
                selector = command[5:].strip()
                if not selector:
                    selector = None
            return {'type': 'stop', 'args': {'selector': selector}}

        # Handle kill command
        if command.startswith('kill '):
            return self._parse_kill(command[5:])

        # Handle gamemode command
        if command.startswith('gamemode '):
            return self._parse_gamemode(command[9:])

        # Handle data command
        if command.startswith('data '):
            return self._parse_data(command[5:])

        # Handle function command
        if command.startswith('function '):
            return self._parse_function(command[9:])

        # Handle tp command
        if command.startswith('tp '):
            return self._parse_tp(command[3:])

        # Handle tag command
        if command.startswith('tag '):
            return self._parse_tag(command[4:])

        # Handle rotate command
        if command.startswith('rotate '):
            return self._parse_rotate(command[7:])

        # Handle other commands
        return {'type': 'unknown', 'raw': command}

    def _parse_function(self, rest: str) -> Dict[str, Any]:
        """Parse function command: function <name>"""
        func_name = rest.strip()

        # Remove .mcfunction extension if present
        if func_name.endswith('.mcfunction'):
            func_name = func_name[:-11]

        return {
            'type': 'function',
            'args': {
                'name': func_name
            }
        }

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

        # execute as @n[type=ball] run <command> - set @s to nearest entity
        match = re.match(r'as\s+(@\w+(?:\[.*?\])?)\s+run\s+(.+)', rest)
        if match:
            selector = match.group(1)
            inner_cmd = match.group(2)

            # Resolve selector to entity
            entity = self._resolve_selector(selector)
            if entity is None:
                return {'type': 'error', 'message': f'No entity found for {selector}'}

            # Set entity context (@s will resolve to this)
            self.execute_entity = entity

            # Parse inner command
            inner_result = self.parse(inner_cmd)

            # Clear entity context
            self.execute_entity = None

            return inner_result

        # execute at @n[type=ball] run <command>
        match = re.match(r'at\s+(@\w+(?:\[.*?\])?)\s+run\s+(.+)', rest)
        if match:
            selector = match.group(1)
            inner_cmd = match.group(2)

            # Resolve selector to entity
            entity = self._resolve_selector(selector)
            if entity is None:
                return {'type': 'error', 'message': f'No entity found for {selector}'}

            # Set position override
            self.execute_position = entity.position.copy()

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

        # Parse rotation FIRST (needed for velocity direction)
        rotation_angle = 0.0
        rotation_axis = np.array([0, 1, 0])
        if 'rotation' in nbt:
            rotation_angle, rotation_axis = RotationParser.parse(nbt['rotation'])
            nbt['rotation'] = {'angle': rotation_angle, 'axis': rotation_axis}

        # Parse spin if present
        if 'spin' in nbt:
            nbt['spin'] = SpinParser.parse(nbt['spin'])

        # Parse velocity
        # Formats:
        #   velocity: 0.5  (scalar - use object's rotation direction)
        #   velocity: [vx, vy, vz]  (direct vector)
        #   velocity: {angle:1.57, axis:[0,1,0], speed:0.5}  (angle-axis + speed)
        #   velocity: {rotation:@s, speed:10}  (player direction + speed)
        if 'velocity' in nbt:
            vel_data = nbt['velocity']
            if isinstance(vel_data, (int, float)):
                # Scalar velocity: use object's rotation direction
                speed = float(vel_data)
                default_dir = np.array([0.0, 0.0, 1.0])
                if abs(rotation_angle) > 1e-6:
                    # Rodrigues' rotation formula
                    k = rotation_axis
                    v = default_dir
                    cos_a = math.cos(rotation_angle)
                    sin_a = math.sin(rotation_angle)
                    direction = v * cos_a + np.cross(k, v) * sin_a + k * np.dot(k, v) * (1 - cos_a)
                else:
                    direction = default_dir
                nbt['velocity'] = direction * speed
            elif isinstance(vel_data, list):
                # Direct vector
                nbt['velocity'] = np.array(vel_data, dtype=float)
            elif isinstance(vel_data, dict):
                # Dict format: check for angle-axis or rotation
                if 'angle' in vel_data and 'axis' in vel_data:
                    # Angle-axis format: {angle:1.57, axis:[0,1,0], speed:0.5}
                    vel_angle = float(vel_data.get('angle', 0))
                    vel_axis = np.array(vel_data.get('axis', [0, 1, 0]), dtype=float)
                    norm = np.linalg.norm(vel_axis)
                    if norm > 0:
                        vel_axis = vel_axis / norm
                    speed = float(vel_data.get('speed', 0))

                    default_dir = np.array([0.0, 0.0, 1.0])
                    if abs(vel_angle) > 1e-6:
                        k = vel_axis
                        v = default_dir
                        cos_a = math.cos(vel_angle)
                        sin_a = math.sin(vel_angle)
                        direction = v * cos_a + np.cross(k, v) * sin_a + k * np.dot(k, v) * (1 - cos_a)
                    else:
                        direction = default_dir
                    nbt['velocity'] = direction * speed
                else:
                    # rotation/@s format: {rotation:@s, speed:10}
                    vel_parser = VelocityParser(self.get_player_yaw(), self.get_player_pitch())
                    nbt['velocity'] = vel_parser.parse(vel_data)

        # Parse acceleration if present
        if 'acceleration' in nbt:
            vel_parser = VelocityParser(self.get_player_yaw(), self.get_player_pitch())
            nbt['acceleration'] = vel_parser.parse(nbt['acceleration'])

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

    def _parse_data(self, rest: str) -> Dict[str, Any]:
        """
        Parse data command:
        - data get entity <selector> [path]
        - data modify entity <selector> <path> set value <value>
        """
        # data get entity <selector>
        match = re.match(r'get\s+entity\s+(@\w+(?:\[.*?\])?)\s*(.*)', rest)
        if match:
            selector = match.group(1)
            path = match.group(2).strip() if match.group(2) else None

            entity = self._resolve_selector(selector)
            if entity is None:
                return {'type': 'error', 'message': f'No entity found for {selector}'}

            return {
                'type': 'data_get',
                'args': {
                    'entity': entity,
                    'path': path
                }
            }

        # data modify entity <selector> <path> set value <value>
        match = re.match(r'modify\s+entity\s+(@\w+(?:\[.*?\])?)\s+(\w+)\s+set\s+value\s+(.+)', rest)
        if match:
            selector = match.group(1)
            path = match.group(2)
            value_str = match.group(3).strip()

            entity = self._resolve_selector(selector)
            if entity is None:
                return {'type': 'error', 'message': f'No entity found for {selector}'}

            # Parse value - support lists, numbers, and strings
            value = self._parse_data_value(value_str)

            return {
                'type': 'data_modify',
                'args': {
                    'entity': entity,
                    'path': path,
                    'value': value
                }
            }

        return {'type': 'error', 'message': 'Usage: data get/modify entity <selector> ...'}

    def _parse_tp(self, rest: str) -> Dict[str, Any]:
        """
        Parse tp command: tp <selector> <x> <y> <z>
        Examples:
            tp @n[type=ball] 0 1 0
            tp @n[type=ball] ~ ~1 ~
        """
        parts = rest.strip().split()
        if len(parts) < 4:
            return {'type': 'error', 'message': 'Usage: tp <selector> <x> <y> <z>'}

        selector = parts[0]
        x_str, y_str, z_str = parts[1], parts[2], parts[3]

        entity = self._resolve_selector(selector)
        if entity is None:
            return {'type': 'error', 'message': f'No entity found for {selector}'}

        # Parse coordinates (supports ~ for relative)
        coord_parser = CoordinateParser(
            entity.position,  # Use entity position as base for relative coords
            self.get_player_yaw(),
            self.get_player_pitch()
        )
        position = coord_parser.parse(x_str, y_str, z_str)

        return {
            'type': 'tp',
            'args': {
                'entity': entity,
                'position': position
            }
        }

    def _parse_rotate(self, rest: str) -> Dict[str, Any]:
        """
        Parse rotate command: rotate <selector>
        Applies player rotation to the entity.
        Examples:
            rotate @n[type=ball]
            rotate @n[type=racket]
        """
        selector = rest.strip()
        if not selector:
            return {'type': 'error', 'message': 'Usage: rotate <selector>'}

        entity = self._resolve_selector(selector)
        if entity is None:
            return {'type': 'error', 'message': f'No entity found for {selector}'}

        # Get player rotation
        angle, axis = self.get_player_rotation_angle_axis()

        return {
            'type': 'rotate',
            'args': {
                'entity': entity,
                'angle': angle,
                'axis': axis
            }
        }

    def _parse_tag(self, rest: str) -> Dict[str, Any]:
        """
        Parse tag command:
        - tag <selector> add <tagname>
        - tag <selector> remove <tagname>
        - tag <selector> list
        """
        parts = rest.strip().split()
        if len(parts) < 2:
            return {'type': 'error', 'message': 'Usage: tag <selector> add|remove|list [tagname]'}

        selector = parts[0]
        action = parts[1]

        if action == 'list':
            return {
                'type': 'tag_list',
                'args': {
                    'selector': selector
                }
            }
        elif action in ('add', 'remove'):
            if len(parts) < 3:
                return {'type': 'error', 'message': f'Usage: tag <selector> {action} <tagname>'}
            tagname = parts[2]
            return {
                'type': f'tag_{action}',
                'args': {
                    'selector': selector,
                    'tagname': tagname
                }
            }
        else:
            return {'type': 'error', 'message': f'Unknown tag action: {action}'}

    def _parse_data_value(self, value_str: str) -> Any:
        """Parse value for data modify command - supports dicts, lists, numbers, strings, selectors"""
        value_str = value_str.strip()

        # Selector @s - return player/entity rotation as dict
        if value_str == '@s':
            # Get rotation from execute context or player
            angle, axis = self.get_player_rotation_angle_axis()
            return {
                'angle': angle,
                'axis': axis.tolist()
            }

        # Dict value (NBT format): {angle:1.57, axis:[0,1,0]}
        if value_str.startswith('{') and value_str.endswith('}'):
            return self.nbt_parser.parse(value_str)

        # List value: [1.2, 0.9] or ["inverted", "pimples"]
        if value_str.startswith('[') and value_str.endswith(']'):
            inner = value_str[1:-1].strip()
            if not inner:
                return []
            # Split by comma, handling potential spaces
            parts = [p.strip() for p in inner.split(',')]
            result = []
            for part in parts:
                # Remove quotes if present
                if (part.startswith('"') and part.endswith('"')) or \
                   (part.startswith("'") and part.endswith("'")):
                    result.append(part[1:-1])
                else:
                    # Try to parse as number
                    try:
                        if '.' in part:
                            result.append(float(part))
                        else:
                            result.append(int(part))
                    except ValueError:
                        result.append(part)
            return result

        # Boolean
        if value_str.lower() == 'true':
            return True
        if value_str.lower() == 'false':
            return False

        # Number
        try:
            if '.' in value_str:
                return float(value_str)
            else:
                return int(value_str)
        except ValueError:
            pass

        # String (remove quotes if present)
        if (value_str.startswith('"') and value_str.endswith('"')) or \
           (value_str.startswith("'") and value_str.endswith("'")):
            return value_str[1:-1]

        return value_str


# Utility function to convert Y-up to Z-up (for rendering)
def yup_to_zup(pos: np.ndarray) -> np.ndarray:
    """Convert [x, y, z] from Y-up to Z-up: [x, z, y]"""
    return np.array([pos[0], pos[2], pos[1]])


# Utility function to convert Z-up to Y-up (for commands)
def zup_to_yup(pos: np.ndarray) -> np.ndarray:
    """Convert [x, y, z] from Z-up to Y-up: [x, z, y]"""
    return np.array([pos[0], pos[2], pos[1]])
