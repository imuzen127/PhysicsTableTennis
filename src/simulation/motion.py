"""
Motion recording and playback system

Allows recording racket motions and replaying them for automated simulations
"""

import numpy as np
import json
from typing import Optional, Callable, Tuple, Dict, Any
from scipy.interpolate import interp1d


class MotionRecorder:
    """
    ラケットの動作を記録するクラス
    """

    def __init__(self, name: str = "motion"):
        """
        Args:
            name: 動作の名前
        """
        self.name = name
        self.is_recording = False

        # 記録データ
        self.positions = []
        self.orientations = []
        self.time_stamps = []

    def start_recording(self):
        """記録を開始"""
        self.is_recording = True
        self.positions = []
        self.orientations = []
        self.time_stamps = []
        print(f"Started recording motion: {self.name}")

    def stop_recording(self):
        """記録を停止"""
        self.is_recording = False
        print(f"Stopped recording motion: {self.name} ({len(self.time_stamps)} frames)")

    def record_frame(
        self,
        time: float,
        position: np.ndarray,
        orientation: np.ndarray
    ):
        """
        1フレームを記録

        Args:
            time: 現在時刻
            position: ラケットの位置
            orientation: ラケットの姿勢（法線ベクトル）
        """
        if not self.is_recording:
            return

        self.time_stamps.append(time)
        self.positions.append(position.copy())
        self.orientations.append(orientation.copy())

    def save(self, filename: str):
        """
        動作データをファイルに保存

        Args:
            filename: 保存先ファイル名（.npz形式）
        """
        if len(self.time_stamps) == 0:
            print("No data to save")
            return

        np.savez(
            filename,
            name=self.name,
            time_stamps=np.array(self.time_stamps),
            positions=np.array(self.positions),
            orientations=np.array(self.orientations)
        )
        print(f"Motion saved to {filename}")

    def load(self, filename: str):
        """
        動作データをファイルから読み込み

        Args:
            filename: 読み込むファイル名
        """
        data = np.load(filename)

        self.name = str(data['name'])
        self.time_stamps = data['time_stamps'].tolist()
        self.positions = data['positions'].tolist()
        self.orientations = data['orientations'].tolist()

        print(f"Motion loaded from {filename} ({len(self.time_stamps)} frames)")

    def get_data(self) -> Dict[str, Any]:
        """記録データを取得"""
        return {
            'name': self.name,
            'time_stamps': np.array(self.time_stamps),
            'positions': np.array(self.positions),
            'orientations': np.array(self.orientations)
        }

    def __repr__(self) -> str:
        return f"MotionRecorder(name={self.name}, frames={len(self.time_stamps)})"


class MotionPlayer:
    """
    記録した動作を再生するクラス
    """

    def __init__(
        self,
        motion_data: Optional[Dict[str, Any]] = None,
        filename: Optional[str] = None
    ):
        """
        Args:
            motion_data: 動作データの辞書
            filename: 動作データのファイル名（.npz形式）
        """
        self.motion_data = None
        self.interpolator_position = None
        self.interpolator_orientation = None

        if motion_data is not None:
            self.load_from_data(motion_data)
        elif filename is not None:
            self.load_from_file(filename)

    def load_from_data(self, motion_data: Dict[str, Any]):
        """
        動作データを読み込む

        Args:
            motion_data: 動作データの辞書
        """
        self.motion_data = motion_data

        time_stamps = motion_data['time_stamps']
        positions = motion_data['positions']
        orientations = motion_data['orientations']

        # 補間関数を作成
        if len(time_stamps) > 1:
            self.interpolator_position = interp1d(
                time_stamps,
                positions,
                axis=0,
                kind='cubic',
                fill_value='extrapolate'
            )
            self.interpolator_orientation = interp1d(
                time_stamps,
                orientations,
                axis=0,
                kind='linear',
                fill_value='extrapolate'
            )
        else:
            # データが1つしかない場合は定数関数
            self.interpolator_position = lambda t: positions[0]
            self.interpolator_orientation = lambda t: orientations[0]

        print(f"Motion player loaded: {motion_data['name']}")

    def load_from_file(self, filename: str):
        """
        ファイルから動作データを読み込む

        Args:
            filename: 動作データのファイル名
        """
        data = np.load(filename)
        motion_data = {
            'name': str(data['name']),
            'time_stamps': data['time_stamps'],
            'positions': data['positions'],
            'orientations': data['orientations']
        }
        self.load_from_data(motion_data)

    def get_motion(
        self,
        time: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        指定時刻のラケット位置と姿勢を取得

        Args:
            time: 時刻

        Returns:
            position: ラケットの位置
            orientation: ラケットの姿勢
        """
        if self.interpolator_position is None:
            raise ValueError("No motion data loaded")

        position = self.interpolator_position(time)
        orientation = self.interpolator_orientation(time)

        # 姿勢ベクトルを正規化
        orientation = orientation / np.linalg.norm(orientation)

        return position, orientation

    def create_controller(self) -> Callable:
        """
        シミュレーションエンジン用のコントローラ関数を作成

        Returns:
            controller: (time, ball_state) -> (position, orientation) の関数
        """
        def controller(time: float, ball_state: Tuple) -> Tuple[np.ndarray, np.ndarray]:
            return self.get_motion(time)

        return controller

    def __repr__(self) -> str:
        if self.motion_data:
            return f"MotionPlayer(name={self.motion_data['name']})"
        else:
            return "MotionPlayer(no data loaded)"


class PredefinedMotions:
    """
    事前定義された動作パターン
    """

    @staticmethod
    def create_forehand_drive(
        side: int = 1,
        table_length: float = 2.74
    ) -> Dict[str, Any]:
        """
        フォアハンドドライブの動作を生成

        Args:
            side: どちら側か (1: +X側, -1: -X側)
            table_length: テーブルの長さ

        Returns:
            motion_data: 動作データ
        """
        # バックスイングから打球、フォロースルーまで
        time_stamps = np.linspace(0, 1.0, 50)

        positions = []
        orientations = []

        for t in time_stamps:
            # バックスイング (t=0-0.3) -> 打球 (t=0.3-0.5) -> フォロースルー (t=0.5-1.0)
            x = side * (table_length / 2 + 0.5)
            y = 0.3 - 0.3 * np.cos(2 * np.pi * t)  # 横方向の動き
            z = 0.8 + 0.2 * np.sin(2 * np.pi * t)  # 上下の動き

            position = np.array([x, y, z])

            # 姿勢（打球面が相手側を向く）
            angle = np.pi / 6 * np.sin(2 * np.pi * t)  # 角度変化
            orientation = np.array([-side * np.cos(angle), 0.0, np.sin(angle)])

            positions.append(position)
            orientations.append(orientation)

        return {
            'name': f'forehand_drive_side{side}',
            'time_stamps': time_stamps,
            'positions': np.array(positions),
            'orientations': np.array(orientations)
        }

    @staticmethod
    def create_static_defense(
        side: int = 1,
        table_length: float = 2.74
    ) -> Dict[str, Any]:
        """
        静的な防御姿勢を生成

        Args:
            side: どちら側か
            table_length: テーブルの長さ

        Returns:
            motion_data: 動作データ
        """
        x = side * (table_length / 2 + 0.3)
        position = np.array([x, 0.0, 0.9])
        orientation = np.array([-side, 0.0, 0.0])

        return {
            'name': f'static_defense_side{side}',
            'time_stamps': np.array([0.0, 10.0]),
            'positions': np.array([position, position]),
            'orientations': np.array([orientation, orientation])
        }

    @staticmethod
    def create_serve_motion(
        side: int = 1,
        table_length: float = 2.74
    ) -> Dict[str, Any]:
        """
        サーブの動作を生成

        Args:
            side: どちら側か
            table_length: テーブルの長さ

        Returns:
            motion_data: 動作データ
        """
        time_stamps = np.linspace(0, 0.5, 30)

        positions = []
        orientations = []

        for t in time_stamps:
            x = side * (table_length / 4)
            y = 0.0
            z = 1.2 - 0.4 * t / 0.5  # 下から上へ

            position = np.array([x, y, z])

            # 姿勢（下から上へ振り上げる）
            angle = -np.pi / 4 + (np.pi / 3) * (t / 0.5)
            orientation = np.array([-side * np.cos(angle), 0.0, np.sin(angle)])

            positions.append(position)
            orientations.append(orientation)

        return {
            'name': f'serve_motion_side{side}',
            'time_stamps': time_stamps,
            'positions': np.array(positions),
            'orientations': np.array(orientations)
        }
