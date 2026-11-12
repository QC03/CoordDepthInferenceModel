import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence
from urllib.request import urlopen
import torch
from torch.utils.data import Dataset

from .coordinates import h36m_root_relative


class Human36MDataset(Dataset):
    """Reads this repository's per-frame joints3d_25 JSON files.

    Images and MediaPipe detections are intentionally injected by a manifest;
    this avoids silently substituting GT projections for real detector input.
    """
    def __init__(self, manifest: str, subjects: Sequence[str],
                 left_hip: int, right_hip: int,
                 h36m_indices: Optional[Sequence[int]] = None) -> None:
        self.samples: List[Dict] = []
        self.left_hip, self.right_hip = left_hip, right_hip
        self.h36m_indices = h36m_indices
        manifest_path = Path(manifest)
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"Manifest not found: {manifest_path.resolve()}. "
                "Create it from examples/manifest.example.json and update "
                "the image_tensor, joints3d_json, and keypoints_2d paths."
            )
        rows = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(rows, list):
            raise ValueError("Manifest must contain a JSON array of samples")
        for row in rows:
            if row["subject"] in subjects:
                sample = dict(row)
                for field in ("image_tensor", "joints3d_json"):
                    data_path = Path(sample[field])
                    if not data_path.is_absolute():
                        data_path = manifest_path.parent / data_path
                    if not data_path.is_file():
                        raise FileNotFoundError(
                            f"Manifest sample for subject "
                            f"{sample['subject']!r} refers to missing "
                            f"{field}: {data_path.resolve()}. "
                            f"Replace the example path in {manifest_path} "
                            "with a real dataset file."
                        )
                    sample[field] = str(data_path)
                self.samples.append(sample)
        if not self.samples:
            raise ValueError(
                f"No manifest samples found for subjects: {', '.join(subjects)}"
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        row = self.samples[index]
        points = torch.tensor(json.loads(
            Path(row["joints3d_json"]).read_text(encoding="utf-8"))
            ["joints3d_25"], dtype=torch.float32)
        # A JSON can contain a frame sequence; manifest selects one frame.
        points = points[row.get("frame", 0)]
        if self.h36m_indices is not None:
            points = points[list(self.h36m_indices)]
        target = h36m_root_relative(points[None], self.left_hip,
                                    self.right_hip)[0, :, 2:]
        image = torch.load(row["image_tensor"], weights_only=True)
        keypoints = torch.tensor(row["keypoints_2d"], dtype=torch.float32)
        valid = torch.tensor(row.get("valid_mask", [True] * len(keypoints)))
        return {"images": image, "keypoints_2d": keypoints,
                "target_depth": target, "valid_mask": valid}


class VideoHuman36MDataset(Dataset):
    """Loads this repository's Human3.6M videos and 3D labels directly.

    MediaPipe detections are computed from each video frame because the
    repository contains videos, not precomputed image tensors.
    """
    _mp_indices = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 31, 32]
    _pose_model_url = (
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        "pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
    )

    def __init__(self, data_root: str, subjects: Sequence[str],
                 left_hip: int, right_hip: int, frame_stride: int = 1,
                 max_samples: Optional[int] = None,
                 pose_model: Optional[str] = None) -> None:
        if frame_stride < 1:
            raise ValueError("frame_stride must be at least 1")
        try:
            import cv2
            import mediapipe as mp
        except ImportError as exc:
            raise ImportError(
                "Automatic train/ loading requires mediapipe and opencv-python. "
                "Install them with: python -m pip install mediapipe opencv-python"
            ) from exc
        self._cv2 = cv2
        import numpy as np
        self._np = np
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
        model_path = self._ensure_pose_model(pose_model)
        options = vision.PoseLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.IMAGE,
            num_poses=1,
        )
        self._pose = vision.PoseLandmarker.create_from_options(options)
        self.left_hip, self.right_hip = left_hip, right_hip
        self.samples = []
        self.labels: Dict[str, torch.Tensor] = {}
        root = Path(data_root)
        for subject in subjects:
            subject_root = root / subject
            labels_root = subject_root / "joints3d_25"
            videos_root = subject_root / "videos"
            for label_path in sorted(labels_root.glob("*.json")):
                action = label_path.stem
                labels = json.loads(label_path.read_text(encoding="utf-8"))
                frames = torch.tensor(labels["joints3d_25"],
                                      dtype=torch.float32)
                self.labels[str(label_path)] = frames
                for video_path in sorted(videos_root.glob(f"*/{action}.mp4")):
                    for frame in range(0, len(frames), frame_stride):
                        self.samples.append((str(video_path), str(label_path),
                                             frame))
        if not self.samples:
            raise ValueError(
                f"No Human3.6M video/label samples found under {root.resolve()}"
            )
        if max_samples is not None and len(self.samples) > max_samples:
            if max_samples < 1:
                raise ValueError("max_samples must be at least 1")
            if max_samples == 1:
                selected = [self.samples[0]]
            else:
                last_index = len(self.samples) - 1
                selected = [
                    self.samples[
                        round(index * last_index / (max_samples - 1))
                    ]
                    for index in range(max_samples)
                ]
            self.samples = selected

    @classmethod
    def _ensure_pose_model(cls, pose_model: Optional[str]) -> Path:
        if pose_model:
            path = Path(pose_model)
            if path.suffix.lower() != ".task":
                raise ValueError("--pose-model must point to a .task file")
            if not path.is_file():
                raise FileNotFoundError(f"Pose model not found: {path.resolve()}")
            return path
        cache_path = Path.home() / ".cache" / "coord-depth" / "pose_landmarker_lite.task"
        if not cache_path.is_file():
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                with urlopen(cls._pose_model_url, timeout=60) as response:
                    cache_path.write_bytes(response.read())
            except Exception as exc:
                raise RuntimeError(
                    "Could not download the MediaPipe pose model. "
                    "Pass --pose-model PATH to a local .task file."
                ) from exc
        return cache_path
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        video_path, label_path, frame_index = self.samples[index]
        capture = self._cv2.VideoCapture(video_path)
        capture.set(self._cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
        capture.release()
        if not ok:
            raise RuntimeError(
                f"Could not read frame {frame_index} from {video_path}"
            )
        rgb = self._cv2.cvtColor(frame, self._cv2.COLOR_BGR2RGB)
        image = self._mp_image(rgb)
        result = self._pose.detect(image)
        if not result.pose_landmarks:
            keypoints = self._np.zeros((len(self._mp_indices), 2),
                                       dtype=self._np.float32)
            valid = self._np.zeros(len(self._mp_indices),
                                   dtype=self._np.bool_)
        else:
            landmarks = result.pose_landmarks[0]
            keypoints = self._np.array(
                [(landmarks[i].x, landmarks[i].y) for i in self._mp_indices],
                dtype=self._np.float32)
            valid = self._np.array(
                [landmarks[i].visibility >= 0.5 and 0 <= landmarks[i].x <= 1
                 and 0 <= landmarks[i].y <= 1 for i in self._mp_indices],
                dtype=self._np.bool_)
        image = (self._cv2.resize(rgb, (224, 224))
                 .astype(self._np.float32) / 255.0)
        image_tensor = torch.from_numpy(image.transpose(2, 0, 1))
        points = self.labels[label_path][frame_index, :15]
        target = h36m_root_relative(points[None], self.left_hip,
                                    self.right_hip)[0, :, 2:]
        return {
            "images": image_tensor,
            "keypoints_2d": torch.from_numpy(keypoints),
            "target_depth": target,
            "valid_mask": torch.from_numpy(valid),
        }

    def _mp_image(self, rgb):
        import mediapipe as mp
        return mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)


def ensure_pose_model(pose_model: Optional[str] = None) -> Path:
    """Return a validated MediaPipe Tasks pose model path."""
    return VideoHuman36MDataset._ensure_pose_model(pose_model)
