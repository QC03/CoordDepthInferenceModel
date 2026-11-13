import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence
import torch
from torch.utils.data import Dataset

from .coordinates import h36m_root_relative, world_to_camera
from .pose_detector import RTMLibPoseDetector


# Output order matches the 15 rtmlib landmarks used by the model:
# nose, shoulders, elbows, wrists, hips, knees, ankles, and feet.
H36M_TO_MODEL_INDICES = (9, 11, 14, 12, 15, 13, 16, 4, 1,
                         5, 2, 6, 3, 17, 18)


class Human36MDataset(Dataset):
    """Reads this repository's per-frame joints3d_25 JSON files.

    Images and rtmlib detections are intentionally injected by a manifest;
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
    """Loads Human3.6M videos and 3D labels with rtmlib pose detection."""

    def __init__(self, data_root: str, subjects: Sequence[str],
                 left_hip: int, right_hip: int, frame_stride: int = 1,
                 max_samples: Optional[int] = None,
                 pose_mode: str = "lightweight",
                 pose_device: Optional[str] = None) -> None:
        if frame_stride < 1:
            raise ValueError("frame_stride must be at least 1")
        self._pose_detector = RTMLibPoseDetector(pose_mode, pose_device)
        self._capture = None
        self._capture_path = None
        self.left_hip, self.right_hip = left_hip, right_hip
        self.samples = []
        self.labels: Dict[str, torch.Tensor] = {}
        self.camera_parameters: Dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
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

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        import cv2
        import numpy as np

        video_path, label_path, frame_index = self.samples[index]
        frame = self._read_video_frame(
            cv2, video_path, frame_index
        )
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        keypoints, valid = self._pose_detector.detect(frame)
        image = (cv2.resize(rgb, (224, 224)).astype(np.float32) / 255.0)
        image_tensor = torch.from_numpy(image.transpose(2, 0, 1))
        points = self.labels[label_path][frame_index]
        camera_id = Path(video_path).parent.name
        parameters = self._load_camera_parameters(
            Path(label_path).stem, camera_id, Path(label_path).parents[1]
        )
        points = world_to_camera(points, *parameters)
        points = points[list(H36M_TO_MODEL_INDICES)]
        target = h36m_root_relative(points[None], self.left_hip,
                                    self.right_hip)[0, :, 2:]
        return {
            "images": image_tensor,
            "keypoints_2d": torch.from_numpy(keypoints),
            "target_depth": target,
            "valid_mask": torch.from_numpy(valid),
        }

    def _read_video_frame(self, cv2, video_path: str,
                          frame_index: int):
        for attempt in range(2):
            if self._capture_path != video_path:
                self._release_capture()
            capture = self._capture
            if capture is None or not capture.isOpened():
                capture = cv2.VideoCapture(video_path)
                self._capture = capture
                self._capture_path = video_path
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if ok:
                return frame

            self._release_capture()
            if attempt == 0:
                continue
        raise RuntimeError(
            f"Could not read frame {frame_index} from {video_path} "
            "after reopening the video capture"
        )

    def _release_capture(self) -> None:
        if self._capture is not None:
            self._capture.release()
        self._capture = None
        self._capture_path = None

    def _load_camera_parameters(
        self, action: str, camera_id: str, subject_root: Path
    ) -> tuple[torch.Tensor, torch.Tensor]:
        path = subject_root / "camera_parameters" / camera_id / f"{action}.json"
        cache_key = str(path)
        if cache_key in self.camera_parameters:
            return self.camera_parameters[cache_key]
        if not path.is_file():
            raise FileNotFoundError(
                f"Camera parameters not found for video {camera_id}/{action}: "
                f"{path.resolve()}"
            )
        data = json.loads(path.read_text(encoding="utf-8"))
        try:
            extrinsics = data["extrinsics"]
            rotation = torch.tensor(extrinsics["R"], dtype=torch.float32)
            translation = torch.tensor(extrinsics["T"], dtype=torch.float32).reshape(3)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid camera extrinsics in {path.resolve()}"
            ) from exc
        self.camera_parameters[cache_key] = rotation, translation
        return rotation, translation


