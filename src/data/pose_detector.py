from typing import Optional, Tuple

import numpy as np
import torch


# Halpe26 order returned by rtmlib.BodyWithFeet. The selected points preserve
# the model's existing 15-joint order.
RTMLIB_TO_MODEL_INDICES = (0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
                           15, 16, 20, 23)


class RTMLibPoseDetector:
    def __init__(self, mode: str = "lightweight",
                 device: Optional[str] = None) -> None:
        if mode not in {"lightweight", "balanced", "performance"}:
            raise ValueError(
                "pose mode must be lightweight, balanced, or performance"
            )
        if device is None or device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device not in {"cuda", "cpu"}:
            raise ValueError("pose device must be auto, cuda, or cpu")
        self.mode = mode
        self.device = device
        self._detector = None

    def _get_detector(self):
        if self._detector is None:
            try:
                from rtmlib import RTMPose
                from rtmlib.tools.solution.body_with_feet import BodyWithFeet
            except ImportError as exc:
                raise ImportError(
                    "rtmlib is required for 2D pose detection. "
                    "Install it with: python -m pip install rtmlib"
                ) from exc
            settings = BodyWithFeet.MODE[self.mode]
            self._detector = RTMPose(
                settings["pose"],
                model_input_size=settings["pose_input_size"],
                backend="onnxruntime",
                device=self.device,
            )
            pose_providers = self._detector.session.get_providers()
            if self.device == "cuda" and \
                    "CUDAExecutionProvider" not in pose_providers:
                raise RuntimeError(
                    "rtmlib CUDA inference is unavailable. Install "
                    "onnxruntime-gpu and verify the NVIDIA driver. "
                    f"pose providers={pose_providers}"
                )
            print(
                f"pose_detector=rtmlib mode={self.mode} device={self.device} "
                f"provider={pose_providers[0]} single_person=true"
            )
        return self._detector

    def detect(self, frame_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        height, width = frame_bgr.shape[:2]
        keypoints, scores = self._get_detector()(
            frame_bgr, bboxes=[[0, 0, width, height]]
        )
        if keypoints.ndim != 3 or keypoints.shape[0] == 0:
            return self._empty_result()

        person_index = int(np.argmax(scores.mean(axis=1)))
        selected = keypoints[person_index, list(RTMLIB_TO_MODEL_INDICES)]
        selected_scores = scores[person_index, list(RTMLIB_TO_MODEL_INDICES)]
        points = selected.astype(np.float32) / np.array(
            [width, height], dtype=np.float32
        )
        valid = (
            (selected_scores >= 0.3)
            & (points[:, 0] >= 0.0)
            & (points[:, 0] <= 1.0)
            & (points[:, 1] >= 0.0)
            & (points[:, 1] <= 1.0)
        )
        return points, valid

    @staticmethod
    def _empty_result() -> Tuple[np.ndarray, np.ndarray]:
        return (
            np.zeros((len(RTMLIB_TO_MODEL_INDICES), 2), dtype=np.float32),
            np.zeros(len(RTMLIB_TO_MODEL_INDICES), dtype=bool),
        )
