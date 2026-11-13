import argparse
import time
from pathlib import Path
from typing import Dict, Optional
import numpy as np

from ..data.pose_detector import RTMLibPoseDetector
from .filtering import OneEuroFilter


_SKELETON_EDGES = (
    (0, 1), (0, 2),
    (1, 2), (1, 3), (3, 5),
    (2, 4), (4, 6),
    (1, 7), (2, 8), (7, 8),
    (7, 9), (9, 11), (11, 13),
    (8, 10), (10, 12), (12, 14),
)


def _draw_skeleton_projection(points: np.ndarray, depths: np.ndarray,
                              plane: str, size: int = 480,
                              depth_extent_m: float = 2.0,
                              valid_mask: Optional[np.ndarray] = None) -> np.ndarray:
    import cv2
    canvas = np.full((size, size, 3), 28, dtype=np.uint8)
    if len(points) == 0:
        return canvas
    if valid_mask is None:
        valid_mask = np.ones(len(points), dtype=bool)
    if len(valid_mask) != len(points):
        raise ValueError("valid_mask length must match points")
    centered = points - 0.5
    z = depths.astype(np.float32)
    z = z - z.mean()
    if plane == "xy":
        projected = centered.copy()
        title = "X-Y skeleton"
    elif plane == "yz":
        if depth_extent_m <= 0:
            raise ValueError("depth_extent_m must be positive")
        projected = np.column_stack((-z / depth_extent_m, centered[:, 1]))
        title = "Y-Z skeleton"
    else:
        raise ValueError(f"Unsupported projection plane: {plane}")
    projected *= size * 0.72
    projected += size / 2

    for left, right in _SKELETON_EDGES:
        if (left < len(projected) and right < len(projected)
                and valid_mask[left] and valid_mask[right]):
            start = tuple(projected[left].astype(int))
            end = tuple(projected[right].astype(int))
            cv2.line(canvas, start, end, (80, 190, 255), 3)
    for index in np.flatnonzero(valid_mask):
        center = tuple(projected[index].astype(int))
        cv2.circle(canvas, center, 6, (80, 255, 120), -1)
        cv2.circle(canvas, center, 6, (230, 230, 230), 1)

    cv2.putText(canvas, title, (16, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (240, 240, 240), 1,
                cv2.LINE_AA)
    axis_label = "x horizontal / y vertical" if plane == "xy" else \
        "-z horizontal / y vertical"
    cv2.putText(canvas, axis_label, (16, size - 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1,
                cv2.LINE_AA)
    return canvas


def run_video(path: Optional[str], onnx_model: str,
              pose_mode: str = "lightweight", pose_device: str = "auto",
              camera_index: int = 0) -> None:
    import cv2
    import onnxruntime as ort

    use_webcam = path is None or not Path(path).is_file()
    source = camera_index if use_webcam else path
    if path is not None and use_webcam:
        print(f"Input video not found: {path}; using webcam {camera_index}.")
    elif use_webcam:
        print(f"Using webcam {camera_index}.")
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {'webcam' if use_webcam else path}")
    detector = RTMLibPoseDetector(pose_mode, pose_device)
    session = ort.InferenceSession(
        onnx_model,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    filters: Dict[int, OneEuroFilter] = {}
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        timestamp = time.monotonic()
        points, valid = detector.detect(frame)
        if not valid.any():
            for f in filters.values():
                f.reset()
            display = cv2.resize(frame, (640, 480))
            skeletons = np.full((480, 960, 3), 28, dtype=np.uint8)
            cv2.putText(skeletons, "No pose detected", (360, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (220, 220, 220), 1,
                        cv2.LINE_AA)
            cv2.imshow("Root-relative depth inference",
                       np.hstack((display, skeletons)))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        safe_points = points.copy()
        valid_root = valid[7] and valid[8]
        if valid_root:
            safe_points[~valid] = (safe_points[7] + safe_points[8]) / 2
        else:
            safe_points[~valid] = 0.5
        # This baseline uses the full frame as a valid crop. Production
        # deployment should replace it with the training-time person box.
        image = cv2.resize(rgb, (224, 224)).astype(np.float32) / 255
        image = image.transpose(2, 0, 1)[None, ...]
        depth = session.run(
            ["root_relative_depth"],
            {"images": image, "keypoints_2d": safe_points[None, ...]},
        )[0][0, :, 0]
        filtered_depth = np.empty_like(depth)
        for joint, value in enumerate(depth.tolist()):
            filtered_depth[joint] = filters.setdefault(
                joint, OneEuroFilter()).filter(value, timestamp, track_id=joint)
        display = cv2.resize(frame, (640, 480))
        xy_skeleton = _draw_skeleton_projection(
            points, filtered_depth, "xy", valid_mask=valid
        )
        yz_skeleton = _draw_skeleton_projection(
            points, filtered_depth, "yz", valid_mask=valid
        )
        cv2.imshow("Root-relative depth inference",
                   np.hstack((display, xy_skeleton, yz_skeleton)))
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    if use_webcam:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("video", nargs="?", default=None)
    parser.add_argument("--onnx", default="depth_lifting.onnx")
    parser.add_argument("--pose-mode", choices=("lightweight", "balanced",
                                                 "performance"),
                        default="lightweight")
    parser.add_argument("--pose-device", choices=("auto", "cuda", "cpu"),
                        default="auto")
    parser.add_argument("--camera-index", type=int, default=0)
    args = parser.parse_args()
    run_video(args.video, args.onnx, args.pose_mode, args.pose_device,
              args.camera_index)
