from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from time import monotonic

import cv2

from .alerts import AlertLogger
from .detector import BodyObservation, FallDetector, FallEvent, Keypoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Real-time elderly fall detection")
    parser.add_argument("--source", default="0", help="Camera index or video file path")
    parser.add_argument("--alert-log", default="alerts/fall_alerts.csv", help="CSV alert log path")
    parser.add_argument("--min-visibility", type=float, default=0.55, help="Required landmark visibility")
    parser.add_argument("--no-window", action="store_true", help="Run without displaying a video window")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = int(args.source) if args.source.isdigit() else args.source
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video source: {args.source}")

    detector = FallDetector()
    logger = AlertLogger(args.alert_log)

    with _pose_model() as pose:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            event = _process_frame(frame, pose, detector, logger, args.min_visibility)
            if not args.no_window:
                _draw_status(frame, event)
                cv2.imshow("Elderly Fall Detection", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    capture.release()
    cv2.destroyAllWindows()


def _pose_model():
    try:
        import mediapipe as mp
    except ImportError as exc:
        raise RuntimeError("MediaPipe is required. Install dependencies with: pip install -r requirements.txt") from exc

    return mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )


def _process_frame(
    frame,
    pose,
    detector: FallDetector,
    logger: AlertLogger,
    min_visibility: float,
) -> FallEvent | None:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)
    if not results.pose_landmarks:
        return None

    observation = _landmarks_to_observation(results.pose_landmarks.landmark, min_visibility)
    if observation is None:
        return None

    event = detector.update(observation)
    if event.is_fall:
        timestamp = datetime.now().isoformat(timespec="seconds")
        logger.log(timestamp, event)
    return event


def _landmarks_to_observation(landmarks, min_visibility: float) -> BodyObservation | None:
    import mediapipe as mp

    pose = mp.solutions.pose.PoseLandmark
    required = {
        "left_shoulder": pose.LEFT_SHOULDER,
        "right_shoulder": pose.RIGHT_SHOULDER,
        "left_hip": pose.LEFT_HIP,
        "right_hip": pose.RIGHT_HIP,
        "left_ankle": pose.LEFT_ANKLE,
        "right_ankle": pose.RIGHT_ANKLE,
    }

    keypoints: dict[str, Keypoint] = {}
    for name, landmark_id in required.items():
        landmark = landmarks[landmark_id.value]
        visibility = getattr(landmark, "visibility", 1.0)
        if visibility < min_visibility:
            return None
        keypoints[name] = Keypoint(landmark.x, landmark.y, visibility)

    return BodyObservation(timestamp=monotonic(), **keypoints)


def _draw_status(frame, event: FallEvent | None) -> None:
    if event is None:
        label = "No reliable pose"
        color = (0, 180, 255)
    elif event.is_fall:
        label = f"FALL DETECTED  conf={event.confidence:.2f}"
        color = (0, 0, 255)
    else:
        label = f"Monitoring  angle={event.torso_angle:.0f}  ratio={event.aspect_ratio:.2f}"
        color = (40, 190, 80)

    cv2.rectangle(frame, (16, 16), (620, 62), (20, 20, 20), thickness=-1)
    cv2.putText(frame, label, (28, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)


if __name__ == "__main__":
    main()
