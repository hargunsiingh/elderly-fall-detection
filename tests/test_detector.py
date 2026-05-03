from fall_detection import BodyObservation, FallDetector, Keypoint


def observation(
    *,
    shoulders_y: float,
    hips_y: float,
    left_x: float,
    right_x: float,
    ankle_y: float,
    timestamp: float,
) -> BodyObservation:
    return BodyObservation(
        left_shoulder=Keypoint(left_x, shoulders_y),
        right_shoulder=Keypoint(right_x, shoulders_y),
        left_hip=Keypoint(left_x + 0.01, hips_y),
        right_hip=Keypoint(right_x + 0.01, hips_y),
        left_ankle=Keypoint(left_x, ankle_y),
        right_ankle=Keypoint(right_x, ankle_y),
        timestamp=timestamp,
    )


def test_upright_pose_is_not_fall() -> None:
    detector = FallDetector()
    event = detector.update(
        observation(
            shoulders_y=0.20,
            hips_y=0.52,
            left_x=0.43,
            right_x=0.57,
            ankle_y=0.92,
            timestamp=1.0,
        )
    )

    assert not event.is_fall
    assert event.confidence < 0.55


def test_horizontal_low_pose_is_fall() -> None:
    detector = FallDetector()
    event = detector.update(
        BodyObservation(
            left_shoulder=Keypoint(0.22, 0.70),
            right_shoulder=Keypoint(0.30, 0.70),
            left_hip=Keypoint(0.70, 0.73),
            right_hip=Keypoint(0.78, 0.73),
            left_ankle=Keypoint(0.86, 0.76),
            right_ankle=Keypoint(0.92, 0.76),
            timestamp=1.0,
        )
    )

    assert event.is_fall
    assert "horizontal posture" in event.reason


def test_cooldown_suppresses_repeated_alerts() -> None:
    detector = FallDetector(cooldown_seconds=5.0)
    pose = BodyObservation(
        left_shoulder=Keypoint(0.20, 0.70),
        right_shoulder=Keypoint(0.28, 0.70),
        left_hip=Keypoint(0.70, 0.74),
        right_hip=Keypoint(0.78, 0.74),
        left_ankle=Keypoint(0.86, 0.76),
        right_ankle=Keypoint(0.92, 0.76),
        timestamp=1.0,
    )

    assert detector.update(pose).is_fall
    repeated = detector.update(
        BodyObservation(
            left_shoulder=pose.left_shoulder,
            right_shoulder=pose.right_shoulder,
            left_hip=pose.left_hip,
            right_hip=pose.right_hip,
            left_ankle=pose.left_ankle,
            right_ankle=pose.right_ankle,
            timestamp=2.0,
        )
    )

    assert not repeated.is_fall
    assert "cooldown active" in repeated.reason
