import pandas as pd

from fall_detection.imu_features import extract_features, make_feature_table


def test_extract_features_contains_motion_signals() -> None:
    window = pd.DataFrame(
        {
            "ax": [0.0, 0.2, 0.5],
            "ay": [0.0, 0.1, 0.4],
            "az": [1.0, 0.8, 0.2],
            "gx": [0.0, 20.0, 70.0],
            "gy": [0.0, 10.0, 80.0],
            "gz": [0.0, 5.0, 40.0],
        }
    )

    features = extract_features(window)

    assert features["acc_mag_max"] > features["acc_mag_min"]
    assert features["gyro_mag_max"] > 100
    assert features["jerk_max_abs"] > 0
    assert features["posture_az_change"] == 0.8


def test_make_feature_table_labels_windows_with_falls() -> None:
    frame = pd.DataFrame(
        {
            "ax": [0.0, 0.1, 0.2, 0.3],
            "ay": [0.0, 0.1, 0.2, 0.3],
            "az": [1.0, 1.0, 0.6, 0.4],
            "gx": [1.0, 2.0, 90.0, 120.0],
            "gy": [1.0, 2.0, 95.0, 130.0],
            "gz": [1.0, 2.0, 40.0, 60.0],
            "label": [0, 0, "fall", 0],
        }
    )

    table = make_feature_table(frame, window_size=3, step=1)

    assert list(table["label"]) == [1, 1]
