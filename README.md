# Fall Detection System for Elderly Care

elderly fall detection project using Raspberry Pi and IMU sensor data. The system processes accelerometer and gyroscope readings, extracts motion features in real time, trains a machine-learning model, and raises alerts when a fall-like event is detected.

This project matches the resume entry:

> Developed ML-based fall recognition using Raspberry Pi and IMU sensor data. Engineered real-time anomaly detection for elderly monitoring applications.

## Key Features

- Reads MPU6050-style IMU streams: `ax, ay, az, gx, gy, gz`.
- Extracts windowed acceleration, gyroscope, jerk, and posture-change features.
- Trains either a supervised classifier when fall labels are available or an anomaly detector when only normal data is available.
- Supports Raspberry Pi serial monitoring from ESP32/Arduino sensor bridges.
- Replays CSV sensor logs for demos without hardware.
- Logs fall alerts to CSV with timestamp, confidence, and detection reason.
- Includes a separate optional webcam/OpenCV pose demo for computer-vision experimentation.

## Tech Stack

- Python
- Raspberry Pi compatible runtime
- MPU6050 / IMU sensor data
- Pandas, NumPy, Scikit-learn
- PySerial for live sensor streams
- OpenCV + MediaPipe for the optional vision demo
- Pytest

## Project Structure

```text
fall_detection/
  imu_features.py   # Windowed IMU feature engineering
  imu_model.py      # Supervised/anomaly ML model wrapper
  simulate_imu.py   # Generates demo IMU training data
  train_imu.py      # Trains fall detection model from CSV
  monitor_imu.py    # Live serial or CSV replay monitoring
  alerts.py         # CSV alert logging
  app.py            # Optional webcam pose-based demo
  detector.py       # Optional pose fall heuristic
tests/
  test_detector.py
  test_imu_features.py
requirements.txt
```

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Demo Without Hardware

Generate sample IMU data:

```powershell
python -m fall_detection.simulate_imu --output data/sample_imu.csv
```

Train the model:

```powershell
python -m fall_detection.train_imu --input data/sample_imu.csv --output models/imu_fall_model.joblib
```

Replay the CSV as a live stream:

```powershell
python -m fall_detection.monitor_imu --model models/imu_fall_model.joblib --csv data/sample_imu.csv
```

Alerts are written to:

```text
alerts/imu_fall_alerts.csv
```

## Raspberry Pi / ESP32 Flow

1. Connect MPU6050 to ESP32 or Arduino.
2. Stream rows over serial in this format:

```text
ax,ay,az,gx,gy,gz
```

Example:

```text
0.02,0.01,1.01,2.4,-1.2,0.8
```

3. On the Raspberry Pi, run:

```powershell
python -m fall_detection.monitor_imu --model models/imu_fall_model.joblib --serial-port COM3
```

On Linux/Raspberry Pi, the port usually looks like `/dev/ttyUSB0` or `/dev/ttyACM0`.

## CSV Format

Training CSV files should include:

```text
timestamp_ms,ax,ay,az,gx,gy,gz,label
```

The `label` column is optional. If labels are present, values such as `1`, `true`, `fall`, or `anomaly` are treated as falls. If labels are not present or only one class exists, the project trains an Isolation Forest anomaly detector.

## Optional Webcam Demo

Use webcam pose detection:

```powershell
python -m fall_detection.app --source 0
```

Use a video file:

```powershell
python -m fall_detection.app --source path\to\video.mp4
```

## Test

```powershell
pytest
```

## Strong Resume Bullet

**Fall Detection System for Elderly Care**: Developed a Raspberry Pi based ML system for elderly fall recognition using MPU6050 IMU accelerometer and gyroscope data. Engineered real-time windowed features for acceleration magnitude, jerk, posture shift, and angular velocity; trained supervised/anomaly detection models with Scikit-learn; and implemented live serial monitoring with alert logging for safety-critical elderly care workflows.

## Notes

This is an academic and prototype-grade monitoring system. Production use would require calibrated hardware placement, larger real-world fall/non-fall datasets, privacy and safety validation, caregiver notification integration, and clinical reliability testing.
