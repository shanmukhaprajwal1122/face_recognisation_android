# Face Recognition System — v1.1

Android + Python face recognition with on-device liveness and ArcFace backend.

```
Android: CameraX → SCRFD (detect+NMS) → FaceOverlayView → MediaPipe blink → align 112x112
Backend: FastAPI → InsightFace ArcFace → cosine similarity → SQLite
```

## What changed in v1.1 (all review issues fixed)

| Issue | Fix |
|---|---|
| Missing model + compatibility risk | New `FaceDetector.kt` auto-detects SCRFD vs RetinaFace from output tensor shapes. Logs all shapes on first run. SCRFD recommended (reliable shape). |
| No bounding box overlay | New `FaceOverlayView.kt` — Canvas view draws boxes, corner accents, landmark dots, and confidence %. Green = liveness confirmed, yellow = waiting for blink. |
| Liveness not gating recognition | `LivenessChecker` state machine: `NO_FACE → WAITING_BLINK → BLINK_DETECTED`. Recognition only fires when `blinkConfirmed == true`. Explicit `reset()` after every attempt. |
| NMS missing | `FaceDetector.applyNms()` — IOU-based non-max suppression before returning detections. |
| No error handling UI | Network errors map to user-friendly messages. Timeout, connection refused, and parse errors all shown in status bar. |
| Double detection on backend | Kept with comment explaining it's the safety net for direct API use. Android sends pre-aligned crops; InsightFace still runs detection internally which is acceptable. |

---

## Quick start

### Backend

```bash
cd backend
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Swagger UI: http://localhost:8000/docs

### Android

**Step 1 — Download model files** into `android-app/app/src/main/assets/`:

```bash
# face_landmarker.task (MediaPipe — no conversion needed)
curl -L -o face_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task

# scrfd_500m_kps.tflite — export from:
# https://github.com/deepinsight/insightface/tree/master/detection/scrfd
```

**Step 2 — Set server IP** in `ApiClient.kt`:
```kotlin
var BASE_URL = "http://10.0.2.2:8000"  // emulator
// var BASE_URL = "http://192.168.x.x:8000"  // real device on LAN
```

**Step 3 — Verify model loaded correctly** (Logcat on first run):
```
FaceDetector: Output[0] shape: [1, 896, 4]
FaceDetector: Output[1] shape: [1, 896, 1]
FaceDetector: Output[2] shape: [1, 896, 10]
FaceDetector: Detected model type: SCRFD
```
If shapes differ, see `FaceDetector.kt` — the postProcess methods handle both SCRFD and RetinaFace formats.

**Step 4 — Build:**
```bash
cd android-app
./gradlew assembleDebug
```

---

## Project structure

```
face-recognition-system/
├── android-app/app/src/main/
│   ├── java/com/facerecog/
│   │   ├── camera/          CameraManager.kt
│   │   ├── detection/       FaceDetector.kt       ← SCRFD+RetinaFace, NMS, auto shape detect
│   │   ├── liveness/        LivenessChecker.kt    ← proper state machine, blink gate
│   │   ├── network/         ApiClient.kt
│   │   └── ui/
│   │       ├── FaceOverlayView.kt                 ← Canvas bounding box overlay
│   │       ├── MainActivity.kt
│   │       └── RegisterActivity.kt
│   └── assets/              ← place model files here (see assets/README.md)
│
├── backend/
│   ├── app/
│   │   ├── main.py          FastAPI routes
│   │   ├── recognition.py   ArcFace engine
│   │   ├── db.py            SQLite
│   │   └── models.py        Pydantic schemas
│   ├── tests/               Unit tests (pytest)
│   └── scripts/             threshold_test.py
```

---

## API

| Method | Route | Description |
|--------|-------|-------------|
| GET | `/health` | Health check |
| GET | `/users` | List users |
| POST | `/register/{id}?name=X` | Register (3–10 images) |
| POST | `/recognize` | Identify face |
| DELETE | `/users/{id}` | Delete user |

### Register (curl)
```bash
curl -X POST "http://localhost:8000/register/u001?name=Ravi" \
  -F "files=@face1.jpg" -F "files=@face2.jpg" \
  -F "files=@face3.jpg" -F "files=@face4.jpg" \
  -F "files=@face5.jpg"
```

### Recognize (curl)
```bash
curl -X POST "http://localhost:8000/recognize" -F "file=@test.jpg"
# {"matched":true,"user_id":"u001","name":"Ravi","confidence":0.6821}
```

---

## Threshold tuning

```bash
cd backend
python scripts/threshold_test.py
# Prints cross-user and same-user similarity scores
# Set THRESHOLD in .env just above the max cross-user score
```

For Indian faces: expected threshold range 0.42–0.52.

---

## Registration tips (especially important for Indian faces)

Capture 7 images per user covering:
- Frontal, slight left turn, slight right turn
- With glasses and without (if applicable)
- Bright light, dim light, indoor fluorescent
- Neutral expression, slight smile

This improves accuracy more than any model change for small deployments (<1000 users).

---

## Tests

```bash
cd backend && pytest tests/ -v
```

---

## Environment variables

| Variable | Default | Notes |
|----------|---------|-------|
| `THRESHOLD` | `0.45` | Tune with threshold_test.py |
| `MIN_IMAGES` | `3` | Min images for registration |
| `DB_PATH` | `faces.db` | SQLite path |
