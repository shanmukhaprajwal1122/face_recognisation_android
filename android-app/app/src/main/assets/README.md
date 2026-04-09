# Model Assets — INCLUDED ✅

All three models are bundled in this directory.

| File | Size | Verified |
|------|------|---------|
| `retinaface.tflite` | 16.2 MB | ✅ Valid TFL3 format, 9 output tensors |
| `silent_face_antispoof.tflite` | 11.6 MB | ✅ Input [1,128,128,3] Output [1,2] |
| `face_landmarker.task` | 3.5 MB | ✅ MediaPipe 478-landmark bundle |

## retinaface.tflite — confirmed specs
- Input:  `[1, 640, 640, 3]`  normalised to `[-1, 1]`
- Output: 9 tensors across 3 scales (stride 8/16/32):
  - `[1, 12800, 2]` cls  `[1, 12800, 4]` bbox  `[1, 12800, 10]` landmarks
  - `[1,  3200, 2]` cls  `[1,  3200, 4]` bbox  `[1,  3200, 10]` landmarks
  - `[1,   800, 2]` cls  `[1,   800, 4]` bbox  `[1,   800, 10]` landmarks
- Total anchors: 16800
- FaceDetector.kt is written for exactly this format.

## silent_face_antispoof.tflite — confirmed specs
- Input:  `[1, 128, 128, 3]`  normalised to `[0, 1]`
- Output: `[1, 2]` → `[spoof_score, live_score]`
- `output[1] > 0.5` = real face, `< 0.5` = spoof/photo
- AntiSpoofChecker.kt is written for exactly this format.

## face_landmarker.task — confirmed
- MediaPipe Face Landmarker, 478 landmarks
- Used by LivenessChecker.kt for blink (EAR) detection
- No conversion needed — loaded directly by MediaPipe SDK.

## On first run — verify in Logcat:
```
FaceDetector: Model has 9 output tensors
FaceDetector: Scale count=12800 → cls=? box=? ldm=?
FaceDetector: Scale count=3200  → cls=? box=? ldm=?
FaceDetector: Scale count=800   → cls=? box=? ldm=?
AntiSpoofChecker: loaded. Input: [1,128,128,3] Output: [1,2]
```
