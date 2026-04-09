package com.facerecog.liveness

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarker
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarkerResult
import kotlin.math.sqrt

enum class LivenessState {
    NO_FACE,        // No face visible
    WAITING_BLINK,  // Face found — prompt user to blink
    BLINK_DETECTED, // Blink confirmed — safe to capture
}

class LivenessChecker(context: Context) {

    private val faceLandmarker: FaceLandmarker
    private val earHistory = ArrayDeque<Float>()
    private val EAR_HISTORY_SIZE = 16

    // MediaPipe 478-landmark model eye indices for Eye Aspect Ratio
    private val LEFT_EYE_IDX  = listOf(362, 385, 387, 263, 373, 380)
    private val RIGHT_EYE_IDX = listOf(33,  160, 158, 133, 153, 144)

    private val EAR_CLOSE_THRESHOLD = 0.21f
    private val EAR_OPEN_THRESHOLD  = 0.26f

    var state: LivenessState = LivenessState.NO_FACE
        private set

    /** Gate recognition on this. Only true after a full blink cycle. */
    val blinkConfirmed: Boolean get() = state == LivenessState.BLINK_DETECTED

    init {
        val baseOptions = BaseOptions.builder()
            .setModelAssetPath("face_landmarker.task")
            .build()

        val options = FaceLandmarker.FaceLandmarkerOptions.builder()
            .setBaseOptions(baseOptions)
            .setRunningMode(RunningMode.IMAGE)
            .setNumFaces(1)
            // Lower confidence floors to avoid frequent false NO_FACE states.
            .setMinFaceDetectionConfidence(0.35f)
            .setMinFacePresenceConfidence(0.35f)
            .setMinTrackingConfidence(0.35f)
            .build()

        faceLandmarker = FaceLandmarker.createFromOptions(context, options)
        Log.d(TAG, "LivenessChecker initialised")
    }

    /**
     * Process one camera frame. Call from analysis thread, never UI thread.
     * State transitions:
     *   NO_FACE -> WAITING_BLINK when face appears
     *   WAITING_BLINK -> BLINK_DETECTED when full blink confirmed
     *   BLINK_DETECTED -> stays until reset() is called explicitly
     */
    fun processFrame(bitmap: Bitmap) {
        if (state == LivenessState.BLINK_DETECTED) return // hold until reset

        val mpImage = BitmapImageBuilder(bitmap).build()
        val result: FaceLandmarkerResult = try {
            faceLandmarker.detect(mpImage)
        } catch (e: Exception) {
            Log.e(TAG, "MediaPipe error: ${e.message}")
            return
        }

        if (result.faceLandmarks().isEmpty()) {
            state = LivenessState.NO_FACE
            earHistory.clear()
            return
        }

        state = LivenessState.WAITING_BLINK

        val landmarks = result.faceLandmarks()[0]
        val leftEar  = computeEar(landmarks, LEFT_EYE_IDX)
        val rightEar = computeEar(landmarks, RIGHT_EYE_IDX)
        val ear = (leftEar + rightEar) / 2f

        earHistory.addLast(ear)
        if (earHistory.size > EAR_HISTORY_SIZE) earHistory.removeFirst()

        if (detectBlink()) {
            state = LivenessState.BLINK_DETECTED
            Log.d(TAG, "Blink confirmed. Min EAR=${earHistory.min()} Last EAR=${earHistory.last()}")
        }
    }

    private fun detectBlink(): Boolean {
        if (earHistory.size < 6) return false
        val recent = earHistory.toList().takeLast(10)
        val minEar  = recent.min()
        val lastEar = recent.last()
        // A blink: eye closes (EAR drops) then reopens (EAR recovers)
        return minEar < EAR_CLOSE_THRESHOLD && lastEar > EAR_OPEN_THRESHOLD
    }

    /**
     * Eye Aspect Ratio (Soukupova & Cech 2016).
     * Ratio of vertical eye openness to horizontal width.
     * Open: ~0.30-0.45. Closed: ~0.10-0.15.
     */
    private fun computeEar(
        landmarks: List<com.google.mediapipe.tasks.components.containers.NormalizedLandmark>,
        indices: List<Int>
    ): Float {
        val p = indices.map { landmarks[it] }
        val v1 = dist(p[1].x(), p[1].y(), p[5].x(), p[5].y())
        val v2 = dist(p[2].x(), p[2].y(), p[4].x(), p[4].y())
        val h  = dist(p[0].x(), p[0].y(), p[3].x(), p[3].y())
        return (v1 + v2) / (2f * h + 1e-6f)
    }

    private fun dist(x1: Float, y1: Float, x2: Float, y2: Float): Float {
        val dx = x1 - x2; val dy = y1 - y2
        return sqrt(dx * dx + dy * dy)
    }

    /** Call after each recognition attempt to allow the next cycle. */
    fun reset() {
        state = LivenessState.NO_FACE
        earHistory.clear()
        Log.d(TAG, "LivenessChecker reset")
    }

    fun close() = faceLandmarker.close()

    companion object { private const val TAG = "LivenessChecker" }
}
