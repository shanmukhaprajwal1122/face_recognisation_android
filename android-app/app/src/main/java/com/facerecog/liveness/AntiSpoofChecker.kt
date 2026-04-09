package com.facerecog.liveness

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Color
import android.util.Log
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil

/**
 * Silent-Face Anti-Spoofing checker using the uploaded silent_face_antispoof.tflite.
 *
 * Confirmed model specs:
 *   Input:  [1, 128, 128, 3]  float32, normalised to [0, 1]
 *   Output: [1, 2]            [spoof_score, live_score]
 *                              index[1] > 0.5 → REAL face
 *                              index[1] < 0.5 → SPOOF (photo/screen)
 *
 * This is a second liveness layer on top of MediaPipe blink detection.
 * It catches printed photos and screen replays that pass the blink check.
 *
 * Usage:
 *   val result = antiSpoof.check(faceCropBitmap)
 *   if (result.isReal && livenessChecker.blinkConfirmed) → safe to capture
 */
class AntiSpoofChecker(context: Context) {

    private val interpreter: Interpreter
    private val INPUT_W = 128
    private val INPUT_H = 128
    private val REAL_THRESHOLD = 0.5f  // output[1] > this → real

    data class AntiSpoofResult(
        val isReal: Boolean,
        val liveScore: Float,   // higher = more likely real
        val spoofScore: Float   // higher = more likely spoof
    )

    init {
        val opts = Interpreter.Options().apply { numThreads = 2 }
        interpreter = Interpreter(
            FileUtil.loadMappedFile(context, "silent_face_antispoof.tflite"),
            opts
        )
        Log.d(TAG, "AntiSpoofChecker loaded. Input: [1,128,128,3] Output: [1,2]")
    }

    /**
     * Run anti-spoof check on a face crop.
     * @param faceCrop Any size bitmap — will be resized to 128x128 internally.
     *                 Should be the face bounding box crop (NOT the 112x112 aligned crop).
     */
    fun check(faceCrop: Bitmap): AntiSpoofResult {
        val input = preprocess(faceCrop)
        val output = Array(1) { FloatArray(2) }

        return try {
            interpreter.run(input, output)
            val spoofScore = output[0][0]
            val liveScore  = output[0][1]
            val isReal = liveScore > REAL_THRESHOLD
            Log.d(TAG, "live=${"%.3f".format(liveScore)} spoof=${"%.3f".format(spoofScore)} → ${if (isReal) "REAL" else "SPOOF"}")
            AntiSpoofResult(isReal, liveScore, spoofScore)
        } catch (e: Exception) {
            Log.e(TAG, "AntiSpoof inference failed: ${e.message}")
            // Fail open — don't block the user if the model crashes
            AntiSpoofResult(isReal = true, liveScore = 1f, spoofScore = 0f)
        }
    }

    /**
     * Preprocess: resize to 128x128, normalise to [0, 1].
     * The Silent-Face model was trained with pixel values in [0,1] (not [-1,1]).
     */
    private fun preprocess(bitmap: Bitmap): Array<Array<Array<FloatArray>>> {
        val resized = Bitmap.createScaledBitmap(bitmap, INPUT_W, INPUT_H, true)
        val result = Array(1) { Array(INPUT_H) { Array(INPUT_W) { FloatArray(3) } } }
        val pixels = IntArray(INPUT_W * INPUT_H)
        resized.getPixels(pixels, 0, INPUT_W, 0, 0, INPUT_W, INPUT_H)
        for (y in 0 until INPUT_H) {
            for (x in 0 until INPUT_W) {
                val px = pixels[y * INPUT_W + x]
                result[0][y][x][0] = Color.red(px)   / 255f
                result[0][y][x][1] = Color.green(px) / 255f
                result[0][y][x][2] = Color.blue(px)  / 255f
            }
        }
        if (!resized.isRecycled && resized != bitmap) resized.recycle()
        return result
    }

    fun close() = interpreter.close()

    companion object { private const val TAG = "AntiSpoofChecker" }
}
