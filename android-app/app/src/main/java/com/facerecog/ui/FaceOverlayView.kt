package com.facerecog.ui

import android.content.Context
import android.graphics.*
import android.util.AttributeSet
import android.view.View
import com.facerecog.detection.FaceDetection

/**
 * Transparent overlay drawn on top of the CameraX PreviewView.
 * Draws bounding boxes and landmark dots for detected faces.
 *
 * Usage in layout: place this view directly on top of PreviewView
 * with match_parent dimensions.
 *
 * Call updateDetections() from the analysis thread — it posts to the UI thread internally.
 */
class FaceOverlayView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyleAttr: Int = 0
) : View(context, attrs, defStyleAttr) {

    private var detections: List<FaceDetection> = emptyList()
    private var previewWidth: Int = 1
    private var previewHeight: Int = 1
    private var isMirrored: Boolean = false  // Set to false to fix bounding box tracking on opposite side

    // Box paint — green when face detected, yellow when liveness pending
    private val boxPaint = Paint().apply {
        style = Paint.Style.STROKE
        strokeWidth = 3f
        color = Color.parseColor("#4CAF50")
        isAntiAlias = true
    }

    // Corner accent paint (thicker corners for a modern look)
    private val cornerPaint = Paint().apply {
        style = Paint.Style.STROKE
        strokeWidth = 5f
        color = Color.WHITE
        isAntiAlias = true
    }

    // Landmark dot paint
    private val landmarkPaint = Paint().apply {
        style = Paint.Style.FILL
        color = Color.parseColor("#FF9800")
        isAntiAlias = true
    }

    // Confidence label
    private val textPaint = Paint().apply {
        color = Color.WHITE
        textSize = 36f
        isAntiAlias = true
        setShadowLayer(4f, 0f, 2f, Color.BLACK)
    }

    // Background pill for label
    private val labelBgPaint = Paint().apply {
        style = Paint.Style.FILL
        color = Color.parseColor("#CC000000")
    }

    
    private var recognitionName: String? = null
    fun markRecognized(name: String) {
        post {
            recognitionName = name
            boxPaint.color = Color.parseColor("#00FF00") // Green
            invalidate()
        }
    }
    fun markUnknown() {
        post {
            recognitionName = "Unknown"
            boxPaint.color = Color.parseColor("#FF0000") // Red
            invalidate()
        }
    }
    fun clearRecognition() {
        post {
            recognitionName = null
            boxPaint.color = if (livenessConfirmed) Color.parseColor("#4CAF50") else Color.parseColor("#FFC107")
            invalidate()
        }
    }

    var livenessConfirmed: Boolean = false
        set(value) {
            field = value
            boxPaint.color = if (value) Color.parseColor("#4CAF50") else Color.parseColor("#FFC107")
            invalidate()
        }

    /**
     * Call this from the camera analysis thread with the source bitmap dimensions.
     * The overlay scales detection coords to match the view size.
     */
    fun setSourceDimensions(width: Int, height: Int) {
        previewWidth = width
        previewHeight = height
    }

    /**
     * Update displayed detections. Safe to call from any thread.
     */
    fun updateDetections(newDetections: List<FaceDetection>) {
        post {
            detections = newDetections
            invalidate()
        }
    }

    fun clearDetections() {
        post {
            detections = emptyList()
            invalidate()
        }
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        if (detections.isEmpty()) return

        val scaleX = width.toFloat() / previewWidth
        val scaleY = height.toFloat() / previewHeight

        for (detection in detections) {
            val box = detection.boundingBox

            // Scale and mirror coordinates to match preview display
            val left   = if (isMirrored) width - box.right * scaleX else box.left * scaleX
            val right  = if (isMirrored) width - box.left * scaleX  else box.right * scaleX
            val top    = box.top * scaleY
            val bottom = box.bottom * scaleY

            val scaledBox = RectF(left, top, right, bottom)

            // Draw main bounding box
            canvas.drawRoundRect(scaledBox, 12f, 12f, boxPaint)

            // Draw corner accents (top-left, top-right, bottom-left, bottom-right)
            val cornerLen = minOf(scaledBox.width(), scaledBox.height()) * 0.15f
            drawCorners(canvas, scaledBox, cornerLen)

            // Draw landmarks
            for (lm in detection.landmarks) {
                val lx = if (isMirrored) width - lm.x * scaleX else lm.x * scaleX
                val ly = lm.y * scaleY
                canvas.drawCircle(lx, ly, 6f, landmarkPaint)
            }

            // Draw confidence label
            val pct = (detection.confidence * 100).toInt()
            val label = recognitionName ?: if (livenessConfirmed) "✓ $pct%" else "$pct%"
            val labelW = textPaint.measureText(label) + 16f
            val labelH = 48f
            val labelRect = RectF(scaledBox.left, scaledBox.top - labelH, scaledBox.left + labelW, scaledBox.top)
            canvas.drawRoundRect(labelRect, 8f, 8f, labelBgPaint)
            val textY = scaledBox.top - 12f
            canvas.drawText(label, scaledBox.left + 8f, textY, textPaint)
        }
    }

    private fun drawCorners(canvas: Canvas, box: RectF, len: Float) {
        // Top-left
        canvas.drawLine(box.left, box.top, box.left + len, box.top, cornerPaint)
        canvas.drawLine(box.left, box.top, box.left, box.top + len, cornerPaint)
        // Top-right
        canvas.drawLine(box.right - len, box.top, box.right, box.top, cornerPaint)
        canvas.drawLine(box.right, box.top, box.right, box.top + len, cornerPaint)
        // Bottom-left
        canvas.drawLine(box.left, box.bottom - len, box.left, box.bottom, cornerPaint)
        canvas.drawLine(box.left, box.bottom, box.left + len, box.bottom, cornerPaint)
        // Bottom-right
        canvas.drawLine(box.right, box.bottom - len, box.right, box.bottom, cornerPaint)
        canvas.drawLine(box.right - len, box.bottom, box.right, box.bottom, cornerPaint)
    }
}
