package com.facerecog.ui
import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Rect
import android.graphics.RectF
import android.util.AttributeSet
import android.view.View
class FaceOverlayView @JvmOverloads constructor(
    context: Context, attrs: AttributeSet? = null, defStyleAttr: Int = 0
) : View(context, attrs, defStyleAttr) {
    private var faceBounds: Rect? = null
    private var name: String = ""
    private var isReal: Boolean = false
    private var imageWidth: Int = 1
    private var imageHeight: Int = 1
    private val boxPaint = Paint().apply {
        color = Color.GREEN
        style = Paint.Style.STROKE
        strokeWidth = 8f
    }
    private val textPaint = Paint().apply {
        color = Color.WHITE
        textSize = 50f
        style = Paint.Style.FILL
        setShadowLayer(5f, 0f, 0f, Color.BLACK)
    }
    private val bgPaint = Paint().apply {
        color = Color.parseColor("#80000000") // Semi-transparent black
        style = Paint.Style.FILL
    }
    fun setFaceData(bounds: Rect, faceName: String, real: Boolean, imgWidth: Int, imgHeight: Int) {
        faceBounds = bounds
        name = faceName
        isReal = real
        imageWidth = imgWidth
        imageHeight = imgHeight
        // Change box color based on liveness
        boxPaint.color = if (isReal) Color.GREEN else Color.RED
        invalidate() // Trigger redraw
    }
    fun clear() {
        faceBounds = null
        name = ""
        invalidate()
    }
    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        val bounds = faceBounds ?: return
        // Calculate scaling factors
        val scaleX = width.toFloat() / imageWidth.toFloat()
        val scaleY = height.toFloat() / imageHeight.toFloat()
        // Important: For front camera, we usually need to mirror the X coordinates
        // Assuming front camera usage:
        val mappedLeft = width - (bounds.right * scaleX)
        val mappedRight = width - (bounds.left * scaleX)
        val mappedTop = bounds.top * scaleY
        val mappedBottom = bounds.bottom * scaleY
        val mappedBounds = RectF(mappedLeft, mappedTop, mappedRight, mappedBottom)
        // Draw bounding box
        canvas.drawRect(mappedBounds, boxPaint)

        // Draw text background and text
        val displayText = "$name (${if(isReal) "Real" else "Spoof"})"
        val textWidth = textPaint.measureText(displayText)
        
        val textX = mappedBounds.left
        val textY = mappedBounds.top - 20f
        val bgRect = RectF(textX - 10f, textY - textPaint.textSize, textX + textWidth + 10f, textY + 10f)
        canvas.drawRect(bgRect, bgPaint)
        canvas.drawText(displayText, textX, textY, textPaint)
    }
}
