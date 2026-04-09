package com.facerecog.detection

import android.content.Context
import android.graphics.*
import android.util.Log
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import kotlin.math.max
import kotlin.math.min

data class FaceDetection(
    val boundingBox: RectF,
    /** 5 landmarks: left eye, right eye, nose, left mouth, right mouth */
    val landmarks: Array<PointF>,
    val confidence: Float
) {
    override fun equals(other: Any?) = other is FaceDetection && boundingBox == other.boundingBox
    override fun hashCode() = boundingBox.hashCode()
}

/**
 * RetinaFace MobileNet TFLite face detector.
 *
 * Confirmed model specs (from uploaded retinaface.tflite):
 *   Input:   [1, 640, 640, 3]   float32, normalised to [-1, 1]
 *   Outputs: 9 tensors across 3 scales:
 *     Output[0] [1, 12800, 2]  — stride-8  classification scores
 *     Output[1] [1,  3200, 2]  — stride-16 classification scores
 *     Output[2] [1,   800, 2]  — stride-32 classification scores
 *     Output[3] [1, 12800, 4]  — stride-8  bounding box offsets
 *     Output[4] [1,  3200, 4]  — stride-16 bounding box offsets
 *     Output[5] [1,   800, 4]  — stride-32 bounding box offsets
 *     Output[6] [1, 12800,10]  — stride-8  landmark offsets (5 pts × 2)
 *     Output[7] [1,  3200,10]  — stride-16 landmark offsets
 *     Output[8] [1,   800,10]  — stride-32 landmark offsets
 *
 * NOTE: actual output index order may vary by export tool.
 * On first run check Logcat tag "FaceDetector" to see all tensor shapes,
 * then verify SCALE_CONFIG matches.
 */
class FaceDetector(context: Context, modelFile: String = "retinaface.tflite") {

    private val interpreter: Interpreter
    private val interpreterLock = Any()
    private val INPUT_SIZE  = 640
    private val CONF_THRESH = 0.5f
    private val NMS_THRESH  = 0.4f
    @Volatile private var isClosed = false
    @Volatile private var inferenceDisabled = false

    // Anchor generation config matching the 640x640 RetinaFace MobileNet
    // stride → (grid_size, anchor_sizes)
    private val STRIDES = listOf(8, 16, 32)
    private val ANCHOR_SIZES = mapOf(8 to listOf(16,32), 16 to listOf(64,128), 32 to listOf(256,512))

    // Pre-computed anchors [cx, cy, w, h] in pixel space for 640x640
    private val anchors: Array<FloatArray> = generateAnchors()

    // Output tensor indices — we auto-detect these in initOutputMap()
    // but the default order matches the standard RetinaFace export
    private data class ScaleOutputs(val clsIdx: Int, val boxIdx: Int, val ldmIdx: Int, val count: Int)
    private lateinit var scaleOutputs: List<ScaleOutputs>

    init {
        val opts = Interpreter.Options().apply {
            numThreads = 2
            // Some devices crash in XNNPACK delegate reshape paths for this model.
            setUseXNNPACK(false)
        }
        interpreter = try {
            Interpreter(FileUtil.loadMappedFile(context, modelFile), opts)
        } catch (e: Exception) {
            Log.e(TAG, "Failed to load $modelFile: ${e.message}")
            throw e
        }
        configureInterpreterIO()
        initOutputMap()
    }

    private fun configureInterpreterIO() {
        synchronized(interpreterLock) {
            interpreter.resizeInput(0, intArrayOf(1, INPUT_SIZE, INPUT_SIZE, 3))
            interpreter.allocateTensors()
        }
    }

    /**
     * Log all tensor shapes. Check Logcat on first run to verify
     * the model loaded with the expected shapes.
     */
    fun logModelShapes() {
        if (isClosed) return
        Log.d(TAG, "=== INPUT TENSORS ===")
        for (i in 0 until interpreter.inputTensorCount) {
            val t = interpreter.getInputTensor(i)
            Log.d(TAG, "  Input[$i]: shape=${t.shape().toList()} dtype=${t.dataType()}")
        }
        Log.d(TAG, "=== OUTPUT TENSORS (${interpreter.outputTensorCount} total) ===")
        for (i in 0 until interpreter.outputTensorCount) {
            val t = interpreter.getOutputTensor(i)
            Log.d(TAG, "  Output[$i]: shape=${t.shape().toList()} dtype=${t.dataType()}")
        }
    }

    /**
     * Auto-detect which output index corresponds to which scale and head type
     * by inspecting the second dimension of each output tensor.
     */
    private fun initOutputMap() {
        synchronized(interpreterLock) {
            val numOutputs = interpreter.outputTensorCount
            Log.d(TAG, "Model has $numOutputs output tensors")

            // Group output indices by anchor count for rank-2 ([N, C]) or rank-3 ([1, N, C]).
            val byAnchorCount = mutableMapOf<Int, MutableList<Pair<Int, Int>>>()
            for (i in 0 until numOutputs) {
                val shape = interpreter.getOutputTensor(i).shape()
                val count = when (shape.size) {
                    3 -> shape[1]
                    2 -> shape[0]
                    else -> -1
                }
                val dim2 = when (shape.size) {
                    3 -> shape[2]
                    2 -> shape[1]
                    else -> -1
                }
                if (count > 0 && dim2 > 0) {
                    byAnchorCount.getOrPut(count) { mutableListOf() }.add(Pair(i, dim2))
                }
                Log.d(TAG, "  Output[$i]: shape=${shape.toList()}")
            }

            // For each anchor count, find cls(dim2=1 or 2), box(dim2=4), ldm(dim2=10)
            val anchorCounts = listOf(12800, 3200, 800, 896, 512, 1024, 2048, 4096)
            val scales = mutableListOf<ScaleOutputs>()

            for (count in anchorCounts) {
                val outputs = byAnchorCount[count] ?: continue
                val cls = outputs.firstOrNull { it.second == 1 || it.second == 2 }?.first ?: -1
                val box = outputs.firstOrNull { it.second == 4 }?.first ?: -1
                val ldm = outputs.firstOrNull { it.second == 10 }?.first ?: -1

                if (cls >= 0 && box >= 0) {
                    scales.add(ScaleOutputs(cls, box, ldm, count))
                    Log.d(TAG, "Scale count=$count → cls=$cls box=$box ldm=$ldm")
                }
            }

            if (scales.isNotEmpty()) {
                if (scales.size == 1) {
                    Log.d(TAG, "Detected model type: SCRFD (Combined Scales)")
                } else {
                    Log.d(TAG, "Detected model type: RetinaFace (Multi-Scale)")
                }
            }

            if (scales.isEmpty()) {
                // Fallback: assume fixed order [cls0,cls1,cls2, box0,box1,box2, ldm0,ldm1,ldm2]
                Log.w(TAG, "Could not auto-detect scale order, using fallback fixed ordering")
                scales.addAll(listOf(
                    ScaleOutputs(0, 3, 6, 12800),
                    ScaleOutputs(1, 4, 7, 3200),
                    ScaleOutputs(2, 5, 8, 800)
                ))
            }

            scaleOutputs = scales
            Log.d(TAG, "Anchors generated: ${anchors.size}")
        }
    }

    /**
     * Detect faces in a bitmap. Returns at most 1 detection (front camera).
     */
    fun detect(bitmap: Bitmap): List<FaceDetection> {
        if (isClosed || inferenceDisabled) return emptyList()

        val input = preprocessBitmap(bitmap)
        val scaleX = bitmap.width.toFloat()  / INPUT_SIZE
        val scaleY = bitmap.height.toFloat() / INPUT_SIZE

        val outputs = HashMap<Int, Any>()
        val outputShapes = HashMap<Int, IntArray>()

        synchronized(interpreterLock) {
            if (isClosed || inferenceDisabled) return emptyList()
            val numOutputs = interpreter.outputTensorCount
            for (i in 0 until numOutputs) {
                val shape = interpreter.getOutputTensor(i).shape()
                outputShapes[i] = shape
                outputs[i] = when (shape.size) {
                    3 -> Array(shape[0]) { Array(shape[1]) { FloatArray(shape[2]) } }
                    2 -> Array(shape[0]) { FloatArray(shape[1]) }
                    else -> FloatArray(shape.fold(1) { acc, d -> acc * d })
                }
            }

            try {
                interpreter.runForMultipleInputsOutputs(arrayOf(input), outputs)
            } catch (e: Exception) {
                Log.e(TAG, "Inference failed: ${e.message}")
                if (e is IllegalArgumentException) {
                    // Disable further runs in this session when delegate/tensor state becomes invalid.
                    inferenceDisabled = true
                }
                return emptyList()
            }
        }

        return postProcess(outputs, outputShapes, scaleX, scaleY)
    }

    @Suppress("UNCHECKED_CAST")
    private fun postProcess(
        outputs: HashMap<Int, Any>,
        outputShapes: HashMap<Int, IntArray>,
        scaleX: Float, scaleY: Float
    ): List<FaceDetection> {
        val candidates = mutableListOf<FaceDetection>()
        var anchorOffset = 0

        for (scale in scaleOutputs) {
            val clsCount = outputShapes[scale.clsIdx].rowCount()
            val boxCount = outputShapes[scale.boxIdx].rowCount()
            val ldmCount = if (scale.ldmIdx >= 0) outputShapes[scale.ldmIdx].rowCount() else 0
            // Some exports expose cls as a broadcast scalar [1,1] per scale.
            val clsBroadcast = clsCount == 1 && scale.count > 1
            val maxRows = minOf(
                scale.count,
                boxCount,
                if (scale.ldmIdx >= 0) ldmCount else Int.MAX_VALUE,
                if (clsBroadcast) Int.MAX_VALUE else clsCount
            )

            if (maxRows <= 0) {
                anchorOffset += scale.count
                continue
            }

            for (i in 0 until maxRows) {
                val anchorIdx = anchorOffset + i
                if (anchorIdx >= anchors.size) break

                val clsRow = getRow(outputs[scale.clsIdx], if (clsBroadcast) 0 else i) ?: continue
                val boxRow = getRow(outputs[scale.boxIdx], i) ?: continue
                if (boxRow.size < 4) continue

                val faceScore = decodeFaceScore(clsRow)

                if (faceScore < CONF_THRESH) continue

                val anchor = anchors[anchorIdx]
                val cx = anchor[0]; val cy = anchor[1]
                val aw = anchor[2]; val ah = anchor[3]

                // Box decode: [dx, dy, dw, dh] → [x1, y1, x2, y2]
                val predCx = cx + boxRow[0] * aw
                val predCy = cy + boxRow[1] * ah
                val predW  = aw * Math.exp(boxRow[2].toDouble()).toFloat()
                val predH  = ah * Math.exp(boxRow[3].toDouble()).toFloat()

                val x1 = (predCx - predW / 2) * scaleX
                val y1 = (predCy - predH / 2) * scaleY
                val x2 = (predCx + predW / 2) * scaleX
                val y2 = (predCy + predH / 2) * scaleY

                // Landmark decode: 5 points
                val landmarks = if (scale.ldmIdx >= 0) {
                    val ldmRow = getRow(outputs[scale.ldmIdx], i)
                    if (ldmRow == null || ldmRow.size < 10) {
                        estimateLandmarks(x1, y1, x2, y2)
                    } else {
                    Array(5) { j ->
                        PointF(
                            (cx + ldmRow[j * 2]     * aw) * scaleX,
                            (cy + ldmRow[j * 2 + 1] * ah) * scaleY
                        )
                    }
                    }
                } else estimateLandmarks(x1, y1, x2, y2)

                candidates.add(FaceDetection(RectF(x1, y1, x2, y2), landmarks, faceScore))
            }
            anchorOffset += scale.count
        }

        return applyNms(candidates)
    }

    @Suppress("UNCHECKED_CAST")
    private fun getRow(tensorData: Any?, row: Int): FloatArray? {
        return when (tensorData) {
            is Array<*> -> {
                val first = tensorData.firstOrNull() ?: return null
                when (first) {
                    is Array<*> -> {
                        val batch = tensorData as? Array<Array<FloatArray>> ?: return null
                        if (batch.isEmpty()) null else batch[0].getOrNull(row)
                    }
                    is FloatArray -> {
                        val arr2d = tensorData as? Array<FloatArray>
                        arr2d?.getOrNull(row)
                    }
                    else -> null
                }
            }
            else -> null
        }
    }

    private fun decodeFaceScore(clsRow: FloatArray): Float {
        return when (clsRow.size) {
            0 -> 0f
            1 -> {
                // Single-channel heads are commonly logits; sigmoid is safe for both logits/probabilities.
                val x = clsRow[0]
                (1f / (1f + kotlin.math.exp(-x))).coerceIn(0f, 1f)
            }
            else -> {
                val e0 = Math.exp(clsRow[0].toDouble())
                val e1 = Math.exp(clsRow[1].toDouble())
                (e1 / (e0 + e1)).toFloat().coerceIn(0f, 1f)
            }
        }
    }

    private fun IntArray?.rowCount(): Int {
        if (this == null) return 0
        return when (size) {
            3 -> this[1]
            2 -> this[0]
            else -> 0
        }
    }

    // ─── Anchor generation ────────────────────────────────────────────────────

    private fun generateAnchors(): Array<FloatArray> {
        val result = mutableListOf<FloatArray>()
        for (stride in STRIDES) {
            val gridSize = INPUT_SIZE / stride
            val sizes = ANCHOR_SIZES[stride] ?: continue
            for (row in 0 until gridSize) {
                for (col in 0 until gridSize) {
                    val cx = (col + 0.5f) * stride
                    val cy = (row + 0.5f) * stride
                    for (size in sizes) {
                        result.add(floatArrayOf(cx, cy, size.toFloat(), size.toFloat()))
                    }
                }
            }
        }
        return result.toTypedArray()
    }

    // ─── Pre-processing ───────────────────────────────────────────────────────

    private fun preprocessBitmap(bitmap: Bitmap): Array<Array<Array<FloatArray>>> {
        val resized = Bitmap.createScaledBitmap(bitmap, INPUT_SIZE, INPUT_SIZE, true)
        val result = Array(1) { Array(INPUT_SIZE) { Array(INPUT_SIZE) { FloatArray(3) } } }
        val pixels = IntArray(INPUT_SIZE * INPUT_SIZE)
        resized.getPixels(pixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE)
        for (y in 0 until INPUT_SIZE) {
            for (x in 0 until INPUT_SIZE) {
                val px = pixels[y * INPUT_SIZE + x]
                result[0][y][x][0] = (Color.red(px)   - 127.5f) / 128f
                result[0][y][x][1] = (Color.green(px) - 127.5f) / 128f
                result[0][y][x][2] = (Color.blue(px)  - 127.5f) / 128f
            }
        }
        if (!resized.isRecycled && resized != bitmap) resized.recycle()
        return result
    }

    // ─── NMS ─────────────────────────────────────────────────────────────────

    private fun applyNms(detections: List<FaceDetection>): List<FaceDetection> {
        if (detections.isEmpty()) return emptyList()
        val sorted = detections.sortedByDescending { it.confidence }
        val kept = mutableListOf<FaceDetection>()
        val suppressed = BooleanArray(sorted.size)
        for (i in sorted.indices) {
            if (suppressed[i]) continue
            kept.add(sorted[i])
            for (j in i + 1 until sorted.size) {
                if (!suppressed[j] && iou(sorted[i].boundingBox, sorted[j].boundingBox) > NMS_THRESH)
                    suppressed[j] = true
            }
        }
        return kept.take(1) // front camera: only highest-confidence face
    }

    private fun iou(a: RectF, b: RectF): Float {
        val il = max(a.left, b.left); val it = max(a.top, b.top)
        val ir = min(a.right, b.right); val ib = min(a.bottom, b.bottom)
        val inter = max(0f, ir - il) * max(0f, ib - it)
        val union = a.width() * a.height() + b.width() * b.height() - inter
        return if (union <= 0f) 0f else inter / union
    }

    // ─── Face alignment ───────────────────────────────────────────────────────

    /**
     * Affine warp to ArcFace standard 112x112 aligned crop.
     * MUST be called before sending to backend — critical for accuracy.
     * Standard eye positions for 112x112: left=(38.29,51.70) right=(73.53,51.50)
     */
    fun alignFace(bitmap: Bitmap, detection: FaceDetection): Bitmap {
        val srcL = detection.landmarks[0]; val srcR = detection.landmarks[1]
        val matrix = Matrix()
        matrix.setPolyToPoly(
            floatArrayOf(srcL.x, srcL.y, srcR.x, srcR.y), 0,
            floatArrayOf(38.2946f, 51.6963f, 73.5318f, 51.5014f), 0, 2
        )
        val aligned = Bitmap.createBitmap(112, 112, Bitmap.Config.ARGB_8888)
        Canvas(aligned).drawBitmap(bitmap, matrix, null)
        return aligned
    }

    private fun estimateLandmarks(x1: Float, y1: Float, x2: Float, y2: Float): Array<PointF> {
        val w = x2 - x1; val h = y2 - y1
        return arrayOf(
            PointF(x1 + w*0.30f, y1 + h*0.40f),
            PointF(x1 + w*0.70f, y1 + h*0.40f),
            PointF(x1 + w*0.50f, y1 + h*0.60f),
            PointF(x1 + w*0.35f, y1 + h*0.80f),
            PointF(x1 + w*0.65f, y1 + h*0.80f)
        )
    }

    fun close() {
        synchronized(interpreterLock) {
            if (isClosed) return
            isClosed = true
            interpreter.close()
        }
    }

    companion object { private const val TAG = "FaceDetector" }
}
