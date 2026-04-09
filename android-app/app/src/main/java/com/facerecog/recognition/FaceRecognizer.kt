package com.facerecog.recognition
import android.content.Context
import android.graphics.Bitmap
import android.graphics.Color
import android.util.Log
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.sqrt
class FaceRecognizer(context: Context) {
    private val interpreter: Interpreter
    init {
        val model = FileUtil.loadMappedFile(context, "mobilefacenet.tflite")
        val options = Interpreter.Options().apply { setNumThreads(4) }
        interpreter = Interpreter(model, options)
    }
    fun extractEmbedding(bitmap: Bitmap): FloatArray {
        val resized = Bitmap.createScaledBitmap(bitmap, 112, 112, true)

        val input = Array(1) { Array(112) { Array(112) { FloatArray(3) } } }

        for (y in 0 until 112) {
            for (x in 0 until 112) {
                val pixel = resized.getPixel(x, y)

                input[0][y][x][0] = (Color.red(pixel) - 127.5f) / 128f
                input[0][y][x][1] = (Color.green(pixel) - 127.5f) / 128f
                input[0][y][x][2] = (Color.blue(pixel) - 127.5f) / 128f
            }
        }

        val outputShape = interpreter.getOutputTensor(0).shape()
        val embeddingSize = outputShape[1] // typically 128 or 512
        Log.d("FaceRecognizer", "Model output embedding size: $embeddingSize")
        
        val outputBuffer = Array(1) { FloatArray(embeddingSize) }
        
        interpreter.run(input, outputBuffer)
        return normalize(outputBuffer[0])
    }

    private fun normalize(embedding: FloatArray): FloatArray {
        var sum = 0f
        for (v in embedding) sum += v * v
        val norm = Math.sqrt(sum.toDouble()).toFloat()

        return embedding.map { it / norm }.toFloatArray()
    }

    companion object {
        fun cosineSimilarity(emb1: FloatArray, emb2: FloatArray): Float {
            var dot = 0.0f
            for (i in emb1.indices) {
                dot += emb1[i] * emb2[i]
            }
            return dot
        }
    }
}
