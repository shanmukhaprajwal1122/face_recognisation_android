package com.facerecog.ui

import android.graphics.Bitmap
import android.os.Bundle
import android.util.Log
import android.view.View
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import com.facerecog.camera.CameraManager
import com.facerecog.databinding.ActivityRegisterBinding
import com.facerecog.detection.FaceDetection
import com.facerecog.detection.FaceDetector
import kotlinx.coroutines.*
import java.util.UUID
import java.util.concurrent.atomic.AtomicBoolean

import com.facerecog.recognition.FaceRecognizer
import com.facerecog.db.AppDatabase
import com.facerecog.db.UserFace
import com.google.gson.Gson

class RegisterActivity : AppCompatActivity() {
    private lateinit var faceRecognizer: FaceRecognizer

    private lateinit var binding: ActivityRegisterBinding
    private lateinit var cameraManager: CameraManager
    private lateinit var detector: FaceDetector

    private val capturedFaces = mutableListOf<Bitmap>()
    private val isCapturing = AtomicBoolean(false)
    private val scope = CoroutineScope(Dispatchers.Main + SupervisorJob())
    @Volatile private var latestBitmap: Bitmap? = null
    @Volatile private var latestDetection: FaceDetection? = null
    @Volatile private var isSubmitting: Boolean = false
    @Volatile private var registerCameraStarted: Boolean = false

    private fun onUi(block: () -> Unit) {
        if (isFinishing || isDestroyed) return
        if (Thread.currentThread() === mainLooper.thread) block() else runOnUiThread(block)
    }

    companion object {
        const val TARGET_IMAGES = 1
        const val MIN_IMAGES = 1
        private const val MAX_UPLOAD_IMAGES = 5
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityRegisterBinding.inflate(layoutInflater)
        setContentView(binding.root)

        detector     = FaceDetector(this)
        faceRecognizer = FaceRecognizer(this)
        cameraManager = CameraManager(this, this)

        binding.btnCapture.setOnClickListener  { captureFrame() }
        binding.btnRegister.setOnClickListener { submitRegistration() }
        binding.btnBack.setOnClickListener     { finish() }

        updateCaptureCount()
    }

    private fun startRegistrationCamera() {
        if (registerCameraStarted || isSubmitting) return
        cameraManager.startCamera(binding.previewView) { bitmap ->
            val detections = detector.detect(bitmap)
            latestBitmap = bitmap
            latestDetection = detections.firstOrNull()
            onUi {
                if (detections.isEmpty()) {
                    binding.overlayView.clear()
                } else {
                    val detection = detections[0]
                    val rect = android.graphics.Rect(
                        detection.boundingBox.left.toInt(),
                        detection.boundingBox.top.toInt(),
                        detection.boundingBox.right.toInt(),
                        detection.boundingBox.bottom.toInt()
                    )
                    binding.overlayView.setFaceData(rect, "Registering", true, bitmap.width, bitmap.height)
                }
            }
        }
        registerCameraStarted = true
    }

    private fun stopRegistrationCamera() {
        if (!registerCameraStarted) return
        cameraManager.stopCamera()
        registerCameraStarted = false
    }

    override fun onStart() {
        super.onStart()
        startRegistrationCamera()
    }

    override fun onStop() {
        stopRegistrationCamera()
        super.onStop()
    }

    private fun imagesForUpload(): List<Bitmap> {
        if (capturedFaces.size <= MAX_UPLOAD_IMAGES) return capturedFaces.toList()
        return capturedFaces.takeLast(MAX_UPLOAD_IMAGES)
    }

    private fun captureFrame() {
        if (!isCapturing.compareAndSet(false, true)) return
        binding.btnCapture.isEnabled = false

        val bitmap = latestBitmap
        val detection = latestDetection
        if (bitmap == null || detection == null) {
            onUi {
                showToast("No face detected — try better lighting")
                binding.btnCapture.isEnabled = true
            }
            isCapturing.set(false)
            return
        }

        scope.launch(Dispatchers.Default) {
            val aligned = detector.alignFace(bitmap, detection)
            withContext(Dispatchers.Main) {
                capturedFaces.add(aligned)
                isCapturing.set(false)
                updateCaptureCount()
                binding.btnCapture.isEnabled = true
                showToast("Captured ${capturedFaces.size}/$TARGET_IMAGES")

                if (capturedFaces.size >= TARGET_IMAGES) {
                    showToast("All $TARGET_IMAGES images captured — ready to register")
                }
            }
        }
    }

    private fun updateCaptureCount() {
        val count = capturedFaces.size
        binding.tvCaptureCount.text = "$count/$TARGET_IMAGES images captured"
        binding.btnRegister.isEnabled = count >= MIN_IMAGES
        binding.progressCapture.progress = (count * 100) / TARGET_IMAGES

        // Show hint if enough images collected
        binding.tvHint.text = when {
            count == 0         -> "Capture at least $MIN_IMAGES images. Vary lighting and angle."
            count < MIN_IMAGES -> "Need ${MIN_IMAGES - count} more image(s)."
            count < TARGET_IMAGES -> "Good! Capture ${TARGET_IMAGES - count} more for best accuracy."
            else               -> "All images captured. Tap Register."
        }
    }

    private fun submitRegistration() {
        if (isSubmitting) return

        val name = binding.etName.text.toString().trim()
        if (name.isEmpty()) {
            binding.etName.error = "Name is required"
            return
        }

        val uploadImages = imagesForUpload()
        if (uploadImages.size < MIN_IMAGES) {
            showToast("Capture at least $MIN_IMAGES images before register")
            return
        }

        isSubmitting = true

        // Pause camera analysis while upload is running to reduce CPU/memory churn.
        stopRegistrationCamera()

        binding.btnRegister.isEnabled = false
        binding.btnCapture.isEnabled  = false
        binding.progressBar.visibility = View.VISIBLE

        scope.launch(Dispatchers.IO) {
            try {
                val embedding = faceRecognizer.extractEmbedding(capturedFaces.first()!!)
                Log.d("RegisterActivity", "Extracted embedding size: ${embedding.size}")
                // Serialize and save to DB
                val embeddingString = Gson().toJson(embedding)
                Log.d("RegisterActivity", "Serialized embedding length: ${embeddingString.length}")
                val newUser = UserFace(name = name, embedding = embeddingString)
                val database = AppDatabase.getDatabase(this@RegisterActivity)
                database.faceDao().insertUser(newUser)

                withContext(Dispatchers.Main) {
                    isSubmitting = false
                    Toast.makeText(this@RegisterActivity, "Successfully registered face offline!", Toast.LENGTH_LONG).show()
                    finish()
                }
            } catch (e: Exception) {
                withContext(Dispatchers.Main) {
                    isSubmitting = false
                    Toast.makeText(this@RegisterActivity, "Registration failed: ${e.message}", Toast.LENGTH_LONG).show()
                    binding.btnRegister.isEnabled = true
                    binding.btnCapture.isEnabled = true
                    binding.progressBar.visibility = View.GONE
                }
            }
        }
    }

    private fun showToast(msg: String) =
        Toast.makeText(this, msg, Toast.LENGTH_SHORT).show()

    override fun onDestroy() {
        super.onDestroy()
        scope.cancel()
        stopRegistrationCamera()
        cameraManager.shutdown()
        detector.close()
        faceRecognizer.close()
        capturedFaces.forEach { it.recycle() }
        capturedFaces.clear()
    }
}
