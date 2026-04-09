package com.facerecog.ui

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.RectF
import android.os.Bundle
import android.util.Log
import android.view.View
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.facerecog.camera.CameraManager
import com.facerecog.databinding.ActivityMainBinding
import com.facerecog.detection.FaceDetection
import com.facerecog.detection.FaceDetector
import com.facerecog.liveness.AntiSpoofChecker
import com.facerecog.liveness.LivenessChecker
import com.facerecog.liveness.LivenessState
import kotlinx.coroutines.*
import java.util.concurrent.atomic.AtomicBoolean

import com.facerecog.recognition.FaceRecognizer
import com.facerecog.recognition.FaceMath
import com.facerecog.db.AppDatabase
import com.facerecog.db.UserFace
import com.google.gson.Gson

class MainActivity : AppCompatActivity() {
    private lateinit var faceRecognizer: FaceRecognizer
    private var registeredFaces: List<Pair<String, FloatArray>> = emptyList()

    private lateinit var binding: ActivityMainBinding
    private lateinit var cameraManager: CameraManager
    private lateinit var detector: FaceDetector
    private lateinit var livenessChecker: LivenessChecker
    private lateinit var antiSpoofChecker: AntiSpoofChecker

    private val isCapturing = AtomicBoolean(false)
    private val scope = CoroutineScope(Dispatchers.Main + SupervisorJob())

    companion object {
        private const val TAG = "MainActivity"
        private const val CAMERA_PERM_REQUEST = 100
        private const val FRAME_INTERVAL_MS   = 100L  // process ~10 fps
    }

    private var lastFrameTime = 0L
    private var lastBitmap: Bitmap? = null
    private var lastDetection: FaceDetection? = null
    @Volatile private var mainCameraStarted = false

    private fun onUi(block: () -> Unit) {
        if (isFinishing || isDestroyed) return
        if (Thread.currentThread() === mainLooper.thread) block() else runOnUiThread(block)
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        binding.btnRegister.setOnClickListener {
            stopMainCamera()
            startActivity(Intent(this, RegisterActivity::class.java))
        }
        binding.btnRetry.setOnClickListener { resetCycle() }

        if (hasCameraPermission()) initComponents()
        else ActivityCompat.requestPermissions(
            this, arrayOf(Manifest.permission.CAMERA), CAMERA_PERM_REQUEST
        )
    }

    private fun initComponents() {
        detector         = FaceDetector(this)
        livenessChecker  = LivenessChecker(this)
        antiSpoofChecker = AntiSpoofChecker(this)
        faceRecognizer   = FaceRecognizer(this)
        cameraManager    = CameraManager(this, this)

        detector.logModelShapes()  // verify tensor shapes in Logcat on first run

        startMainCamera()

        setStatus("Look at the camera and blink", StatusType.INFO)
    }

    private fun loadUsersFromDatabase() {
        scope.launch(Dispatchers.IO) {
            val db = AppDatabase.getDatabase(this@MainActivity)
            val users = db.faceDao().getAllUsers()
            val faces = users.map { 
                it.name to Gson().fromJson(it.embedding, FloatArray::class.java)
            }
            withContext(Dispatchers.Main) {
                registeredFaces = faces
                Log.d(TAG, "Loaded ${faces.size} users from DB")
            }
        }
    }

    private fun startMainCamera() {
        if (!::cameraManager.isInitialized || mainCameraStarted) return
        cameraManager.startCamera(binding.previewView) { bitmap -> processFrame(bitmap) }
        mainCameraStarted = true
    }

    private fun stopMainCamera() {
        if (!::cameraManager.isInitialized || !mainCameraStarted) return
        cameraManager.stopCamera()
        mainCameraStarted = false
    }

    override fun onStart() {
        super.onStart()
        if (::cameraManager.isInitialized && hasCameraPermission()) {
            startMainCamera()
        }
    }

    override fun onResume() {
        super.onResume()
        loadUsersFromDatabase()
    }

    override fun onStop() {
        stopMainCamera()
        super.onStop()
    }

    private fun processFrame(bitmap: Bitmap) {
        val now = System.currentTimeMillis()
        if (now - lastFrameTime < FRAME_INTERVAL_MS) return
        if (isCapturing.get()) return
        lastFrameTime = now

        // 1. Detect face
        val detections = detector.detect(bitmap)
        onUi {
            binding.overlayView.setSourceDimensions(bitmap.width, bitmap.height)
            binding.overlayView.updateDetections(detections)
        }

        if (detections.isEmpty()) {
            if (livenessChecker.state != LivenessState.NO_FACE) {
                livenessChecker.reset()
                onUi { setStatus("No face detected", StatusType.INFO) }
            }
            return
        }

        lastBitmap    = bitmap
        lastDetection = detections[0]

        // 2. Blink liveness
        livenessChecker.processFrame(bitmap)

        onUi {
            when (livenessChecker.state) {
                LivenessState.NO_FACE      -> setStatus("Face found. Hold steady and blink", StatusType.INFO)
                LivenessState.WAITING_BLINK -> {
                    binding.overlayView.livenessConfirmed = false
                    setStatus("Blink to verify liveness", StatusType.INFO)
                }
                LivenessState.BLINK_DETECTED -> {
                    binding.overlayView.livenessConfirmed = true
                    if (isCapturing.compareAndSet(false, true)) {
                        setStatus("Checking...", StatusType.INFO)
                        binding.progressBar.visibility = View.VISIBLE
                        capture()
                    }
                }
            }
        }
    }

    private fun capture() {
        val bitmap    = lastBitmap    ?: return run { isCapturing.set(false) }
        val detection = lastDetection ?: return run { isCapturing.set(false) }

        scope.launch {
            val (aligned, faceCrop) = withContext(Dispatchers.Default) {
                val aligned = detector.alignFace(bitmap, detection)
                // Crop bounding box for anti-spoof (NOT the aligned 112x112)
                val box = detection.boundingBox
                val cx = box.left.toInt().coerceAtLeast(0)
                val cy = box.top.toInt().coerceAtLeast(0)
                val cw = (box.width().toInt()).coerceAtMost(bitmap.width - cx)
                val ch = (box.height().toInt()).coerceAtMost(bitmap.height - cy)
                val crop = if (cw > 0 && ch > 0)
                    Bitmap.createBitmap(bitmap, cx, cy, cw, ch)
                else bitmap
                Pair(aligned, crop)
            }

            // 3. Anti-spoof check (runs on main — quick model)
            val spoofResult = withContext(Dispatchers.Default) {
                antiSpoofChecker.check(faceCrop)
            }

            if (!spoofResult.isReal) {
                onUi {
                    binding.progressBar.visibility = View.GONE
                    binding.btnRetry.visibility    = View.VISIBLE
                    setStatus(
                        "Spoof detected (score: ${"%.2f".format(spoofResult.liveScore)}). Use a real face.",
                        StatusType.ERROR
                    )
                }
                isCapturing.set(false)
                livenessChecker.reset()
                return@launch
            }

            // 4. Send to backend for recognition
            onUi { setStatus("Recognising...", StatusType.INFO) }
            val embedding = faceRecognizer.extractEmbedding(aligned)
            
            var bestMatchName = "Unknown"
            var highestSim = -1f
            for ((name, faceEmbedding) in registeredFaces) {
                val sim = FaceMath.cosineSimilarity(embedding, faceEmbedding)
                if (sim > highestSim) {
                    highestSim = sim
                    if (sim > 0.5f) {
                        bestMatchName = name
                    }
                }
            }
            
            isCapturing.set(false)
            if (highestSim > 0.5f) {
                val msg = "Recognized: $bestMatchName (${String.format("%.2f", highestSim)})"
                onUi {
                    binding.progressBar.visibility = View.GONE
                    binding.tvStatus.text = msg
                    binding.overlayView.markRecognized(bestMatchName)
                }
            } else {
                onUi {
                    binding.progressBar.visibility = View.GONE
                    binding.tvStatus.text = "Unknown Face"
                    binding.overlayView.markUnknown()
                }
            }

            scope.launch {
                delay(2000)
                onUi {
                    binding.overlayView.clearRecognition()
                    resetCycle()
                }
            }
        }
    }

    private fun resetCycle() {
        livenessChecker.reset()
        isCapturing.set(false)
        binding.resultCard.visibility         = View.GONE
        binding.btnRetry.visibility           = View.GONE
        binding.overlayView.livenessConfirmed = false
        setStatus("Look at the camera and blink", StatusType.INFO)
    }

    private enum class StatusType { INFO, SUCCESS, ERROR }

    private fun setStatus(msg: String, type: StatusType) {
        val (text, bg) = when (type) {
            StatusType.SUCCESS -> "#1B5E20" to "#E8F5E9"
            StatusType.ERROR   -> "#B71C1C" to "#FFEBEE"
            StatusType.INFO    -> "#0D47A1" to "#E3F2FD"
        }
        binding.tvStatus.text = msg
        binding.tvStatus.setTextColor(android.graphics.Color.parseColor(text))
        binding.tvStatus.setBackgroundColor(android.graphics.Color.parseColor(bg))
    }

    private fun hasCameraPermission() =
        ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) ==
                PackageManager.PERMISSION_GRANTED

    override fun onRequestPermissionsResult(
        requestCode: Int, permissions: Array<out String>, grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == CAMERA_PERM_REQUEST &&
            grantResults.firstOrNull() == PackageManager.PERMISSION_GRANTED
        ) initComponents()
        else setStatus("Camera permission required", StatusType.ERROR)
    }

    override fun onDestroy() {
        super.onDestroy()
        scope.cancel()
        if (::cameraManager.isInitialized) {
            stopMainCamera()
            cameraManager.shutdown()
        }
        if (::detector.isInitialized)         detector.close()
        if (::livenessChecker.isInitialized)  livenessChecker.close()
        if (::antiSpoofChecker.isInitialized) antiSpoofChecker.close()
    }
}
