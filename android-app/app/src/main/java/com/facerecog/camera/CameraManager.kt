package com.facerecog.camera

import android.content.Context
import android.graphics.Bitmap
import android.util.Size
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit

class CameraManager(
    private val context: Context,
    private val lifecycleOwner: LifecycleOwner
) {
    private var cameraExecutor: ExecutorService = Executors.newSingleThreadExecutor()
    private var imageAnalysis: ImageAnalysis? = null
    private var preview: Preview? = null
    private var cameraProvider: ProcessCameraProvider? = null
    @Volatile private var isStarted: Boolean = false
    private var lastFrameTimeMs = 0L
    private val frameIntervalMs = 100L

    /**
     * Start the front camera.
     * onFrame is called for every frame — keep it fast, offload heavy work to a background thread.
     */
    fun startCamera(
        previewView: PreviewView,
        onFrame: (Bitmap) -> Unit
    ) {
        if (isStarted) return

        if (cameraExecutor.isShutdown) {
            cameraExecutor = Executors.newSingleThreadExecutor()
        }

        val cameraProviderFuture = ProcessCameraProvider.getInstance(context)

        cameraProviderFuture.addListener({
            val provider = cameraProviderFuture.get()
            cameraProvider = provider

            if (isStarted) return@addListener

            val previousUseCases = listOfNotNull(preview, imageAnalysis).toTypedArray()
            if (previousUseCases.isNotEmpty()) {
                provider.unbind(*previousUseCases)
            }

            val newPreview = Preview.Builder()
                .setTargetResolution(Size(640, 480))
                .build()
                .also { it.setSurfaceProvider(previewView.surfaceProvider) }

            val newAnalysis = ImageAnalysis.Builder()
                .setTargetResolution(Size(640, 480))
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .build()

            newAnalysis.setAnalyzer(cameraExecutor) { imageProxy ->
                try {
                    val now = System.currentTimeMillis()
                    if (now - lastFrameTimeMs >= frameIntervalMs) {
                        lastFrameTimeMs = now
                        val bitmap = imageProxy.toBitmap()
                        onFrame(bitmap)
                    }
                } catch (e: Exception) {
                    android.util.Log.e("CameraManager", "Frame analysis failed", e)
                } finally {
                    imageProxy.close()
                }
            }

            try {
                provider.bindToLifecycle(
                    lifecycleOwner,
                    CameraSelector.DEFAULT_FRONT_CAMERA,
                    newPreview,
                    newAnalysis
                )
                preview = newPreview
                imageAnalysis = newAnalysis
                isStarted = true
            } catch (e: Exception) {
                android.util.Log.e("CameraManager", "Camera binding failed", e)
            }
        }, ContextCompat.getMainExecutor(context))
    }

    fun stopCamera() {
        if (!isStarted && imageAnalysis == null && preview == null) return
        imageAnalysis?.clearAnalyzer()
        val useCasesToUnbind = listOfNotNull(preview, imageAnalysis).toTypedArray()
        if (useCasesToUnbind.isNotEmpty()) {
            cameraProvider?.unbind(*useCasesToUnbind)
        }
        imageAnalysis = null
        preview = null
        isStarted = false
    }

    fun shutdown() {
        stopCamera()
        cameraExecutor.shutdown()
        if (!cameraExecutor.awaitTermination(500, TimeUnit.MILLISECONDS)) {
            cameraExecutor.shutdownNow()
        }
    }
}
