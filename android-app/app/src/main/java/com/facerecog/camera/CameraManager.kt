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
    private var cameraProvider: ProcessCameraProvider? = null
    @Volatile private var isStarted: Boolean = false

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

            val preview = Preview.Builder()
                .setTargetResolution(Size(640, 480))
                .build()
                .also { it.setSurfaceProvider(previewView.surfaceProvider) }

            imageAnalysis = ImageAnalysis.Builder()
                .setTargetResolution(Size(640, 480))
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .build()

            imageAnalysis!!.setAnalyzer(cameraExecutor) { imageProxy ->
                try {
                    val bitmap = imageProxy.toBitmap()
                    onFrame(bitmap)
                } catch (e: Exception) {
                    android.util.Log.e("CameraManager", "Frame analysis failed", e)
                } finally {
                    imageProxy.close()
                }
            }

            try {
                provider.unbindAll()
                provider.bindToLifecycle(
                    lifecycleOwner,
                    CameraSelector.DEFAULT_FRONT_CAMERA,
                    preview,
                    imageAnalysis
                )
                isStarted = true
            } catch (e: Exception) {
                android.util.Log.e("CameraManager", "Camera binding failed", e)
            }
        }, ContextCompat.getMainExecutor(context))
    }

    fun stopCamera() {
        if (!isStarted && imageAnalysis == null) return
        imageAnalysis?.clearAnalyzer()
        imageAnalysis = null
        cameraProvider?.unbindAll()
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
