package com.facerecog.recognition

import kotlin.math.sqrt

object FaceMath {
    fun cosineSimilarity(emb1: FloatArray, emb2: FloatArray): Float {
        var dotProduct = 0.0f
        var normA = 0.0f
        var normB = 0.0f
        for (i in emb1.indices) {
            dotProduct += emb1[i] * emb2[i]
            normA += emb1[i] * emb1[i]
            normB += emb2[i] * emb2[i]
        }
        return if (normA == 0.0f || normB == 0.0f) 0.0f else dotProduct / (sqrt(normA) * sqrt(normB))
    }
}

