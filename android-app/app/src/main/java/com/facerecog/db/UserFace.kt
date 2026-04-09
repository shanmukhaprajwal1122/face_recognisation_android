package com.facerecog.db

import androidx.room.Entity
import androidx.room.PrimaryKey

@Entity(tableName = "users")
data class UserFace(
    @PrimaryKey(autoGenerate = true) val id: Int = 0,
    val name: String,
    val embedding: String // Store FloatArray as JSON String
)

