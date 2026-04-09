package com.facerecog.db

import androidx.room.Dao
import androidx.room.Insert
import androidx.room.Query

@Dao
interface FaceDao {
    @Insert
    suspend fun insertUser(user: UserFace)

    @Query("SELECT * FROM users")
    suspend fun getAllUsers(): List<UserFace>
}

