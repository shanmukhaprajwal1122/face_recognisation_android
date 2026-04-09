import com.google.gson.Gson
fun main() {
    val arr = floatArrayOf(1.0f, 2.0f)
    val json = Gson().toJson(arr)
    println("json: $json")
    val obj = Gson().fromJson(json, FloatArray::class.java)
    println("class: ${obj::class.java}")
    println("content: ${obj.joinToString()}")
}
