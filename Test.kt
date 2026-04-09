import kotlin.math.sqrt
fun main() {
    val arr = floatArrayOf(1f, 2f, 3f)
    var sum = 0f
    for (v in arr) sum += v * v
    val norm = Math.sqrt(sum.toDouble()).toFloat()
    val normArr = arr.map { it / norm }.toFloatArray()
    println(normArr.joinToString())
}
