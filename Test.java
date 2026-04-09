package com.facerecog.test;
import com.google.gson.Gson;
public class Test {
    public static void main(String[] args) {
        float[] arr = {1.0f, 2.0f};
        String json = new Gson().toJson(arr);
        System.out.println("json: " + json);
        float[] obj = new Gson().fromJson(json, float[].class);
        System.out.println("length: " + obj.length);
        System.out.println("content: " + obj[0] + ", " + obj[1]);
    }
}
