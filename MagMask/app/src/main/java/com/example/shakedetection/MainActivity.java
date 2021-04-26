package com.example.shakedetection;

import androidx.appcompat.app.AppCompatActivity;

import android.annotation.SuppressLint;
import android.content.Context;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.TextView;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;

public class MainActivity extends AppCompatActivity implements SensorEventListener {

    private SensorManager sensorManager;
    private Sensor accel;
    private Sensor gyro;
    private Sensor magneto;
    private boolean run = false;
    private int activity_num;
    private int attempt_num = 0;
    private String initials;
    private ArrayList<String> accel_data = new ArrayList<>();
    private ArrayList<String> gyro_data = new ArrayList<>();
    private ArrayList<String> magneto_data = new ArrayList<>();
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        Button start_button = findViewById(R.id.start_button);
        Button stop_button = findViewById(R.id.stop_button);

        sensorManager = (SensorManager) getSystemService(Context.SENSOR_SERVICE);
        accel = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER);
        gyro = sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE);
        magneto = sensorManager.getDefaultSensor(Sensor.TYPE_MAGNETIC_FIELD);

        start_button.setOnClickListener(new View.OnClickListener(){
            public void onClick(View v){
                run = true;
                attempt_num += 1;
                try {
                    activity_num = Integer.parseInt(((EditText) findViewById(R.id.editText)).getText().toString());
                }catch (Exception e) {
                    activity_num = 0;
                }
                EditText edit = (EditText) findViewById(R.id.editText2);
                initials = edit.getText().toString();
                onResume();
            }
        });

        stop_button.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                run = false;
                onPause();
            }
        });
    }

    protected void onResume(){
        super.onResume();
        sensorManager.registerListener(this, accel, SensorManager.SENSOR_DELAY_NORMAL);
        sensorManager.registerListener(this, gyro, SensorManager.SENSOR_DELAY_NORMAL);
        sensorManager.registerListener(this, magneto, SensorManager.SENSOR_DELAY_NORMAL);
    }

    protected void onPause(){
        super.onPause();
        sensorManager.unregisterListener(this);
        TextView tv = findViewById(R.id.textView2);
        tv.setText("");
        TextView tv2 = findViewById(R.id.textView4);
        tv2.setText("");
        FileWriter writer = null;
        try {
            String filename = initials + "_" + activity_num + "_" + attempt_num + ".csv";
            File filepath = new File(getExternalFilesDir(null)+filename);
            filepath.createNewFile();
            writer = new FileWriter(filepath);
        } catch (IOException e) {
            e.printStackTrace();
        }
//        try {
//            writer.write(threshold + System.lineSeparator());
//        } catch (IOException e) {
//            e.printStackTrace();
//        }
        for(String line: accel_data){
            try {
                writer.write(line + System.lineSeparator());
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
        try {
            writer.write(System.lineSeparator());
        } catch (IOException e) {
            e.printStackTrace();
        }
        for(String line: gyro_data){
            try {
                writer.write(line + System.lineSeparator());
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
        try {
            writer.write(System.lineSeparator());
        } catch (IOException e) {
            e.printStackTrace();
        }
        for(String line: magneto_data){
            try {
                writer.write(line + System.lineSeparator());
            } catch (IOException e) {
                e.printStackTrace();
            }
        }

        try {
            writer.flush();
        } catch (IOException e) {
            e.printStackTrace();
        }
        try {
            writer.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    @Override
    public void onSensorChanged(SensorEvent event) {
        if(run) {
            if (event.sensor.getType() == Sensor.TYPE_ACCELEROMETER) {
                TextView tv = findViewById(R.id.textView2);
                float x = event.values[0];
                float y = event.values[1];
                float z = event.values[2];
                float timestamp = event.timestamp;
//                shake_detect(x, y, z);
                String line = x + "," + y + "," + z;
                tv.setText(line);
                //        String line = getString(R.string.coordinates, x, y, z);
                line += "," + timestamp;
                accel_data.add(line);
            }
            else if (event.sensor.getType() == Sensor.TYPE_GYROSCOPE) {
                TextView tv = findViewById(R.id.textView4);
                float x = event.values[0];
                float y = event.values[1];
                float z = event.values[2];
                float timestamp = event.timestamp;
                String line = x + "," + y + "," + z;
                tv.setText(line);
                line += "," + timestamp;
                gyro_data.add(line);
            }
            else if (event.sensor.getType() == Sensor.TYPE_MAGNETIC_FIELD) {
                TextView tv = findViewById(R.id.textView7);
                float x = event.values[0];
                float y = event.values[1];
                float z = event.values[2];
                float timestamp = event.timestamp;
                String line = x + "," + y + "," + z;
                tv.setText(line);
                line += "," + timestamp;
                magneto_data.add(line);
            }
        }
    }

    public void onAccuracyChanged(Sensor sensor, int accuracy) {
    }




//    @SuppressLint("SetTextI18n")
//    private void shake_detect(float x, float y, float z){
//        TextView tv = findViewById(R.id.textView4);
//        if (Math.sqrt(x*x + y*y + z*z) < threshold){
////            getString(R.string.shake, "No Shake");
//            tv.setText("No Shake");
//        }
//        else {
////            getString(R.string.shake, "Shake");
//            tv.setText("Shake");
//        }
//    }
}
