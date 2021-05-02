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

import com.chaquo.python.PyObject;
import com.chaquo.python.Python;
import com.chaquo.python.android.AndroidPlatform;

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
    private String env_code;
    private String py_input;
    private ArrayList<String> accel_data;
    private ArrayList<String> gyro_data;
    private ArrayList<String> magneto_data;

    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        Button start_button = findViewById(R.id.start_button);
        Button stop_button = findViewById(R.id.stop_button);

        sensorManager = (SensorManager) getSystemService(Context.SENSOR_SERVICE);
        accel = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER);
        gyro = sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE);
        magneto = sensorManager.getDefaultSensor(Sensor.TYPE_MAGNETIC_FIELD);

        if(!Python.isStarted()){
            Python.start(new AndroidPlatform(this));
        }
        Python py = Python.getInstance();
        final PyObject nc_file = py.getModule("load_normalizer_classifier"); //normalizer, classifier
        final PyObject nc_model = nc_file.callAttr("get_classifier_normalizer");


        start_button.setOnClickListener(new View.OnClickListener(){
            public void onClick(View v){
                run = true;
                attempt_num += 1;
                accel_data = new ArrayList<>();
                gyro_data = new ArrayList<>();
                magneto_data = new ArrayList<>();
                try {
                    activity_num = Integer.parseInt(((EditText) findViewById(R.id.editText)).getText().toString());
                }catch (Exception e) {
                    activity_num = 0;
                }
                EditText edit = (EditText) findViewById(R.id.editText2);
                EditText edit2 = (EditText) findViewById(R.id.editText3);
                initials = edit.getText().toString();
                env_code = edit2.getText().toString();
                onResume();
            }
        });

        stop_button.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                run = false;
                onPause(nc_model);
            }
        });
    }

    protected void onResume(){
        super.onResume();
        sensorManager.registerListener(this, accel, SensorManager.SENSOR_DELAY_NORMAL);
        sensorManager.registerListener(this, gyro, SensorManager.SENSOR_DELAY_NORMAL);
        sensorManager.registerListener(this, magneto, SensorManager.SENSOR_DELAY_NORMAL);
    }

    protected void onPause(PyObject model){
        super.onPause();
        sensorManager.unregisterListener(this);
        TextView tv = findViewById(R.id.textView2);
        tv.setText("");
        TextView tv2 = findViewById(R.id.textView4);
        tv2.setText("");
        FileWriter writer = null;
        try {
            String filename = initials + "_" + env_code + "_" + activity_num + "_" + attempt_num + ".csv";
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
        py_input = "";
        for(String line: accel_data){
            py_input += line + "\n";
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
        py_input += ";\n";
        for(String line: gyro_data){
            py_input += line + "\n";
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
        py_input += ";\n";
        for(String line: magneto_data){
            py_input += line + "\n";
            try {
                writer.write(line + System.lineSeparator());
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
        py_input += ";";

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
        if(!Python.isStarted()){
            Python.start(new AndroidPlatform(this));
        }
        Python py = Python.getInstance();
        final PyObject pyobj = py.getModule("make_predict");
        final PyObject obj = pyobj.callAttr("get_pred", model, py_input);
        TextView ptv = (TextView) findViewById(R.id.textView6);
        String results = obj.toJava(String.class);
        ptv.setText(results);
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
