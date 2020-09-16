package com.example.covid19detection;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

import android.graphics.Color;
import android.os.Bundle;


import android.app.Activity;
import android.content.Intent;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.net.Uri;
import android.provider.MediaStore;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.List;
import android.graphics.Matrix;


import org.tensorflow.lite.Interpreter;


public class MainActivity extends AppCompatActivity {

    ImageView imageView;
    Button btn_gallery,btn_predict,close;
    TextView text;
    private int PICK_IMAGE_REQUEST = 1000;

    /** Name of the model file stored in Assets. */
    private static final String MODEL_PATH = "covid19.tflite";

    /** Name of the label file stored in Assets. */
    private static final String LABEL_PATH = "lab.txt";

    private Interpreter tflite;

    /** Labels corresponding to the output of the vision model. */
    private List<String> labelList;

    Uri uri;
    Bitmap bitmap;

    private static final float IMAGE_STD = 255.0f;

    private int inputSize = 224;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        imageView = findViewById(R.id.imageView);
        btn_gallery = findViewById(R.id.button);
        btn_predict = findViewById(R.id.button2);
        text = findViewById(R.id.textView);
        close = findViewById(R.id.button3);

        btn_gallery.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {

                Intent i = chooseImage();
            }
        });

        btn_predict.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {


                predict(uri);

            }
        });


        close.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {

                System.exit(1);
            }
        });

    }

    /** hors main */

    public Intent chooseImage() {
        Intent intent = new Intent();
        intent.setType("image/*");
        intent.setAction(Intent.ACTION_GET_CONTENT);
        startActivityForResult(Intent.createChooser(intent, "Select Picture"), PICK_IMAGE_REQUEST);

        return  intent;
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);

        if (requestCode == PICK_IMAGE_REQUEST && resultCode == RESULT_OK && data != null && data.getData() != null) {

            uri = data.getData();

            try {
                bitmap = MediaStore.Images.Media.getBitmap(getContentResolver(), uri);
                // Log.d(TAG, String.valueOf(bitmap));
                imageView.setImageBitmap(bitmap);
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }

    /** Reads label list from Assets. */
    private List<String> loadLabelList() throws IOException {
        List<String> labelList = new ArrayList<String>();
        BufferedReader reader =
                new BufferedReader(new InputStreamReader(this.getAssets().open(LABEL_PATH)));
        String line;
        while ((line = reader.readLine()) != null) {
            labelList.add(line);
        }
        reader.close();
        return labelList;
    }

    private MappedByteBuffer loadModelFile() throws IOException {
        AssetFileDescriptor fileDescriptor = this.getAssets().openFd(MODEL_PATH);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    public Bitmap getResizedBitmap(Bitmap bm, int newWidth, int newHeight) {
        int width = bm.getWidth();
        int height = bm.getHeight();
        float scaleWidth = ((float) newWidth) / width;
        float scaleHeight = ((float) newHeight) / height;
        // CREATE A MATRIX FOR THE MANIPULATION
        Matrix matrix = new Matrix();
        // RESIZE THE BIT MAP
        matrix.postScale(scaleWidth, scaleHeight);

        // "RECREATE" THE NEW BITMAP
        Bitmap resizedBitmap = Bitmap.createBitmap(
                bm, 0, 0, width, height, matrix, false);
        bm.recycle();
        return resizedBitmap;
    }

    private ByteBuffer convertBitmapToByteBuffer(Bitmap b,int size) {

        ByteBuffer byteBuffer = ByteBuffer.allocateDirect(size);

        byteBuffer.order(ByteOrder.nativeOrder());

        int[] intValues = new int[inputSize * inputSize];
        b.getPixels(intValues, 0, b.getWidth(), 0, 0, b.getWidth(), b.getHeight());

        int pixel = 0;
        for (int i = 0; i < inputSize; ++i) {
            for (int j = 0; j < inputSize; ++j) {
                final int val = intValues[pixel++];
                byteBuffer.putFloat((((val >> 16) & 0xFF))/IMAGE_STD);
                byteBuffer.putFloat((((val >> 8) & 0xFF))/IMAGE_STD);
                byteBuffer.putFloat((((val) & 0xFF))/IMAGE_STD);
            }
        }
        return byteBuffer;
    }

    void  predict(Uri uri)
    {
        float[][] output = new float[1][2];
        String predict_value = null;

        try{
            tflite = new Interpreter(loadModelFile());
            labelList = loadLabelList();/** labelList.get(0) */

            Bitmap b  = getResizedBitmap(bitmap,inputSize,inputSize);
            ByteBuffer buffer = convertBitmapToByteBuffer(b,602112);

            /*ByteBuffer buffer = ByteBuffer.allocate(602112);*/
            tflite.run(buffer, output);

            if(output[0][0]>output[0][1])
            {
                predict_value = labelList.get(0)+" accuracy : "+(output[0][0]*100)+"%";
                text.setText(predict_value);
                text.setTextColor(Color.parseColor("#FF0000"));
            }
            else
            {
                predict_value = labelList.get(1)+" accuracy : "+((output[0][1])*100)+"%";
                text.setText(predict_value);
                text.setTextColor(Color.parseColor("#00FF00"));
            }


        }catch(Exception ex)
        {
            ex.printStackTrace();
        }


    }
}
