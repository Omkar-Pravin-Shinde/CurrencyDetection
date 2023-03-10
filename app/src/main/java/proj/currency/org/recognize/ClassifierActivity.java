/*
 * Copyright 2016 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package proj.currency.org.recognize;

import android.graphics.Bitmap;
import android.graphics.Bitmap.Config;
import android.graphics.Canvas;
import android.graphics.Typeface;

import android.media.ImageReader.OnImageAvailableListener;
import android.os.SystemClock;
import android.speech.tts.TextToSpeech;
import android.util.Size;
import android.util.TypedValue;
import android.view.Display;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Vector;

import proj.currency.org.recognize.OverlayView.DrawCallback;
import proj.currency.org.recognize.env.BorderedText;
import proj.currency.org.recognize.env.Logger;

public class ClassifierActivity extends CameraActivity implements OnImageAvailableListener {
    private static final Logger LOGGER = new Logger();

    private static final Size DESIRED_PREVIEW_SIZE = new Size(640, 480);

    private Integer sensorOrientation;
    private MSCognitiveServicesClassifier classifier;

    TextToSpeech t1;

    private BorderedText borderedText;

    @Override
    protected int getLayoutId() {
        return R.layout.camera_connection_fragment;
    }

    @Override
    protected Size getDesiredPreviewFrameSize() {
        return DESIRED_PREVIEW_SIZE;
    }

    private static final float TEXT_SIZE_DIP = 10;

    @Override
    public void onPreviewSizeChosen(final Size size, final int rotation) {
        final float textSizePx = TypedValue.applyDimension(
                TypedValue.COMPLEX_UNIT_DIP, TEXT_SIZE_DIP, getResources().getDisplayMetrics());
        borderedText = new BorderedText(textSizePx);
        borderedText.setTypeface(Typeface.MONOSPACE);

        classifier = new MSCognitiveServicesClassifier(ClassifierActivity.this);

        previewWidth = size.getWidth();
        previewHeight = size.getHeight();

        final Display display = getWindowManager().getDefaultDisplay();
        final int screenOrientation = display.getRotation();

        LOGGER.i("Sensor orientation: %d, Screen orientation: %d", rotation, screenOrientation);

        sensorOrientation = rotation + screenOrientation;

        LOGGER.i("Initializing at size %dx%d", previewWidth, previewHeight);
        rgbFrameBitmap = Bitmap.createBitmap(previewWidth, previewHeight, Config.ARGB_8888);

        yuvBytes = new byte[3][];

        addCallback(
                new DrawCallback() {
                    @Override
                    public void drawCallback(final Canvas canvas) {
                        renderDebug(canvas);
                    }
                });
    }

    protected void processImageRGBbytes(int[] rgbBytes) {
        rgbFrameBitmap.setPixels(rgbBytes, 0, previewWidth, 0, 0, previewWidth, previewHeight);

        runInBackground(
                new Runnable() {
                    @Override
                    public void run() {
                        final long startTime = SystemClock.uptimeMillis();
                        Classifier.Recognition r = classifier.classifyImage(rgbFrameBitmap, sensorOrientation);
                        lastProcessingTimeMs = SystemClock.uptimeMillis() - startTime;

                        final List<Classifier.Recognition> results = new ArrayList<>();

                        if (r.getConfidence() > 0.7) {
                            results.add(r);
                        }

                        LOGGER.i("Detect: %s", results);
                        if (resultsView == null) {
                            resultsView = findViewById(R.id.results);
                        }
                        resultsView.setResults(results);

                        t1=new TextToSpeech(getApplicationContext(), new TextToSpeech.OnInitListener() {
                            @Override
                            public void onInit(int status) {
                                if (status!=TextToSpeech.ERROR)
                                {
                                    t1.setLanguage(Locale.UK);
                                }

                            }
                        });

                        String t2=results.toString();
                        t1.speak(t2,TextToSpeech.QUEUE_FLUSH,null);
                        requestRender();
                        computing = false;
                        if (postInferenceCallback != null) {
                            postInferenceCallback.run();
                        }
                    }
                });

    }

    @Override
    public void onSetDebug(boolean debug) {
    }

    private void renderDebug(final Canvas canvas) {
        if (!isDebug()) {
            return;
        }

        final Vector<String> lines = new Vector<String>();
        lines.add("Inference time: " + lastProcessingTimeMs + "ms");
        borderedText.drawLines(canvas, 10, canvas.getHeight() - 10, lines);
    }
}
