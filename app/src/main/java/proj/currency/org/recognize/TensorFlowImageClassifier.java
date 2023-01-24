/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

package proj.currency.org.recognize;

import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.os.Trace;
import android.util.Log;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.PriorityQueue;
import java.util.Vector;
import org.tensorflow.Operation;
import org.tensorflow.contrib.android.TensorFlowInferenceInterface;


public class TensorFlowImageClassifier implements Classifier {
  private static final String TAG = "TensorFlowImageClassifier";


  private static final int MAX_RESULTS = 3;
  private static final float THRESHOLD = 0.1f;


  private String inputName;
  private String outputName;
  private int inputSize;
  private int imageMean;
  private float imageStd;


  private Vector<String> labels = new Vector<String>();
  private int[] intValues;
  private float[] floatValues;
  private float[] outputs;
  private String[] outputNames;

  private boolean logStats = false;

  private TensorFlowInferenceInterface inferenceInterface;

  private TensorFlowImageClassifier() {}


  public static Classifier create(
      AssetManager assetManager,
      String modelFilename,
      String labelFilename,
      int inputSize,
      int imageMean,
      float imageStd,
      String inputName,
      String outputName) {
    TensorFlowImageClassifier c = new TensorFlowImageClassifier();
    c.inputName = inputName;
    c.outputName = outputName;


    // TODO(andrewharp): make this handle non-assets.
    String actualFilename = labelFilename.split("file:///android_asset/")[1];
    Log.i(TAG, "Reading labels from: " + actualFilename);
    BufferedReader br = null;
    try {
      br = new BufferedReader(new InputStreamReader(assetManager.open(actualFilename)));
      String line;
      while ((line = br.readLine()) != null) {
        c.labels.add(line);
      }
      br.close();
    } catch (IOException e) {
      throw new RuntimeException("Problem reading label file!" , e);
    }

    c.inferenceInterface = new TensorFlowInferenceInterface(assetManager, modelFilename);

    // The shape of the output is [N, NUM_CLASSES], where N is the batch size.
    final Operation operation = c.inferenceInterface.graphOperation(outputName);
    final int numClasses = (int) operation.output(0).shape().size(1);
    Log.i(TAG, "Read " + c.labels.size() + " labels, output layer size is " + numClasses);


    c.inputSize = inputSize;
    c.imageMean = imageMean;
    c.imageStd = imageStd;

    // Pre-allocate buffers.
    c.outputNames = new String[] {outputName};
    c.intValues = new int[inputSize * inputSize];
    c.floatValues = new float[inputSize * inputSize * 3];
    c.outputs = new float[numClasses];

    return c;
  }

  @Override
  public List<Recognition> recognizeImage(final Bitmap bitmap) {
    // Log this method so that it can be analyzed with systrace.
    Trace.beginSection("recognizeImage");

    Trace.beginSection("preprocessBitmap");
    // Preprocess the image data from 0-255 int to normalized float based
    // on the provided parameters.
    bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
    for (int i = 0; i < intValues.length; ++i) {
      final int val = intValues[i];
      floatValues[i * 3 + 0] = (((val >> 16) & 0xFF) - imageMean) / imageStd;
      floatValues[i * 3 + 1] = (((val >> 8) & 0xFF) - imageMean) / imageStd;
      floatValues[i * 3 + 2] = ((val & 0xFF) - imageMean) / imageStd;
    }
    Trace.endSection();


    Trace.beginSection("feed");
    inferenceInterface.feed(inputName, floatValues, 1, inputSize, inputSize, 3);
    Trace.endSection();


    Trace.beginSection("run");
    inferenceInterface.run(outputNames, logStats);
    Trace.endSection();


    Trace.beginSection("fetch");
    inferenceInterface.fetch(outputName, outputs);
    Trace.endSection();


    PriorityQueue<Recognition> pq =
        new PriorityQueue<Recognition>(
            3,
            new Comparator<Recognition>() {
              @Override
              public int compare(Recognition lhs, Recognition rhs) {
                // Intentionally reversed to put high confidence at the head of the queue.
                return Float.compare(rhs.getConfidence(), lhs.getConfidence());
              }
            });
    for (int i = 0; i < outputs.length; ++i) {
      if (outputs[i] > THRESHOLD) {
        pq.add(
            new Recognition(
                "" + i, labels.size() > i ? labels.get(i) : "unknown", outputs[i], null));
      }
    }
    final ArrayList<Recognition> recognitions = new ArrayList<Recognition>();
    int recognitionsSize = Math.min(pq.size(), MAX_RESULTS);
    for (int i = 0; i < recognitionsSize; ++i) {
      recognitions.add(pq.poll());
    }
    Trace.endSection(); // "recognizeImage"
    return recognitions;
  }

  @Override
  public void enableStatLogging(boolean logStats) {
    this.logStats = logStats;
  }

  @Override
  public String getStatString() {
    return inferenceInterface.getStatString();
  }

  @Override
  public void close() {
    inferenceInterface.close();
  }
}
