package one.entropy.demo;

import deepnetts.data.DataSets;
import deepnetts.data.DeepNettsBasicDataSet;
import deepnetts.util.DeepNettsException;
import io.quarkus.runtime.StartupEvent;
import io.vavr.control.Try;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import visrec.ri.ml.classification.MultiClassClassifierNetwork;

import javax.enterprise.context.ApplicationScoped;
import javax.enterprise.event.Observes;
import javax.visrec.ml.classification.MultiClassClassifier;
import javax.visrec.ml.data.BasicDataSet;
import javax.visrec.ml.data.DataSet;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.URL;
import java.net.URLConnection;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.Map;
import java.util.stream.Collectors;

@ApplicationScoped
public class IrisClassificator {
    private static final Logger LOGGER = LoggerFactory.getLogger(IrisClassificator.class.getCanonicalName());

    static MultiClassClassifier<float[], String> irisClassifier;

    void onStart(@Observes StartupEvent ev) {
        LOGGER.info("Training net...");
        DataSet dataSet = Try.of(() -> fromURL(getClass().getResource("/iris.txt"), ",", 4, 3, true)).get();
        DataSet[] trainTest = DataSets.trainTestSplit(dataSet, 0.7);
        // Build multi class classifier using Deep Netts implementation of Feed Forward Network under the hood
        irisClassifier = MultiClassClassifierNetwork.builder()
                .inputsNum(4)
                .hiddenLayers(16)
                .outputsNum(3)
                .maxEpochs(9000)
                .maxError(0.03f)
                .learningRate(0.01f)
                .trainingSet(trainTest[0])
                .build();
    }

    public static Map<String, Float> classify(float ph, float pw, float sh, float sw){
        return irisClassifier.classify(new float[] {ph, pw, sh, sw}).entrySet().stream().collect(Collectors.toMap(e -> e.getKey().trim(), e -> e.getValue()));
    }

    private static BasicDataSet fromURL(URL url, String delimiter, int inputsNum, int outputsNum, boolean hasColumnNames) throws IOException {
        DeepNettsBasicDataSet dataSet = new DeepNettsBasicDataSet(inputsNum, outputsNum);

        URLConnection conn = url.openConnection();
        String[] content;
        try (BufferedReader reader = new BufferedReader(new InputStreamReader(conn.getInputStream(), StandardCharsets.UTF_8))) {
            content = reader.lines().toArray(String[]::new);
        }
        if (content == null) {
            throw new NullPointerException("content == null");
        } else if (content.length <= 1 && hasColumnNames) {
            throw new IllegalArgumentException("content has one line of columns");
        } else if (content.length == 0) {
            throw new IllegalArgumentException("content has no lines");
        }

        int skipCount = 0;
        if (hasColumnNames) {    // get col names from the first line
            String[] colNames = content[0].split(delimiter);
            dataSet.setColumnNames(colNames);
            skipCount = 1;
        } else {
            String[] colNames = new String[inputsNum+outputsNum];
            for(int i=0; i<inputsNum;i++)
                colNames[i] = "in"+(i+1);

            for(int j=0; j<outputsNum;j++)
                colNames[inputsNum+j] = "out"+(j+1);

            dataSet.setColumnNames(colNames);
        }


        Arrays.stream(content)
                .skip(skipCount)
                .filter(l -> !l.isEmpty())
                .map(l -> toBasicDataSetItem(l, delimiter, inputsNum, outputsNum))
                .forEach(dataSet::add);
        return dataSet;
    }

    private static DeepNettsBasicDataSet.Item toBasicDataSetItem(String line, String delimiter, int inputsNum, int outputsNum) {
        String[] values = line.split(delimiter);
        if (values.length != (inputsNum + outputsNum)) {
            throw new DeepNettsException("Wrong number of values found " + values.length + " expected " + (inputsNum + outputsNum));
        }
        float[] in = new float[inputsNum];
        float[] out = new float[outputsNum];

        try {
            // these methods could be extracted into parse float vectors
            for (int i = 0; i < inputsNum; i++) { //parse inputs
                in[i] = Float.parseFloat(values[i]);
            }

            for (int j = 0; j < outputsNum; j++) { // parse outputs
                out[j] = Float.parseFloat(values[inputsNum + j]);
            }
        } catch (NumberFormatException nex) {
            throw new DeepNettsException("Error parsing csv, number expected: " + nex.getMessage(), nex);
        }

        return new DeepNettsBasicDataSet.Item(in, out);
    }
}
