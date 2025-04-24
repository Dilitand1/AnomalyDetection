package org.example.v0;

import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.*;
import org.deeplearning4j.nn.conf.layers.CenterLossOutputLayer;
import org.deeplearning4j.nn.conf.layers.variational.*;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.util.*;
import java.util.stream.Collectors;

public class AutoencoderAnomalyDetection {

    public static void main(String[] args) {
        // 1. ��������� ��� �������� ������
        INDArray timeSeriesData = generateTimeSeriesData(1000);

        // 2. ���������� ������
        int windowSize = 30;
        DataSetIterator trainData = prepareDataset(timeSeriesData, windowSize);

        // 3. ������������ ������
        NormalizerMinMaxScaler normalizer = new NormalizerMinMaxScaler();
        normalizer.fit(trainData);
        trainData.setPreProcessor(normalizer);

        // 4. �������� � �������� ������������
        var vae = createVAE(windowSize);
        vae.setListeners(new ScoreIterationListener(50));

        int numEpochs = 100;
        trainVAE(vae, trainData, numEpochs);

        // 5. ����������� ��������
        detectAnomalies(vae, timeSeriesData, windowSize, normalizer);
    }

    // ��������� ���������� ���� � ����������
    private static INDArray generateTimeSeriesData(int length) {
        INDArray data = Nd4j.zeros(length);
        Random rand = new Random();

        // ������� ������� (��������� + ���)
        for (int i = 0; i < length; i++) {
            double value = Math.sin(i * 0.05) + 0.1 * rand.nextGaussian();
            data.putScalar(i, value);
        }

        // ���������� ��������
        addSpike(data, 100, 3.0);
        addSpike(data, 250, -2.5);
        addSpike(data, 400, 2.8);
        addSpike(data, 600, 3.2);
        addSpike(data, 800, -3.0);

        return data;
    }

    private static void addSpike(INDArray data, int position, double magnitude) {
        data.putScalar(position, magnitude);
        // ��������� ��������� ������� ������ �������� ��� ��������������
        for (int i = 1; i <= 3; i++) {
            if (position - i >= 0) {
                data.putScalar(position - i, data.getDouble(position - i) + magnitude * 0.3 / i);
            }
            if (position + i < data.length()) {
                data.putScalar(position + i, data.getDouble(position + i) + magnitude * 0.3 / i);
            }
        }
    }

    // ���������� ������ � ���������� �����
    static DataSetIterator prepareDataset(INDArray timeSeries, int windowSize) {
        List<DataSet> dataSets = new ArrayList<>();

        for (int i = 0; i < timeSeries.length() - windowSize; i++) {
            INDArray window = timeSeries.get(NDArrayIndex.interval(i, i + windowSize));
            window = window.reshape(1, windowSize); // ������ [batchSize, features]

            // ��� ������������ ���� � ����� ���������
            dataSets.add(new DataSet(window, window));
        }

        return new ListDataSetIterator<>(dataSets, 64); // Batch size 64
    }

    public static ComputationGraph createVAE(int inputSize) {
        // ��������� ������
        int latentSize = 10;
        int encoderLayerSize = 128;
        int decoderLayerSize = 128;

        // 1. ������������ VAE
        VariationalAutoencoder vaeLayer = new VariationalAutoencoder.Builder()
                .encoderLayerSizes(encoderLayerSize, encoderLayerSize)
                .decoderLayerSizes(decoderLayerSize, decoderLayerSize)
                .pzxActivationFunction(Activation.IDENTITY)
                .reconstructionDistribution(new GaussianReconstructionDistribution(Activation.IDENTITY))
                .nIn(inputSize)
                .nOut(latentSize)
                .build();

        // 2. ���������� ������������ ComputationGraph
        ComputationGraphConfiguration config = new NeuralNetConfiguration.Builder()
                .seed(12345)
                .updater(new Adam(0.001))
                .graphBuilder()
                .addInputs("input")
                // ��������� VAE ��� ������������� ����
                .addLayer("vae", vaeLayer, "input")
                // ��������� �������� ���� ��� �������������
                .addLayer("output", new CenterLossOutputLayer.Builder()
                        .nIn(latentSize)
                        .nOut(inputSize)
                        .activation(Activation.IDENTITY)
                        .lossFunction(LossFunctions.LossFunction.MSE)
                        .build(), "vae")
                .setOutputs("output") // ��������� ���������� �����
                .build();

        ComputationGraph vae = new ComputationGraph(config);
        vae.init();
        return vae;
    }

    // ����� ��� �������� VAE
    public static void trainVAE(ComputationGraph vae, DataSetIterator trainData, int numEpochs) {
        for (int epoch = 0; epoch < numEpochs; epoch++) {
            while (trainData.hasNext()) {
                DataSet ds = trainData.next();
                // ��� VAE ���� � ����� ������ ���� ���������
                vae.fit(new DataSet(ds.getFeatures(), ds.getFeatures()));
            }
            trainData.reset();
            System.out.println("Epoch " + (epoch + 1) + " complete");
        }
    }

    // ���������� ����� �������� ��� VAE
    private static void trainVAE(MultiLayerNetwork model, DataSetIterator trainData, int numEpochs) {
        System.out.println("������ �������� ������������...");
        for (int i = 0; i < numEpochs; i++) {
            // ��� VAE ���������� fit � ������ pretrain
            model.pretrain(trainData); // ������ model.fit()
            trainData.reset();
            if ((i + 1) % 10 == 0) {
                System.out.println("����� " + (i + 1) + " ���������");
            }
        }
        System.out.println("�������� ���������");
    }

    static void detectAnomalies(ComputationGraph vae, INDArray timeSeries,
                                int windowSize, NormalizerMinMaxScaler normalizer) {
        List<Double> reconstructionErrors = new ArrayList<>();
        List<Integer> anomalyPositions = new ArrayList<>();

        // 1. ���������� ������ �������������
        for (int i = 0; i < timeSeries.length() - windowSize; i++) {
            // �������� ������� ���� ������
            INDArray window = timeSeries.get(NDArrayIndex.interval(i, i + windowSize));
            window = window.reshape(1, windowSize); // ������ [1, windowSize]

            // ������������ ������
            INDArray normalizedWindow = window.dup();
            normalizer.transform(normalizedWindow);

            // �������� ������������� �� VAE
            INDArray[] outputs = vae.output(normalizedWindow);
            INDArray reconstructed = outputs[0]; // ������ ����� - �������������

            // �������� ������������
            normalizer.revertFeatures(reconstructed);

            // ��������� ������ ������������� (MSE)
            double error = reconstructed.sub(window).norm2Number().doubleValue();
            reconstructionErrors.add(error);
        }

        // 2. �������������� ������ ������
        double mean = reconstructionErrors.stream().mapToDouble(d -> d).average().orElse(0);
        double std = Math.sqrt(reconstructionErrors.stream()
                .mapToDouble(e -> Math.pow(e - mean, 2))
                .average()
                .orElse(0));

        // 3. ����������� ������ (3 �����)
        double threshold = mean + 3 * std;
        System.out.printf("\nAnomaly threshold: %.4f (mean=%.4f, std=%.4f)\n",
                threshold, mean, std);

        // 4. ����� ��������
        for (int i = 0; i < reconstructionErrors.size(); i++) {
            if (reconstructionErrors.get(i) > threshold) {
                // ���������� ��� ����� � ���� ���� ��� ����������
                for (int j = 0; j < windowSize; j++) {
                    if (i + j < timeSeries.length()) {
                        anomalyPositions.add(i + j);
                    }
                }
            }
        }

        // 5. �������� ���������� � ����������
        anomalyPositions = anomalyPositions.stream()
                .distinct()
                .sorted()
                .collect(Collectors.toList());

        // 6. ������������
        visualizeResults(timeSeries, anomalyPositions, reconstructionErrors, threshold, windowSize);
    }

    // ��������������� ������
    private static double calculateMean(double[] values) {
        return Arrays.stream(values).average().orElse(0);
    }

    private static double calculateStdDev(double[] values, double mean) {
        return Math.sqrt(Arrays.stream(values)
                .map(v -> Math.pow(v - mean, 2))
                .average()
                .orElse(0));
    }

    private static List<Integer> findAnomalyPositions(List<Integer> anomalyWindows,
                                                      int windowSize, List<Double> errors) {
        List<Integer> positions = new ArrayList<>();
        Set<Integer> addedPositions = new HashSet<>();

        for (int windowIdx : anomalyWindows) {
            // ������� ������� � ������������ ������� � ����
            int start = windowIdx;
            int end = Math.min(start + windowSize, errors.size());

            double maxError = -1;
            int maxPos = -1;

            for (int i = start; i < end; i++) {
                if (errors.get(i) > maxError && !addedPositions.contains(i)) {
                    maxError = errors.get(i);
                    maxPos = i;
                }
            }

            if (maxPos != -1) {
                positions.add(maxPos);
                addedPositions.add(maxPos);
            }
        }

        return positions;
    }

    private static void visualizeResults(INDArray timeSeries, List<Integer> anomalies,
                                         List<Double> errors, double threshold, int windowSize) {
        System.out.println("\nDetection results:");
        System.out.println("Total anomalies detected: " + anomalies.size());

        // ����� ���������� ���� � ��������� ��������
        System.out.println("\nTime series with anomalies:");
        int displayLength = (int) Math.min(100, timeSeries.length());
        for (int i = 0; i < displayLength; i++) {
            String marker = anomalies.contains(i) ? " <<< ANOMALY" : "";
            System.out.printf("%4d: %8.4f%s\n", i, timeSeries.getDouble(i), marker);
        }

        // ����� ������� ������
        System.out.println("\nReconstruction errors:");
        int errorDisplayLength = Math.min(50, errors.size());
        for (int i = 0; i < errorDisplayLength; i++) {
            String mark = errors.get(i) > threshold ? " !!!" : "";
            System.out.printf("Window %3d: error=%.4f%s\n", i, errors.get(i), mark);
        }
    }
}