package org.example.v3;

import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;

public class DeepAnomalyDetection {

    public static void main(String[] args) {
        Map<String, List<Double>> data = readData("C:\\Users\\Dilit\\IdeaProjects\\DeepLearning4j\\src\\main\\resources\\timeseriesWithnames.csv");
        detectAnomaliesWithDL(data);
    }

    // ������ ������ (���������� ����������� �������)
    private static Map<String, List<Double>> readData(String filename) {
        Map<String, List<Double>> dataMap = new HashMap<>();
        try (BufferedReader br = new BufferedReader(new FileReader(filename))) {
            String line;
            while ((line = br.readLine()) != null) {
                String[] parts = line.split("\t");
                if (parts.length != 2) continue;
                String name = parts[0].trim();
                double value = Double.parseDouble(parts[1].trim());
                dataMap.computeIfAbsent(name, k -> new ArrayList<>()).add(value);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        return dataMap;
    }

    // ����������� �������� � �������������� ������������
    private static void detectAnomaliesWithDL(Map<String, List<Double>> data) {
        for (Map.Entry<String, List<Double>> entry : data.entrySet()) {
            String name = entry.getKey();
            List<Double> values = entry.getValue();

            if (values.size() < 10) { // ������� 10 �������� ��� ��������
                System.out.println("������������ ������ ���: " + name);
                continue;
            }

            // ������������ ������
            DataNormalization normalization = new DataNormalization(values);
            List<Double> normalizedValues = normalization.normalize(values);

            // �������� datasets
            INDArray input = Nd4j.create(normalizedValues.stream()
                    .mapToDouble(Double::doubleValue)
                    .toArray(), new int[]{values.size(), 1});

            DataSet dataSet = new DataSet(input, input); // Autoencoder: input = output

            // ������������ ������������
            MultiLayerNetwork model = new MultiLayerNetwork(new NeuralNetConfiguration.Builder()
                    .seed(123)
                    .updater(new Adam(0.01))
                    .weightInit(WeightInit.XAVIER)
                    .list()
                    .layer(new DenseLayer.Builder()
                            .nIn(1)
                            .nOut(3) // ������ �������� ����
                            .activation(Activation.RELU)
                            .build())
                    .layer(new OutputLayer.Builder()
                            .nIn(3)
                            .nOut(1)
                            .lossFunction(LossFunctions.LossFunction.MSE)
                            .activation(Activation.IDENTITY)
                            .build())
                    .build());
            model.init();

            // �������� ������
            for (int i = 0; i < 100; i++) { // 100 ����
                model.fit(dataSet);
            }

            // ��������� ������ �������������
            INDArray output = model.output(input);
            INDArray diff = output.sub(input);          // ��������� �������
            INDArray errors = diff.mul(diff);          // ������� ������ (������ pow(2))

            // ����������� ������ ��� �������� (95-� ����������)
            double threshold = getPercentile(errors.toDoubleVector(), 95);

            // ����� ��������
            List<Double> anomalies = new ArrayList<>();
            for (int i = 0; i < values.size(); i++) {
                if (errors.getDouble(i) > threshold) {
                    anomalies.add(values.get(i)); // ���������� ������������ ��������
                }
            }

            System.out.println("DL �������� ��� " + name + ": " + anomalies);
        }
    }

    // ��������������� ����� ��� ������������ ������
    static class DataNormalization {
        private final double min;
        private final double max;

        public DataNormalization(List<Double> data) {
            this.min = Collections.min(data);
            this.max = Collections.max(data);
        }

        public List<Double> normalize(List<Double> data) {
            List<Double> normalized = new ArrayList<>();
            for (Double value : data) {
                normalized.add((value - min) / (max - min));
            }
            return normalized;
        }
    }

    // ������ ���������� (���������� ����������� �������)
    private static double getPercentile(double[] data, double percentile) {
        Arrays.sort(data);
        int index = (int) Math.ceil((percentile / 100) * data.length) - 1;
        return data[Math.min(index, data.length - 1)];
    }
}