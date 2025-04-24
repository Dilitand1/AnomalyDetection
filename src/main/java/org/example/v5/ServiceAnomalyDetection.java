package org.example.v5;

import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.*;

//������� ������ ������� ��������. ������� ������������ � v6
public class ServiceAnomalyDetection {

    static class ServiceData {
        Map<String, List<Double>> subServices = new HashMap<>(); // Key: sub-service name
    }

    public static void main(String[] args) {
        Map<String, ServiceData> services = readData("C:\\Users\\Dilit\\IdeaProjects\\DeepLearning4j\\src\\main\\resources\\timeseriesWithClusters.csv");
        detectAnomalies(services);
    }

    // 1. ������ ������
    private static Map<String, ServiceData> readData(String filename) {
        Map<String, ServiceData> data = new HashMap<>();
        // ������ ������: service1,subserviceA,12.5
        try (Scanner scanner = new Scanner(new File(filename))) {
            while (scanner.hasNextLine()) {
                String[] parts = scanner.nextLine().split("\t");
                String service = parts[0].trim();
                String subService = parts[1].trim();
                double value = Double.parseDouble(parts[2].trim());

                data.computeIfAbsent(service, k -> new ServiceData())
                        .subServices
                        .computeIfAbsent(subService, k -> new ArrayList<>())
                        .add(value);
            }
        } catch (FileNotFoundException e) {
            throw new RuntimeException(e);
        }
        return data;
    }

    // 2. ����������� ��������
    private static void detectAnomalies(Map<String, ServiceData> services) {
        for (Map.Entry<String, ServiceData> entry : services.entrySet()) {
            String serviceName = entry.getKey();
            ServiceData data = entry.getValue();

            // �������������� � ������� [samples x features]
            INDArray matrix = convertToMatrix(data);

            // �������� � �������� ������
            MultiLayerNetwork model = createModel(data.subServices.size());
            trainModel(model, matrix);

            // ����� ��������
            List<String> anomalies = findAnomalies(model, matrix, data.subServices);
            System.out.println("�������� � " + serviceName + ": " + anomalies);
        }
    }

    // 3. ����������� ������ � �������
    private static INDArray convertToMatrix(ServiceData data) {
        int numSubServices = data.subServices.size();
        int numSamples = data.subServices.values().iterator().next().size();

        INDArray matrix = Nd4j.create(numSamples, numSubServices);

        List<String> subServiceNames = new ArrayList<>(data.subServices.keySet());
        for (int col = 0; col < numSubServices; col++) {
            List<Double> values = data.subServices.get(subServiceNames.get(col));
            for (int row = 0; row < numSamples; row++) {
                matrix.putScalar(row, col, values.get(row));
            }
        }
        return matrix;
    }

    // 4. �������� ������ ������������
    private static MultiLayerNetwork createModel(int numFeatures) {
        int encodingDim = Math.max(numFeatures / 4, 3); // ������ � 4 ����

        return new MultiLayerNetwork(new NeuralNetConfiguration.Builder()
                .seed(123)
                .updater(new Adam(0.001))
                .weightInit(WeightInit.XAVIER)
                .list()
                .layer(new DenseLayer.Builder()
                        .nIn(numFeatures)
                        .nOut(encodingDim)
                        .activation(Activation.RELU)
                        .build())
                .layer(new OutputLayer.Builder()
                        .nIn(encodingDim)
                        .nOut(numFeatures)
                        .lossFunction(LossFunctions.LossFunction.MSE)
                        .activation(Activation.IDENTITY)
                        .build())
                .build());
    }

    // 5. �������� ������
    private static void trainModel(MultiLayerNetwork model, INDArray data) {
        NormalizerMinMaxScaler normalizer = new NormalizerMinMaxScaler();
        normalizer.fit(new DataSet(data, data));
        normalizer.transform(data);

        model.init();
        for (int i = 0; i < 1000; i++) {
            model.fit(new DataSet(data, data));
        }
    }

    // 6. ����� ��������
    private static List<String> findAnomalies(MultiLayerNetwork model,
                                              INDArray data,
                                              Map<String, List<Double>> subServices) {
        INDArray reconstructions = model.output(data);
        INDArray diff = reconstructions.sub(data);
        INDArray errors = diff.mul(diff); // [samples x features]

        double threshold = calculateThreshold(errors);
        List<String> anomalies = new ArrayList<>();
        List<String> subServiceNames = new ArrayList<>(subServices.keySet());

        // ��������� �� ���� ������ (������ ��������)
        for (int row = 0; row < data.rows(); row++) {
            for (int col = 0; col < data.columns(); col++) {
                if (errors.getDouble(row, col) > threshold) {
                    String subService = subServiceNames.get(col);
                    //var value = data.getDouble(row, col);
                    var value = subServices.get(subService).get(row);
                    anomalies.add(String.format("%s: %.10f", subService, value));
                }
            }
        }

        return anomalies;
    }

    private static double calculateThreshold(INDArray errors) {
        INDArray flattened = errors.ravel();
        double[] flatErrors = flattened.toDoubleVector();
        Arrays.sort(flatErrors);
        return flatErrors[(int) (flatErrors.length * 0.95)];
    }
}