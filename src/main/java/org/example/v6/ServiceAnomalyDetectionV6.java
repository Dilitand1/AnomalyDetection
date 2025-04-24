package org.example.v6;

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

public class ServiceAnomalyDetectionV6 {

    static class ServiceData {
        Map<String, List<Double>> subServices = new HashMap<>(); // Key: sub-service name
    }

    public static void main(String[] args) {
        trainAndSaveModels();
        Map<String, ServiceData> services = readData("C:\\Users\\Dilit\\IdeaProjects\\DeepLearning4j\\src\\main\\resources\\timeseriesWithClusters.csv");
        var data = services.get("1cluster");

        INDArray matrix = convertToMatrix(data);

        detectAnomalies("1cluster", matrix).forEach(t -> System.out.println(t + "\n"));
    }


    public static void trainAndSaveModels() {
        Map<String, ServiceData> services = readData("C:\\Users\\Dilit\\IdeaProjects\\DeepLearning4j\\src\\main\\resources\\timeseriesWithClusters.csv");

        for (Map.Entry<String, ServiceData> entry : services.entrySet()) {
            String serviceName = entry.getKey();
            ServiceData data = entry.getValue();

            // Пропускаем сервисы с недостаточными данными
            //if (data.samples.size() < MIN_SAMPLES) continue;

            // Конвертация данных
            INDArray matrix = convertToMatrix(data);

            // Нормализация
            NormalizerMinMaxScaler normalizer = new NormalizerMinMaxScaler();
            normalizer.fit(new DataSet(matrix, matrix));

            // Создание и обучение модели
            MultiLayerNetwork model = createModel(data.subServices.size());
            trainModel(model, matrix, normalizer);

            // Сохранение модели и параметров
            ModelManager.saveModel(serviceName, model, normalizer);
        }
    }

    public static List<String> detectAnomalies(String serviceName,
                                               INDArray newData) {
        ModelManager.ModelWrapper wrapper = ModelManager.loadModel(serviceName);

        if (wrapper.model == null || wrapper.normalizer == null) {
            throw new RuntimeException("Model not found for service: " + serviceName);
        }
        // Нормализация новых данных
        INDArray normalizedData = ModelManager.preprocessData(newData, wrapper.normalizer);
        //var originalData = newData.dup();

        // Получение предсказаний
        //INDArray output = wrapper.model.output(newData);
        INDArray output = wrapper.model.output(normalizedData);
        INDArray diff = output.sub(normalizedData);
        INDArray errors = diff.mul(diff); // [samples x features]


        return findAnomalies(errors, newData, List.of(serviceName));
    }

    private static void trainModel(MultiLayerNetwork model,
                                   INDArray data,
                                   NormalizerMinMaxScaler normalizer) {
        normalizer.transform(data);
        model.init();
        for (int i = 0; i < 100; i++) {
            model.fit(new DataSet(data, data));
        }
    }

//    public static void main(String[] args) {
//        Map<String, ServiceData> services = readData("C:\\Users\\Dilit\\IdeaProjects\\DeepLearning4j\\src\\main\\resources\\timeseriesWithClusters.csv");
//
//        for (String serviceName : services.keySet()) {
//            ServiceData data = services.get(serviceName);
//            INDArray matrix = convertToMatrix(data);
//
//            DataNormalizer normalizer = new DataNormalizer(matrix);
//            INDArray normalized = normalizer.normalize(matrix);
//
//            MultiLayerNetwork model = createModel(matrix.columns());
//            model.init();
//            model.fit(new DataSet(normalized, normalized));
//
//            INDArray reconstructions = model.output(normalized);
//            INDArray diff = reconstructions.sub(normalized);
//            INDArray errors = diff.mul(diff); // [samples x features]
//
//            List<String> anomalies = findAnomalies(errors, matrix, data.subServices.keySet().stream().toList());
//            System.out.println("Аномалии в " + serviceName + ": " + anomalies);
//        }
//    }

    // 1. Чтение данных
    private static Map<String, ServiceData> readData(String filename) {
        Map<String, ServiceData> data = new HashMap<>();
        // Пример данных: service1,subserviceA,12.5
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


    // 3. Конвертация данных в матрицу
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

    // 4. Создание модели автоэнкодера
    private static MultiLayerNetwork createModel(int numFeatures) {
        int encodingDim = Math.max(numFeatures / 2, 8); // Увеличиваем размер скрытого слоя

        return new MultiLayerNetwork(new NeuralNetConfiguration.Builder()
                .seed(123)
                .updater(new Adam(0.001))
                .weightInit(WeightInit.XAVIER)
                .list()
                .layer(new DenseLayer.Builder()
                        .nIn(numFeatures)
                        .nOut(encodingDim * 2)
                        .activation(Activation.RELU)
                        .build())
                .layer(new DenseLayer.Builder()
                        .nIn(encodingDim * 2)
                        .nOut(encodingDim)
                        .activation(Activation.RELU)
                        .build())
                .layer(new DenseLayer.Builder()
                        .nIn(encodingDim)
                        .nOut(encodingDim * 2)
                        .activation(Activation.RELU)
                        .build())
                .layer(new OutputLayer.Builder()
                        .nIn(encodingDim * 2)
                        .nOut(numFeatures)
                        .lossFunction(LossFunctions.LossFunction.MSE)
                        .activation(Activation.IDENTITY)
                        .build())
                .build());
    }

    // 5. Обучение модели
    private static void trainModel(MultiLayerNetwork model, INDArray data) {
        DataNormalizer normalizer = new DataNormalizer(data);
        INDArray normalizedData = normalizer.normalize(data);

        model.init();

        double bestLoss = Double.MAX_VALUE;
        int noImprovement = 0;

        for (int epoch = 0; epoch < 1000; epoch++) {
            model.fit(new DataSet(normalizedData, normalizedData));
            double loss = model.score();

            // Ранняя остановка при отсутствии улучшений
            if (loss < bestLoss - 0.001) {
                bestLoss = loss;
                noImprovement = 0;
            } else {
                noImprovement++;
                if (noImprovement > 50) break;
            }
        }
    }

    private static List<String> findAnomalies(INDArray errors,
                                              INDArray originalData,
                                              List<String> subServiceNames) {
        double threshold = calculateThreshold(errors);
        List<String> anomalies = new ArrayList<>();

        for (int row = 0; row < errors.rows(); row++) {
            for (int col = 0; col < errors.columns(); col++) {
                if (errors.getDouble(row, col) > threshold) {
                    String service = subServiceNames.get(col);
                    double value = originalData.getDouble(row, col);
                    anomalies.add(String.format("%s: %.6f", service, value));
                }
            }
        }
        return anomalies;
    }

    private static double calculateThreshold(INDArray errors) {
        INDArray flattened = errors.ravel();
        double mean = flattened.meanNumber().doubleValue();
        double std = flattened.stdNumber().doubleValue();
        return mean + 3 * std; // 3 сигмы от среднего
    }

//    private static double calculateThreshold(INDArray errors) {
//        INDArray flattened = errors.ravel();
//        double[] flatErrors = flattened.toDoubleVector();
//        Arrays.sort(flatErrors);
//        return flatErrors[(int) (flatErrors.length * 0.95)];
//    }
}