package org.example;

import org.deeplearning4j.nn.conf.*;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.util.*;

public class AnomalyDetectionWithTraining {

    static class AnomalyResult {
        String name;
        double value;
        double error;
        boolean isAnomaly;

        public AnomalyResult(String name, double value, double error, boolean isAnomaly) {
            this.name = name;
            this.value = value;
            this.error = error;
            this.isAnomaly = isAnomaly;
        }
    }

    public static void main(String[] args) {
        // 1. Загрузка и подготовка данных
        Map<String, List<Double>> data = new HashMap<>();
        data.put("sensor1", Arrays.asList(1.0, 1.1, 1.2, 1.1, 1.3, 5.0, 1.2, 1.1));
        data.put("sensor2", Arrays.asList(2.0, 2.1, 2.0, 2.3, 2.2, 2.1, 10.0, 2.2));

        // 2. Обучение модели для каждого name
        Map<String, MultiLayerNetwork> models = trainModels(data, 1); // windowSize = 3

        // 3. Обнаружение аномалий
        Map<String, List<AnomalyResult>> results = new HashMap<>();
        for (Map.Entry<String, MultiLayerNetwork> entry : models.entrySet()) {
            String name = entry.getKey();
            INDArray series = Nd4j.create(data.get(name).stream().mapToDouble(d -> d).toArray());
            results.put(name, detectAnomalies(name, series, entry.getValue(), 3));
        }

        // 4. Вывод результатов
        printAnomalies(results);
    }

    public static Map<String, MultiLayerNetwork> trainModels(Map<String, List<Double>> data, int windowSize) {
        Map<String, MultiLayerNetwork> models = new HashMap<>();

        for (Map.Entry<String, List<Double>> entry : data.entrySet()) {
            String name = entry.getKey();
            INDArray series = Nd4j.create(entry.getValue().stream().mapToDouble(d -> d).toArray());

            // Создаем и обучаем модель для этого name
            MultiLayerNetwork model = createModel(windowSize);
            trainModel(model, series, windowSize, 100); // 100 эпох

            models.put(name, model);
            System.out.println("Модель для " + name + " обучена");
        }

        return models;
    }

    public static MultiLayerNetwork createModel(int windowSize) {
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(12345)
                .weightInit(WeightInit.XAVIER)
                .updater(new Adam(0.01))
                .list()
                .layer(new LSTM.Builder()
                        .nIn(1)
                        .nOut(64)
                        .activation(Activation.TANH)
                        .build())
                .layer(new LSTM.Builder()
                        .nIn(64)
                        .nOut(32)
                        .activation(Activation.TANH)
                        .build())
                .layer(new RnnOutputLayer.Builder()
                        .nIn(32)
                        .nOut(1)
                        .activation(Activation.IDENTITY)
                        .lossFunction(LossFunctions.LossFunction.MSE)
                        .build())
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        return model;
    }

    public static void trainModel(MultiLayerNetwork model, INDArray series, int windowSize, int epochs) {
        List<DataSet> trainingData = new ArrayList<>();

        // Подготовка обучающих данных (скользящее окно)
        for (int i = 0; i < series.length() - windowSize; i++) {
            INDArray input = series.get(NDArrayIndex.interval(i, i + windowSize - 1));
            INDArray label = series.get(NDArrayIndex.point(i + windowSize - 1));

            input = input.reshape(1, 1, windowSize - 1);
            label = label.reshape(1, 1, 1);

            trainingData.add(new DataSet(input, label));
        }

        // Обучение модели
        for (int epoch = 0; epoch < epochs; epoch++) {
            for (DataSet ds : trainingData) {
                model.fit(ds);
            }
            if ((epoch + 1) % 10 == 0) {
                System.out.println("Эпоха " + (epoch + 1) + " завершена");
            }
        }
    }

    public static List<AnomalyResult> detectAnomalies(String name, INDArray series,
                                                      MultiLayerNetwork model, int windowSize) {
        List<AnomalyResult> results = new ArrayList<>();
        List<Double> errors = new ArrayList<>();
        var length = series.length();

        for (int i = 0; i < length - windowSize; i++) {
            INDArray window = series.get(NDArrayIndex.interval(i, i + windowSize - 1));
            INDArray actual = series.get(NDArrayIndex.point(i + windowSize - 1));

            window = window.reshape(1, 1, windowSize - 1);
            INDArray predicted = model.output(window, false);

            double error = Math.abs(actual.getDouble(0) - predicted.getDouble(0));
            errors.add(error);

            results.add(new AnomalyResult(name, actual.getDouble(0), error, false));
        }

        // Определение порога аномалий
        double mean = errors.stream().mapToDouble(d -> d).average().orElse(0);
        double std = Math.sqrt(errors.stream()
                .mapToDouble(e -> Math.pow(e - mean, 2))
                .average()
                .orElse(0));
        double threshold = mean + 3 * std;

        // Помечаем аномалии
        for (AnomalyResult res : results) {
            res.isAnomaly = res.error > threshold;
        }

        return results;
    }

    public static void printAnomalies(Map<String, List<AnomalyResult>> results) {
        for (Map.Entry<String, List<AnomalyResult>> entry : results.entrySet()) {
            System.out.println("\nАномалии для " + entry.getKey() + ":");
            for (AnomalyResult res : entry.getValue()) {
                if (res.isAnomaly) {
                    System.out.printf("Значение: %.2f, Ошибка: %.2f\n", res.value, res.error);
                }
            }
        }
    }
}