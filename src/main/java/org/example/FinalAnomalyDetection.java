package org.example;

import org.deeplearning4j.nn.conf.*;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.util.*;

public class FinalAnomalyDetection {

    public static void main(String[] args) {
        // 1. Тестовые данные с явной аномалией
        double[] data = {1.0, 1.1, 1.2, 1.1, 1.3, 1.2, 1.4, 1.3,
                10.0, // Явная аномалия
                1.2, 1.1, 1.3, 1.2, 1.1};

        // 2. Правильное создание 3D массива [batchSize, features, timesteps]
        INDArray series = Nd4j.create(data).reshape(1, 1, data.length);

        // 3. Параметры
        int windowSize = 3;
        int epochs = 300;
        double anomalyThreshold = 4.0;

        // 4. Создание и обучение модели
        MultiLayerNetwork model = createLSTMModel();
        trainModel(model, series, windowSize, epochs);

        // 5. Обнаружение аномалий
        List<Integer> anomalies = detectAnomalies(model, series, windowSize, anomalyThreshold);

        // 6. Визуализация
        printResults(data, anomalies);
    }

    public static MultiLayerNetwork createLSTMModel() {
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(12345)
                .weightInit(WeightInit.XAVIER)
                .updater(new Adam(0.005))
                .list()
                .layer(new LSTM.Builder()
                        .nIn(1)  // Один признак
                        .nOut(64)
                        .activation(Activation.TANH)
                        .build())
                .layer(new LSTM.Builder()
                        .nIn(64)
                        .nOut(32)
                        .activation(Activation.TANH)
                        .build())
                .layer(new RnnOutputLayer.Builder(LossFunctions.LossFunction.MSE)
                        .nIn(32)
                        .nOut(1)  // Предсказываем одно значение
                        .activation(Activation.IDENTITY)
                        .build())
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        return model;
    }

    public static void trainModel(MultiLayerNetwork model, INDArray series,
                                  int windowSize, int epochs) {
        var timesteps = series.size(2);

        for (int epoch = 0; epoch < epochs; epoch++) {
            for (int i = 0; i < timesteps - windowSize; i++) {
                // Правильный 3D вход [1, 1, windowSize]
                INDArray input = series.get(
                        NDArrayIndex.point(0),
                        NDArrayIndex.point(0),
                        NDArrayIndex.interval(i, i + windowSize)
                ).reshape(1, 1, windowSize);

                // Метка: следующее значение [1, 1, 1]
                INDArray label = series.get(
                        NDArrayIndex.point(0),
                        NDArrayIndex.point(0),
                        NDArrayIndex.point(i + windowSize)
                ).reshape(1, 1, 1);

                model.fit(input, label);
            }
        }
    }

    public static List<Integer> detectAnomalies(MultiLayerNetwork model,
                                                INDArray series,
                                                int windowSize,
                                                double threshold) {
        List<Integer> anomalies = new ArrayList<>();
        var timesteps = series.size(2);

        for (int i = 0; i < timesteps - windowSize; i++) {
            // Входное окно [1, 1, windowSize]
            INDArray input = series.get(
                    NDArrayIndex.point(0),
                    NDArrayIndex.point(0),
                    NDArrayIndex.interval(i, i + windowSize)
            ).reshape(1, 1, windowSize);

            // Предсказание
            INDArray output = model.output(input, false);
            double predicted = output.getDouble(0, 0, 0);

            // Фактическое значение
            double actual = series.getDouble(0, 0, i + windowSize);

            if (Math.abs(actual - predicted) > threshold) {
                anomalies.add(i + windowSize);
            }
        }

        return anomalies;
    }

    public static void printResults(double[] data, List<Integer> anomalies) {
        System.out.println("\nРезультаты обнаружения аномалий:");
        System.out.println("Порог аномалии: 4.0");
        System.out.println("Обнаруженные аномалии:");

        for (int i = 0; i < data.length; i++) {
            String marker = anomalies.contains(i) ? " <--- АНОМАЛИЯ" : "";
            System.out.printf("Позиция %2d: %5.2f%s\n", i, data[i], marker);
        }
    }
}