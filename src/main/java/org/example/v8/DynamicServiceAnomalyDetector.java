package org.example.v8;

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
import java.util.*;

public class DynamicServiceAnomalyDetector {

    private static final int EMBEDDING_SIZE = 64;
    private static final int HASH_SPACE = 1000;
    private MultiLayerNetwork model;
    private Map<String, Integer> metricIndexMap = new HashMap<>();

    // Инициализация модели
    public void initializeModel() {
        NeuralNetConfiguration.ListBuilder builder = new NeuralNetConfiguration.Builder()
                .seed(1234)
                .updater(new Adam(0.001))
                .list()
                .layer(new DenseLayer.Builder()
                        .nIn(EMBEDDING_SIZE + 1) // +1 для значения
                        .nOut(128)
                        .activation(Activation.RELU)
                        .build());

        builder.layer(new OutputLayer.Builder()
                .nIn(128)
                .nOut(1)
                .activation(Activation.SIGMOID)
                .lossFunction(LossFunctions.LossFunction.XENT)
                .build());

        model = new MultiLayerNetwork(builder.build());
        model.init();
    }

    // Преобразование метрик в вектор с правильной размерностью
    private INDArray processMetrics(Map<String, Double> metrics) {
        INDArray features = Nd4j.zeros(1, EMBEDDING_SIZE + 1); // [1, 65]

        for (Map.Entry<String, Double> entry : metrics.entrySet()) {
            int hash = getMetricHash(entry.getKey());
            double value = entry.getValue();

            // Обновляем embedding часть
            int embeddingIndex = hash % EMBEDDING_SIZE;
            features.putScalar(0, embeddingIndex,
                    features.getDouble(0, embeddingIndex) + value);

            // Сохраняем оригинальное значение
            features.putScalar(0, EMBEDDING_SIZE, value);
        }
        return features;
    }

    // Хеширование имени метрики
    private int getMetricHash(String metricName) {
        return metricIndexMap.computeIfAbsent(metricName,
                k -> Math.abs(k.hashCode()) % HASH_SPACE);
    }

    // Обучение модели
    public void train(List<Map<String, Double>> metricsList, List<Boolean> labels) {
        int batchSize = metricsList.size();
        INDArray featureMatrix = Nd4j.zeros(batchSize, EMBEDDING_SIZE + 1);
        INDArray labelMatrix = Nd4j.zeros(batchSize, 1);

        for (int i = 0; i < batchSize; i++) {
            INDArray features = processMetrics(metricsList.get(i));
            featureMatrix.putRow(i, features);
            labelMatrix.putScalar(i, 0, labels.get(i) ? 1.0 : 0.0);
        }

        DataSet dataset = new DataSet(featureMatrix, labelMatrix);
        model.fit(dataset);
    }

    // Предсказание аномалий
    public double predict(Map<String, Double> metrics) {
        INDArray input = processMetrics(metrics);
        return model.output(input).getDouble(0);
    }

    public static void main(String[] args) {
        DynamicServiceAnomalyDetector detector = new DynamicServiceAnomalyDetector();
        detector.initializeModel();

        // Пример данных для обучения
        List<Map<String, Double>> trainData = Arrays.asList(
                new HashMap<>() {{ put("CPU_Usage", 25.0); put("Memory_Used", 45.0); }},
                new HashMap<>() {{ put("Latency", 120.0); put("Requests", 150.0); }}
        );

        List<Boolean> labels = Arrays.asList(false, true);
        detector.train(trainData, labels);

        // Пример предсказания
        Map<String, Double> testMetrics = new HashMap<>() {{
            put("New_Metric", 95.0);
            put("CPU_Usage", 85.0);
            put("Requests", 85.0);
            put("Requests", 85.0);
            put("Requests", 5.0);
            put("Requests", 85.0);
        }};

        double anomalyProbability = detector.predict(testMetrics);
        System.out.printf("Вероятность аномалии: %.2f%%\n", anomalyProbability * 100);
    }
}