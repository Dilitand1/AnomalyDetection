package org.example.v4;

import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.*;
import java.util.*;

public class DeepAnomalyDetection {
    private static final String MODEL_DIR = "models/";
    private static final int MIN_TRAIN_SAMPLES = 10;

    private static final Map<String, ModelWrapper> modelCache = new HashMap<>();

    static class ModelWrapper {
        MultiLayerNetwork model;
        double minValue;
        double maxValue;
    }

    public static void main(String[] args) {
        Map<String, List<Double>> data = readData("C:\\Users\\Dilit\\IdeaProjects\\DeepLearning4j\\src\\main\\resources\\timeseriesWithnames.csv");
        detectAnomaliesWithDL(data);
    }

    // Чтение данных (аналогично предыдущему примеру)
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

    private static void detectAnomaliesWithDL(Map<String, List<Double>> data) {
        // Создаем директорию для моделей
        new File(MODEL_DIR).mkdirs();

        for (Map.Entry<String, List<Double>> entry : data.entrySet()) {
            String name = entry.getKey();
            List<Double> values = entry.getValue();

            if (values.size() < MIN_TRAIN_SAMPLES) {
                System.out.println("Недостаточно данных для: " + name);
                continue;
            }

            ModelWrapper wrapper = loadOrCreateModel(name, values);
            List<Double> anomalies = detect(name, values, wrapper);

            System.out.println("DL Аномалии для " + name + ": " + anomalies);
        }
    }

    private static List<Double> detect(String name, List<Double> values, ModelWrapper wrapper) {
        List<Double> normalized = normalize(values, wrapper.minValue, wrapper.maxValue);
        INDArray input = Nd4j.create(normalized.stream()
                .mapToDouble(Double::doubleValue)
                .toArray(), new int[]{values.size(), 1});

        INDArray output = wrapper.model.output(input);
        INDArray diff = output.sub(input);
        INDArray errors = diff.mul(diff);

        double threshold = getPercentile(errors.toDoubleVector(), 95);

        List<Double> anomalies = new ArrayList<>();
        for (int i = 0; i < values.size(); i++) {
            if (errors.getDouble(i) > threshold) {
                anomalies.add(values.get(i));
            }
        }
        return anomalies;
    }

    // Расчет перцентиля (аналогично предыдущему примеру)
    private static double getPercentile(double[] data, double percentile) {
        Arrays.sort(data);
        int index = (int) Math.ceil((percentile / 100) * data.length) - 1;
        return data[Math.min(index, data.length - 1)];
    }

    private static ModelWrapper loadOrCreateModel(String name, List<Double> values) {
        if (modelCache.containsKey(name)) {
            return modelCache.get(name);
        }

        ModelWrapper wrapper = new ModelWrapper();
        File modelFile = new File(MODEL_DIR + name + ".zip");
        File propsFile = new File(MODEL_DIR + name + ".props");

        try {
            if (modelFile.exists() && propsFile.exists()) {
                // Загрузка существующей модели и параметров
                wrapper = loadModelAndParams(name);
            } else {
                // Создание новой модели
                wrapper.model = createAutoencoder();
                wrapper.minValue = Collections.min(values);
                wrapper.maxValue = Collections.max(values);
                trainAndSaveModel(name, values, wrapper);
            }
        } catch (Exception e) {
            e.printStackTrace();
        }

        modelCache.put(name, wrapper);
        return wrapper;
    }

    private static ModelWrapper loadModelAndParams(String name) {
        ModelWrapper wrapper = new ModelWrapper();
        try {
            // Загружаем модель
            wrapper.model = ModelSerializer.restoreMultiLayerNetwork(MODEL_DIR + name + ".zip");

            // Загружаем параметры нормализации
            Properties props = new Properties();
            try (InputStream in = new FileInputStream(MODEL_DIR + name + ".props")) {
                props.load(in);
                wrapper.minValue = Double.parseDouble(props.getProperty("min"));
                wrapper.maxValue = Double.parseDouble(props.getProperty("max"));
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
        return wrapper;
    }

    private static MultiLayerNetwork createAutoencoder() {
        MultiLayerNetwork model = new MultiLayerNetwork(
                new NeuralNetConfiguration.Builder()
                        // 1. Инициализация генератора случайных чисел для воспроизводимости
                        .seed(123)

                        // 2. Настройка оптимизатора (Adam с learning rate = 0.01)
                        .updater(new Adam(0.01))

                        // 3. Метод инициализации весов (Xavier для равномерного распределения)
                        .weightInit(WeightInit.XAVIER)

                        // 4. Начало описания слоев сети
                        .list()

                        // 5. Первый скрытый слой (энкодер)
                        .layer(new DenseLayer.Builder()
                                .nIn(1)                // 5.1. 1 входной нейрон (значение)
                                .nOut(3)              // 5.2. 3 нейрона в слое (сжатие)
                                .activation(Activation.RELU) // 5.3. Функция активации
                                .build())

                        // 6. Выходной слой (декодер)
                        .layer(new OutputLayer.Builder()
                                .nIn(3)                // 6.1. 3 входа (из предыдущего слоя)
                                .nOut(1)               // 6.2. 1 выход (реконструкция значения)
                                .lossFunction(LossFunctions.LossFunction.MSE) // 6.3. Функция потерь
                                .activation(Activation.IDENTITY) // 6.4. Линейная активация
                                .build())

                        // 7. Финальная сборка конфигурации
                        .build()
        );
        return model;
    }

    private static void trainAndSaveModel(String name, List<Double> values, ModelWrapper wrapper) {
        // Нормализация данных
        List<Double> normalized = normalize(values, wrapper.minValue, wrapper.maxValue);

        // Создание dataset
        INDArray input = Nd4j.create(normalized.stream()
                .mapToDouble(Double::doubleValue)
                .toArray(), new int[]{values.size(), 1});

        DataSet dataSet = new DataSet(input, input);

        // Обучение
        for (int i = 0; i < 100; i++) {
            wrapper.model.fit(dataSet);
        }

        // Сохранение модели и параметров нормализации
        try {
            ModelSerializer.writeModel(wrapper.model, MODEL_DIR + name + ".zip", true);
            ModelSerializer.addNormalizerToModel(
                    new File(MODEL_DIR + name + ".zip"),
                    new MinMaxNormalizer(wrapper.minValue, wrapper.maxValue)
            );
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private static List<Double> normalize(List<Double> data, double min, double max) {
        List<Double> normalized = new ArrayList<>();
        for (Double value : data) {
            normalized.add((value - min) / (max - min));
        }
        return normalized;
    }
}