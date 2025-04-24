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

    // Обнаружение аномалий с использованием автоэнкодера
    private static void detectAnomaliesWithDL(Map<String, List<Double>> data) {
        for (Map.Entry<String, List<Double>> entry : data.entrySet()) {
            String name = entry.getKey();
            List<Double> values = entry.getValue();

            if (values.size() < 10) { // Минимум 10 значений для обучения
                System.out.println("Недостаточно данных для: " + name);
                continue;
            }

            // Нормализация данных
            DataNormalization normalization = new DataNormalization(values);
            List<Double> normalizedValues = normalization.normalize(values);

            // Создание datasets
            INDArray input = Nd4j.create(normalizedValues.stream()
                    .mapToDouble(Double::doubleValue)
                    .toArray(), new int[]{values.size(), 1});

            DataSet dataSet = new DataSet(input, input); // Autoencoder: input = output

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

            // 8. Инициализация весов сети
            model.init();

            // Обучение модели
            for (int i = 0; i < 100; i++) { // 100 эпох
                model.fit(dataSet);
            }

            // Получение ошибок реконструкции
            INDArray output = model.output(input);
            INDArray diff = output.sub(input);          // Вычисляем разницу
            INDArray errors = diff.mul(diff);          // Квадрат ошибки (замена pow(2))

            // Определение порога для аномалий (95-й перцентиль)
            double threshold = getPercentile(errors.toDoubleVector(), 95);

            // Поиск аномалий
            List<Double> anomalies = new ArrayList<>();
            for (int i = 0; i < values.size(); i++) {
                if (errors.getDouble(i) > threshold) {
                    anomalies.add(values.get(i)); // Возвращаем оригинальные значения
                }
            }

            System.out.println("DL Аномалии для " + name + ": " + anomalies);
        }
    }

    // Вспомогательный класс для нормализации данных
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

    // Расчет перцентиля (аналогично предыдущему примеру)
    private static double getPercentile(double[] data, double percentile) {
        Arrays.sort(data);
        int index = (int) Math.ceil((percentile / 100) * data.length) - 1;
        return data[Math.min(index, data.length - 1)];
    }
}