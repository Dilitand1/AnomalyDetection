package org.example.v6;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler;
import org.nd4j.linalg.dataset.api.preprocessor.serializer.NormalizerSerializer;
import org.nd4j.linalg.factory.Nd4j;
import java.io.File;
import java.util.*;

public class ModelInference {

    public static void main(String[] args) {
        // 1. Загрузка модели и нормализатора
        ModelWrapper wrapper = loadModel("1cluster");

        // 2. Подготовка новых данных
        INDArray newData = prepareNewData(wrapper);

        // 3. Нормализация данных
        INDArray normalizedData = normalizeData(wrapper.normalizer, newData);

        // 4. Получение предсказаний
        INDArray predictions = wrapper.model.output(normalizedData);

        // 5. Вычисление ошибок
        INDArray diff = predictions.sub(normalizedData);
        INDArray errors = diff.mul(diff); // [samples x features]

        // 6. Обнаружение аномалий
        Map<Integer, List<AnomalyInfo>> anomalies = detectAnomaliesPerFeature(
                errors,
                newData,
                List.of("Service 1", "Service 2")
        );

        // 7. Вывод результатов
//        printResults(newData, anomalies);
        printAnomalies(anomalies);
    }

    static class ModelWrapper {
        MultiLayerNetwork model;
        NormalizerMinMaxScaler normalizer;
    }

    // Загрузка модели и нормализатора
    private static ModelWrapper loadModel(String serviceName) {
        ModelWrapper wrapper = new ModelWrapper();
        try {
            // Загрузка модели
            File modelFile = new File("models/" + serviceName + ".model");
            wrapper.model = ModelSerializer.restoreMultiLayerNetwork(modelFile);

            // Загрузка нормализатора
            File normFile = new File("models/" + serviceName + ".norm");
            NormalizerSerializer serializer = NormalizerSerializer.getDefault();
            wrapper.normalizer = (NormalizerMinMaxScaler) serializer.restore(normFile);

        } catch (Exception e) {
            throw new RuntimeException("Ошибка загрузки модели", e);
        }
        return wrapper;
    }

    // Подготовка новых данных (пример)
    private static INDArray prepareNewData(ModelWrapper wrapper) {
        // В реальном приложении данные должны быть в том же формате, что и обучающие
        INDArray newData = Nd4j.create(new double[][]{
                {10000.22, 1.3}
                ,{0.21, 1.3} // 1 образец, 3 признака
                ,{0.22, 1.3} // 1 образец, 3 признака
                ,{0.22, 1.3} // 1 образец, 3 признака
                ,{0.12, 1.3} // 1 образец, 3 признака
                ,{0.22, 2222.3} // 1 образец, 3 признака
                ,{0.32, 1.3} // 1 образец, 3 признака
                ,{111.22, 3.3} // 1 образец, 3 признака
                ,{0.22, 0.3} // 1 образец, 3 признака
                ,{0.22, 2.3} // 1 образец, 3 признака
        });

        // Проверка формы
//        if (newData.columns() != wrapper.model.getLayer(0).input().length().inputSize()) {
//            throw new IllegalArgumentException("Неверное количество признаков. Ожидается: "
//                    + wrapper.model.getLayer(0).inputSize());
//        }
        return newData;
    }

    // Нормализация данных
    private static INDArray normalizeData(NormalizerMinMaxScaler normalizer, INDArray data) {
        INDArray normalized = data.dup();
        normalizer.transform(normalized);
        return normalized;
    }

    // Обнаружение аномалий (порог = 95-й перцентиль)
    private static List<Integer> detectAnomalies(INDArray errors) {
        double threshold = calculateThreshold(errors);
        List<Integer> anomalies = new ArrayList<>();

        for (int i = 0; i < errors.rows(); i++) {
            if (errors.getRow(i).sumNumber().doubleValue() > threshold) {
                anomalies.add(i);
            }
        }
        return anomalies;
    }

//    private static double calculateThreshold(INDArray errors) {
//        double[] flatErrors = errors.ravel().toDoubleVector();
//        Arrays.sort(flatErrors);
//        return flatErrors[(int) (flatErrors.length * 0.95)];
//    }

        private static double calculateThreshold(INDArray errors) {
        INDArray flattened = errors.ravel();
        double mean = flattened.meanNumber().doubleValue();
        double std = flattened.stdNumber().doubleValue();
        return mean + 3 * std; // 3 сигмы от среднего
    }

    // Вывод результатов
    private static void printResults(INDArray data, List<Integer> anomalies) {
        System.out.println("Проанализировано образцов: " + data.rows());
        System.out.println("Обнаружено аномалий: " + anomalies.size());

        for (int row : anomalies) {
            System.out.println("\nАномалия в строке " + (row + 1));
            for (int col = 0; col < data.columns(); col++) {
                System.out.printf("Подсервис %d: %.2f%n",
                        col + 1, data.getDouble(row, col));
            }
        }
    }

    private static void printAnomalies(Map<Integer, List<AnomalyInfo>> anomalies) {
        System.out.println("\nДетализация аномалий:");
        for (Map.Entry<Integer, List<AnomalyInfo>> entry : anomalies.entrySet()) {
            System.out.printf("\nСтрока %d:%n", entry.getKey() + 1);
            for (AnomalyInfo info : entry.getValue()) {
                System.out.printf("?? Подсервис: %s%n", info.subServiceName);
                System.out.printf("?? Значение: %.6f%n", info.originalValue);
                System.out.printf("?? Уровень ошибки: %.6f%n%n", info.error);
            }
        }
    }

    private static class AnomalyInfo {
        String subServiceName;
        double originalValue;
        double error;

        public AnomalyInfo(String subServiceName, double value, double error) {
            this.subServiceName = subServiceName;
            this.originalValue = value;
            this.error = error;
        }
    }

    // Расчет порога для каждого подсервиса
    private static Map<Integer, Double> calculateThresholdsPerFeature(INDArray errors) {
        Map<Integer, Double> thresholds = new HashMap<>();
        for (int col = 0; col < errors.columns(); col++) {
            INDArray featureErrors = errors.getColumn(col);
            double[] values = featureErrors.toDoubleVector();
            Arrays.sort(values);
            thresholds.put(col, values[(int)(values.length * 0.95)]);
        }
        return thresholds;
    }

    private static Map<Integer, List<AnomalyInfo>> detectAnomaliesPerFeature(INDArray errors,
                                                                             INDArray originalData,
                                                                             List<String> subServiceNames) {
        // Определяем порог для каждого подсервиса отдельно
        Map<Integer, Double> thresholds = calculateThresholdsPerFeature(errors);

        Map<Integer, List<AnomalyInfo>> anomalies = new HashMap<>();

        for (int row = 0; row < errors.rows(); row++) {
            for (int col = 0; col < errors.columns(); col++) {
                double error = errors.getDouble(row, col);

                if (error > thresholds.get(col)) {
                    AnomalyInfo info = new AnomalyInfo(
                            subServiceNames.get(col),
                            originalData.getDouble(row, col),
                            error
                    );

                    anomalies.computeIfAbsent(row, k -> new ArrayList<>()).add(info);
                }
            }
        }
        return anomalies;
    }
}