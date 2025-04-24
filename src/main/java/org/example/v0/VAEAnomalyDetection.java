package org.example.v0;

import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.writable.Writable;
import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.CenterLossOutputLayer;
import org.deeplearning4j.nn.conf.layers.variational.GaussianReconstructionDistribution;
import org.deeplearning4j.nn.conf.layers.variational.VariationalAutoencoder;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler;
import org.nd4j.linalg.dataset.api.preprocessor.serializer.NormalizerSerializer;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.util.*;
import java.util.stream.Collectors;

public class VAEAnomalyDetection {

    public static void main(String[] args) {
        try {
            // 1. Загрузка данных
            String filePath = "C:\\Users\\Dilit\\IdeaProjects\\DeepLearning4j\\src\\main\\resources\\timeseries.csv";
            INDArray timeSeriesData = loadTimeSeriesFromCSV(filePath);

            // 2. Подготовка датасета
            int windowSize = 1;
            DataSetIterator trainData = prepareDataset(timeSeriesData, windowSize);

            // 3. Проверка, что данные не пустые
            if (!trainData.hasNext()) {
                throw new IllegalStateException("Загружен пустой набор данных");
            }

            // 4. Нормализация данных (исправленная версия)
            NormalizerMinMaxScaler normalizer = new NormalizerMinMaxScaler();

            // Сначала собираем все данные для нормализации
            List<DataSet> allData = new ArrayList<>();
            while (trainData.hasNext()) {
                allData.add(trainData.next());
            }
            trainData.reset();

            // Создаем временный DataSet для нормализации
            DataSet tempDataSet = DataSet.merge(allData);
            normalizer.fit(tempDataSet);

            // Устанавливаем нормализатор для итератора
            trainData.setPreProcessor(normalizer);

            // 5. Сохранение нормализатора для последующего использования
            NormalizerSerializer serializer = NormalizerSerializer.getDefault();
            serializer.write(normalizer, new File("normalizer.bin"));

            System.out.println("Минимальные значения: " + normalizer.getMin());
            System.out.println("Максимальные значения: " + normalizer.getMax());

            // Далее идет создание и обучение модели...


            // 4. Создание и обучение автоэнкодера
            var vae = createVAE(windowSize);
            vae.setListeners(new ScoreIterationListener(50));

            int numEpochs = 100;
            trainVAE(vae, trainData, numEpochs);

            // 5. Обнаружение аномалий
            detectAnomalies(vae, timeSeriesData, windowSize, normalizer);

        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    static void detectAnomalies(ComputationGraph vae, INDArray timeSeries,
                                int windowSize, NormalizerMinMaxScaler normalizer) {
        List<Double> reconstructionErrors = new ArrayList<>();
        List<Integer> anomalyPositions = new ArrayList<>();

        // 1. Вычисление ошибок реконструкции
        for (int i = 0; i < timeSeries.length() - windowSize; i++) {
            // Получаем текущее окно данных
            INDArray window = timeSeries.get(NDArrayIndex.interval(i, i + windowSize));
            window = window.reshape(1, windowSize); // Формат [1, windowSize]

            // Нормализация данных
            INDArray normalizedWindow = window.dup();
            normalizer.transform(normalizedWindow);

            // Получаем реконструкцию от VAE
            INDArray[] outputs = vae.output(normalizedWindow);
            INDArray reconstructed = outputs[0]; // Первый выход - реконструкция

            // Обратная нормализация
            normalizer.revertFeatures(reconstructed);

            // Вычисляем ошибку реконструкции (MSE)
            double error = reconstructed.sub(window).norm2Number().doubleValue();
            reconstructionErrors.add(error);
        }

        // 2. Статистический анализ ошибок
        double mean = reconstructionErrors.stream().mapToDouble(d -> d).average().orElse(0);
        double std = Math.sqrt(reconstructionErrors.stream()
                .mapToDouble(e -> Math.pow(e - mean, 2))
                .average()
                .orElse(0));

        // 3. Определение порога (3 сигмы)
        double threshold = mean + 3 * std;
        System.out.printf("\nAnomaly threshold: %.4f (mean=%.4f, std=%.4f)\n",
                threshold, mean, std);

        // 4. Поиск аномалий
        for (int i = 0; i < reconstructionErrors.size(); i++) {
            if (reconstructionErrors.get(i) > threshold) {
                // Записываем все точки в этом окне как аномальные
                for (int j = 0; j < windowSize; j++) {
                    if (i + j < timeSeries.length()) {
                        anomalyPositions.add(i + j);
                    }
                }
            }
        }

        // 5. Удаление дубликатов и сортировка
        anomalyPositions = anomalyPositions.stream()
                .distinct()
                .sorted()
                .collect(Collectors.toList());

        // 6. Визуализация
        visualizeResults(timeSeries, anomalyPositions, reconstructionErrors, threshold, windowSize);
    }

    private static void visualizeResults(INDArray timeSeries, List<Integer> anomalies,
                                         List<Double> errors, double threshold, int windowSize) {
        System.out.println("\nDetection results:");
        System.out.println("Total anomalies detected: " + anomalies.size());

        // Вывод временного ряда с отметками аномалий
        System.out.println("\nTime series with anomalies:");
        int displayLength = (int) Math.min(100, timeSeries.length());
        for (int i = 0; i < displayLength; i++) {
            String marker = anomalies.contains(i) ? " <<< ANOMALY" : "";
            System.out.printf("%4d: %8.4f%s\n", i, timeSeries.getDouble(i), marker);
        }

        // Вывод графика ошибок
        System.out.println("\nReconstruction errors:");
        int errorDisplayLength = Math.min(50, errors.size());
        for (int i = 0; i < errorDisplayLength; i++) {
            String mark = errors.get(i) > threshold ? " !!!" : "";
            System.out.printf("Window %3d: error=%.4f%s\n", i, errors.get(i), mark);
        }
    }

    // Метод для обучения VAE
    public static void trainVAE(ComputationGraph vae, DataSetIterator trainData, int numEpochs) {
        for (int epoch = 0; epoch < numEpochs; epoch++) {
            while (trainData.hasNext()) {
                DataSet ds = trainData.next();
                // Для VAE вход и выход должны быть одинаковы
                vae.fit(new DataSet(ds.getFeatures(), ds.getFeatures()));
            }
            trainData.reset();
            System.out.println("Epoch " + (epoch + 1) + " complete");
        }
    }

    public static ComputationGraph createVAE(int inputSize) {
        // Параметры модели
        int latentSize = 10;
        int encoderLayerSize = 128;
        int decoderLayerSize = 128;

        // 1. Конфигурация VAE
        VariationalAutoencoder vaeLayer = new VariationalAutoencoder.Builder()
                .encoderLayerSizes(encoderLayerSize, encoderLayerSize)
                .decoderLayerSizes(decoderLayerSize, decoderLayerSize)
                .pzxActivationFunction(Activation.IDENTITY)
                .reconstructionDistribution(new GaussianReconstructionDistribution(Activation.IDENTITY))
                .nIn(inputSize)
                .nOut(latentSize)
                .build();

        // 2. Правильная конфигурация ComputationGraph
        ComputationGraphConfiguration config = new NeuralNetConfiguration.Builder()
                .seed(12345)
                .updater(new Adam(0.001))
                .graphBuilder()
                .addInputs("input")
                // Добавляем VAE как промежуточный слой
                .addLayer("vae", vaeLayer, "input")
                // Добавляем выходной слой для реконструкции
                .addLayer("output", new CenterLossOutputLayer.Builder()
                        .nIn(latentSize)
                        .nOut(inputSize)
                        .activation(Activation.IDENTITY)
                        .lossFunction(LossFunctions.LossFunction.MSE)
                        .build(), "vae")
                .setOutputs("output") // Указываем правильный выход
                .build();

        ComputationGraph vae = new ComputationGraph(config);
        vae.init();
        return vae;
    }

    private static DataSetIterator prepareDataset(INDArray timeSeries, int windowSize) {
        List<DataSet> dataSets = new ArrayList<>();

        for (int i = 0; i < timeSeries.length() - windowSize; i++) {
            INDArray window = timeSeries.get(NDArrayIndex.interval(i, i + windowSize));
            window = window.reshape(1, windowSize);

            // Для автоэнкодера вход и выход одинаковы
            dataSets.add(new DataSet(window, window));
        }

        System.out.println("Загружено " + timeSeries.length() + " точек временного ряда");
        System.out.println("Создано " + dataSets.size() + " окон для обучения");

        return new ListDataSetIterator<>(dataSets, 32); // Batch size 32
    }

    private static INDArray loadTimeSeriesFromCSV(String filePath) throws Exception {
        CSVRecordReader recordReader = new CSVRecordReader(0, ',');
        recordReader.initialize(new FileSplit(new File(filePath)));

        List<double[]> allData = new ArrayList<>();
        while (recordReader.hasNext()) {
//            var record = recordReader.next();
            List<Writable> writables = recordReader.next();

            double[] values = new double[writables.size()];
            for (int i = 0; i < writables.size(); i++) {
                values[i] = writables.get(i).toDouble();
            }
            allData.add(values);
        }

        int rows = allData.size();
        int cols = allData.get(0).length;

        INDArray timeSeries = Nd4j.zeros(rows, cols);
        for (int i = 0; i < rows; i++) {
            timeSeries.putRow(i, Nd4j.create(allData.get(i)));
        }

        return timeSeries;
    }

}