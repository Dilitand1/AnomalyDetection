package org.example.v2;
import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVRecord;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.CenterLossOutputLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.variational.GaussianReconstructionDistribution;
import org.deeplearning4j.nn.conf.layers.variational.VariationalAutoencoder;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.graph.vertex.impl.PreprocessorVertex;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.io.FileReader;
import java.io.IOException;
import java.io.Reader;
import java.util.ArrayList;
import java.util.List;

public class VAEAnomalyDetection {

    public static void main(String[] args) throws IOException {
        // 1. Загрузка данных из CSV
        List<Double> data = loadDataFromCSV("C:\\Users\\Dilit\\IdeaProjects\\DeepLearning4j\\src\\main\\resources\\timeseries.csv");
        double[] values = data.stream().mapToDouble(Double::doubleValue).toArray();

        // 2. Нормализация данных (0-1)
        double[] normalizedValues = normalize(values);

        // 3. Подготовка DataSet
        int batchSize = 32;
        int timeSeriesLength = 2; // Размер окна временного ряда
        DataSet dataSet = createDataSet(normalizedValues, timeSeriesLength, batchSize);

        // 4. Создание и обучение VAE
        int inputSize = timeSeriesLength;
        int latentDim = 3; // Размер латентного пространства
        ComputationGraph vae = buildVAE(inputSize, latentDim);
        vae.init();

        // 5. Обучение модели
        trainModel(vae, dataSet, 50); // 50 эпох

        // 6. Поиск аномалий
        List<Double> reconstructionErrors = computeReconstructionErrors(vae, normalizedValues, timeSeriesLength);
        double threshold = computeThreshold(reconstructionErrors, 0.9); // 90-й перцентиль

        // 7. Вывод аномалий
        System.out.println("Порог ошибки реконструкции: " + threshold);
        System.out.println("Аномалии:");
        for (int i = 0; i < reconstructionErrors.size(); i++) {
            if (reconstructionErrors.get(i) > threshold) {
                System.out.printf("Индекс: %d, Ошибка: %.4f, Значение: %.2f%n",
                        i, reconstructionErrors.get(i), values[i]);
            }
        }
    }

    // Загрузка данных из CSV
    private static List<Double> loadDataFromCSV(String filePath) throws IOException {
        List<Double> data = new ArrayList<>();
        try (Reader reader = new FileReader(filePath);
             CSVParser csvParser = new CSVParser(reader, CSVFormat.DEFAULT.withFirstRecordAsHeader())) {
            for (CSVRecord record : csvParser) {
                double value = Double.parseDouble(record.get("value"));
                data.add(value);
            }
        }
        return data;
    }

    // Нормализация данных (0-1)
    private static double[] normalize(double[] data) {
        double min = Double.MAX_VALUE;
        double max = Double.MIN_VALUE;
        for (double v : data) {
            if (v < min) min = v;
            if (v > max) max = v;
        }
        double[] normalized = new double[data.length];
        for (int i = 0; i < data.length; i++) {
            normalized[i] = (data[i] - min) / (max - min);
        }
        return normalized;
    }

    // Создание DataSet для временного ряда
    private static DataSet createDataSet(double[] values, int windowSize, int batchSize) {
        int numSamples = values.length - windowSize;
        INDArray input = Nd4j.create(numSamples, windowSize);
        for (int i = 0; i < numSamples; i++) {
            for (int j = 0; j < windowSize; j++) {
                input.putScalar(i, j, values[i + j]);
            }
        }
        return new DataSet(input, input); // Автоэнкодер: вход = выход
    }

    // Построение VAE
    private static ComputationGraph buildVAE(int inputSize, int latentDim) {
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

    // Обучение модели
    private static void trainModel(ComputationGraph vae, DataSet dataSet, int epochs) {
        for (int i = 0; i < epochs; i++) {
            vae.fit(dataSet);
            System.out.println("Эпоха " + i + ", Loss: " + vae.score());
        }
    }

    // Вычисление ошибок реконструкции
    private static List<Double> computeReconstructionErrors(ComputationGraph vae, double[] values, int windowSize) {
        List<Double> errors = new ArrayList<>();
        for (int i = 0; i < values.length - windowSize; i++) {
            INDArray input = Nd4j.create(1, windowSize);
            for (int j = 0; j < windowSize; j++) {
                input.putScalar(0, j, values[i + j]);
            }
            INDArray output = vae.output(input)[0];
            //double error = Nd4j.sum(Nd4j.pow(input.sub(output), 2)).getDouble(0); // MSE
            double error = Transforms.pow(input.sub(output), 2).meanNumber().doubleValue();; // MSE
            errors.add(error);
        }
        return errors;
    }

    // Вычисление порога (перцентиль)
    private static double computeThreshold(List<Double> errors, double percentile) {
        List<Double> sorted = new ArrayList<>(errors);
        sorted.sort(Double::compare);
        int index = (int) (percentile * sorted.size());
        return sorted.get(index);
    }
}