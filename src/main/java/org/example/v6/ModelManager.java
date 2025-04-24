package org.example.v6;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler;
import org.nd4j.linalg.dataset.api.preprocessor.serializer.NormalizerSerializer;
import java.io.File;

public class ModelManager {
    private static final String MODEL_DIR = "models/";

    public static class ModelWrapper {
        public MultiLayerNetwork model;
        public NormalizerMinMaxScaler normalizer;
    }

    // Сохранение нормализатора
    public static void saveNormalizer(String serviceName, NormalizerMinMaxScaler normalizer) {
        try {
            NormalizerSerializer serializer = NormalizerSerializer.getDefault();
            serializer.write(normalizer, MODEL_DIR + serviceName + ".norm");
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    // Загрузка нормализатора
    public static NormalizerMinMaxScaler loadNormalizer(String serviceName) {
        try {
            NormalizerSerializer serializer = NormalizerSerializer.getDefault();
            return (NormalizerMinMaxScaler) serializer.restore(MODEL_DIR + serviceName + ".norm");
        } catch (Exception e) {
            e.printStackTrace();
            return null;
        }
    }

    // Сохранение модели и нормализатора
    public static void saveModel(String serviceName,
                                 MultiLayerNetwork model,
                                 NormalizerMinMaxScaler normalizer) {
        try {
            new File(MODEL_DIR).mkdirs();

            // Сохраняем модель
            ModelSerializer.writeModel(model,
                    new File(MODEL_DIR + serviceName + ".model"),
                    true);

            // Сохраняем нормализатор
            saveNormalizer(serviceName, normalizer);

        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    // Загрузка модели и нормализатора
    public static ModelWrapper loadModel(String serviceName) {
        ModelWrapper wrapper = new ModelWrapper();
        try {
            // Загрузка модели
            File modelFile = new File(MODEL_DIR + serviceName + ".model");
            if (modelFile.exists()) {
                wrapper.model = ModelSerializer.restoreMultiLayerNetwork(modelFile);
            }

            // Загрузка нормализатора
            wrapper.normalizer = loadNormalizer(serviceName);

        } catch (Exception e) {
            e.printStackTrace();
        }
        return wrapper;
    }

    // Использование нормализатора
    public static INDArray preprocessData(INDArray rawData, NormalizerMinMaxScaler normalizer) {
        INDArray data = rawData.dup();
        normalizer.transform(data);
        return data;
    }

}