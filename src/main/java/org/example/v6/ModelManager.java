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

    // ���������� �������������
    public static void saveNormalizer(String serviceName, NormalizerMinMaxScaler normalizer) {
        try {
            NormalizerSerializer serializer = NormalizerSerializer.getDefault();
            serializer.write(normalizer, MODEL_DIR + serviceName + ".norm");
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    // �������� �������������
    public static NormalizerMinMaxScaler loadNormalizer(String serviceName) {
        try {
            NormalizerSerializer serializer = NormalizerSerializer.getDefault();
            return (NormalizerMinMaxScaler) serializer.restore(MODEL_DIR + serviceName + ".norm");
        } catch (Exception e) {
            e.printStackTrace();
            return null;
        }
    }

    // ���������� ������ � �������������
    public static void saveModel(String serviceName,
                                 MultiLayerNetwork model,
                                 NormalizerMinMaxScaler normalizer) {
        try {
            new File(MODEL_DIR).mkdirs();

            // ��������� ������
            ModelSerializer.writeModel(model,
                    new File(MODEL_DIR + serviceName + ".model"),
                    true);

            // ��������� ������������
            saveNormalizer(serviceName, normalizer);

        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    // �������� ������ � �������������
    public static ModelWrapper loadModel(String serviceName) {
        ModelWrapper wrapper = new ModelWrapper();
        try {
            // �������� ������
            File modelFile = new File(MODEL_DIR + serviceName + ".model");
            if (modelFile.exists()) {
                wrapper.model = ModelSerializer.restoreMultiLayerNetwork(modelFile);
            }

            // �������� �������������
            wrapper.normalizer = loadNormalizer(serviceName);

        } catch (Exception e) {
            e.printStackTrace();
        }
        return wrapper;
    }

    // ������������� �������������
    public static INDArray preprocessData(INDArray rawData, NormalizerMinMaxScaler normalizer) {
        INDArray data = rawData.dup();
        normalizer.transform(data);
        return data;
    }

}