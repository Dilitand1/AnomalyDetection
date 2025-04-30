package org.example.v7;

import org.nd4j.linalg.api.ndarray.INDArray;

public class DataNormalizer {
    private final INDArray min;
    private final INDArray max;
    private final double epsilon = 1e-10;

    public DataNormalizer(INDArray data) {
        this.min = data.min(0); // �������� �� ������� �������
        this.max = data.max(0); // ��������� �� ������� �������
        this.min.addi(epsilon); // �������� ������� �� ����
    }

    public INDArray normalize(INDArray data) {
        INDArray range = max.sub(min);
        return data.sub(min).div(range);
    }

    public INDArray denormalize(INDArray normalizedData) {
        INDArray range = max.sub(min);
        return normalizedData.mul(range).add(min);
    }
}