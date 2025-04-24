package org.example.v4;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;

public class MinMaxNormalizer extends NormalizerStandardize {
    private final double min;
    private final double max;

    public MinMaxNormalizer(double min, double max) {
        this.min = min;
        this.max = max;
    }

    @Override
    public void transform(INDArray input) {
        input.subi(min).divi(max - min);
    }
}