package org.example.v2;

import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.nd4j.common.primitives.Pair;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.ops.transforms.Transforms;

public class ActivationSampling implements IActivation {

    @Override
    public INDArray getActivation(INDArray input, boolean training) {
        // ��������� ���� �� mu � logVar
        int halfSize = input.columns() / 2;
        INDArray mu = input.get(NDArrayIndex.all(), NDArrayIndex.interval(0, halfSize));
        INDArray logVar = input.get(NDArrayIndex.all(), NDArrayIndex.interval(halfSize, input.columns()));

        // ���������� ��� ? ~ N(0,1)
        INDArray epsilon = Nd4j.randn(mu.shape());

        // z = ? + exp(logVar/2) * ?
        INDArray sigma = Transforms.exp(logVar.div(2));
        return mu.add(sigma.mul(epsilon));
    }

    @Override
    public Pair<INDArray, INDArray> backprop(INDArray input, INDArray epsilon) {
        // ��������� ��� mu � logVar
        int halfSize = input.columns() / 2;
        INDArray dLdz = epsilon;

        INDArray dLdmu = dLdz.dup();
        INDArray dLdlogVar = input.get(NDArrayIndex.all(), NDArrayIndex.interval(halfSize, input.columns()))
                .div(2)
                .mul(dLdz)
                .mul(Transforms.exp(input.get(NDArrayIndex.all(),
                        NDArrayIndex.interval(halfSize, input.columns()))));

        // ���������� ���������
        INDArray combinedGradients = Nd4j.hstack(dLdmu, dLdlogVar);
        return new Pair<>(combinedGradients, null); // ������ ������� - ��������� ���������� (� ��� �� ���)
    }

    @Override
    public int numParams(int inputSize) {
        return 0; // � ����� ���� ��� ��������� ����������
    }

    @Override
    public String toString() {
        return "ActivationSampling()";
    }
}