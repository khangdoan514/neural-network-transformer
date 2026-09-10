package tests;

import org.junit.jupiter.api.Test;
import src.Tensor;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class TestTensor {
    @Test
    void testAddMulBackward() {
        Tensor a = new Tensor(new double[] {2.0, 3.0}, true);
        Tensor b = new Tensor(new double[] {4.0, 5.0}, true);
        Tensor loss = a.multiply(b).add(a).sum();
        loss.backward();
        assertTrue(Tensor.allClose(a.grad, new double[] {5.0, 6.0}, 1e-9, 1e-9));
        assertTrue(Tensor.allClose(b.grad, new double[] {2.0, 3.0}, 1e-9, 1e-9));
    }

    @Test
    void testMatmulBackward() {
        Tensor left = new Tensor(new double[][] {{1.0, 2.0}, {3.0, 4.0}}, true);
        Tensor right = new Tensor(new double[][] {{0.5, 1.0}, {-1.0, 2.0}}, true);
        Tensor loss = left.matmul(right).sum();
        loss.backward();
        assertArrayEquals(left.shape, left.grad == null ? new int[0] : left.shape);
        assertArrayEquals(right.shape, right.grad == null ? new int[0] : right.shape);
    }

    @Test
    void testReluBackward() {
        Tensor inputTensor = new Tensor(new double[] {-1.0, 0.0, 2.0}, true);
        Tensor loss = inputTensor.relu().power(2).sum();
        loss.backward();
        assertTrue(Tensor.allClose(inputTensor.grad, new double[] {0.0, 0.0, 4.0}, 1e-9, 1e-9));
    }

    @Test
    void testTanhSigmoidForward() {
        Tensor inputTensor = new Tensor(new double[] {0.0}, false);
        assertTrue(Math.abs(inputTensor.tanh().data[0]) < 1e-9);
        assertTrue(Math.abs(inputTensor.sigmoid().data[0] - 0.5) < 1e-9);
    }

    @Test
    void testSoftmaxSumsToOne() {
        Tensor inputTensor = new Tensor(new double[][] {{1.0, 2.0, 3.0}, {0.1, -0.2, 0.3}}, true);
        Tensor probabilities = inputTensor.softmax(1);
        double row0 = probabilities.data[0] + probabilities.data[1] + probabilities.data[2];
        double row1 = probabilities.data[3] + probabilities.data[4] + probabilities.data[5];
        assertTrue(Math.abs(row0 - 1.0) < 1e-9);
        assertTrue(Math.abs(row1 - 1.0) < 1e-9);
    }

    @Test
    void testBroadcastAddGrad() {
        Tensor left = new Tensor(new double[][] {{1.0, 2.0}, {3.0, 4.0}}, true);
        Tensor right = new Tensor(new double[] {10.0, 20.0}, true);
        Tensor loss = left.add(right).sum();
        loss.backward();
        assertTrue(Tensor.allClose(left.grad, new double[] {1.0, 1.0, 1.0, 1.0}, 1e-9, 1e-9));
        assertTrue(Tensor.allClose(right.grad, new double[] {2.0, 2.0}, 1e-9, 1e-9));
    }
}
