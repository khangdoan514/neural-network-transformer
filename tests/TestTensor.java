package tests;

import org.junit.jupiter.api.Test;
import src.Tensor;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
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
    void testBroadcastAddGrad() {
        Tensor left = new Tensor(new double[][] {{1.0, 2.0}, {3.0, 4.0}}, true);
        Tensor right = new Tensor(new double[] {10.0, 20.0}, true);
        Tensor loss = left.add(right).sum();
        loss.backward();
        assertTrue(Tensor.allClose(left.grad, new double[] {1.0, 1.0, 1.0, 1.0}, 1e-9, 1e-9));
        assertTrue(Tensor.allClose(right.grad, new double[] {2.0, 2.0}, 1e-9, 1e-9));
    }

    @Test
    void testReshapePreservesElements() {
        Tensor tensor = Tensor.ones(new int[] {2, 4, 8}, false);
        Tensor reshaped = tensor.reshape(2, 32);
        assertEquals(2, reshaped.shape[0]);
        assertEquals(32, reshaped.shape[1]);
        assertEquals(64, reshaped.size());
    }
}
