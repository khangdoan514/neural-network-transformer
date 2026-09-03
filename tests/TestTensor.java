package tests;

import org.junit.jupiter.api.Test;
import src.Tensor;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class TestTensor {
    @Test
    void testZerosOnesShape() {
        Tensor zeros = Tensor.zeros(new int[] {2, 3}, false);
        assertEquals(6, zeros.size());
        assertEquals(2, zeros.ndim());
    }

    @Test
    void testMatrixConstructor() {
        Tensor matrix = new Tensor(new double[][] {{1.0, 2.0}, {3.0, 4.0}}, false);
        assertEquals(2, matrix.shape[0]);
        assertEquals(2, matrix.shape[1]);
    }

    @Test
    void testAddMulBackward() {
        Tensor a = new Tensor(new double[] {2.0, 3.0}, true);
        Tensor b = new Tensor(new double[] {4.0, 5.0}, true);
        Tensor output = a.multiply(b).add(a);
        output.backward(new double[] {1.0, 1.0});
        assertTrue(Tensor.allClose(a.grad, new double[] {5.0, 6.0}, 1e-9, 1e-9));
        assertTrue(Tensor.allClose(b.grad, new double[] {2.0, 3.0}, 1e-9, 1e-9));
    }

    @Test
    void testBroadcastAddGrad() {
        Tensor left = new Tensor(new double[][] {{1.0, 2.0}, {3.0, 4.0}}, true);
        Tensor right = new Tensor(new double[] {10.0, 20.0}, true);
        Tensor output = left.add(right);
        output.backward(new double[] {1.0, 1.0, 1.0, 1.0});
        assertTrue(Tensor.allClose(left.grad, new double[] {1.0, 1.0, 1.0, 1.0}, 1e-9, 1e-9));
        assertTrue(Tensor.allClose(right.grad, new double[] {2.0, 2.0}, 1e-9, 1e-9));
    }
}
