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
        assertEquals(2, zeros.shape[0]);
        assertEquals(3, zeros.shape[1]);
    }

    @Test
    void testMatrixConstructor() {
        Tensor matrix = new Tensor(new double[][] {{1.0, 2.0}, {3.0, 4.0}}, false);
        assertEquals(2, matrix.shape[0]);
        assertEquals(2, matrix.shape[1]);
        assertEquals(4.0, matrix.data[3]);
    }

    @Test
    void testFlattenUtility() {
        double[] flat = Tensor.flatten(new double[][] {{1.0, 2.0}, {3.0, 4.0}});
        assertTrue(Tensor.allClose(flat, new double[] {1.0, 2.0, 3.0, 4.0}, 1e-9, 1e-9));
    }
}
