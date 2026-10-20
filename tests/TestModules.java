package tests;

import java.util.List;

import org.junit.jupiter.api.Test;
import src.Adam;
import src.BCEWithLogitsLoss;
import src.CrossEntropyLoss;
import src.Gradcheck;
import src.Linear;
import src.MSELoss;
import src.ReLU;
import src.SGD;
import src.Sequential;
import src.Softmax;
import src.Tensor;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class TestModules {
    @Test
    void testLinearShape() {
        Linear layer = new Linear(4, 3);
        Tensor inputTensor = Tensor.ones(new int[] {2, 4}, false);
        Tensor output = layer.forward(inputTensor);
        assertEquals(2, output.shape[0]);
        assertEquals(3, output.shape[1]);
        assertEquals(2, layer.parameters().size());
    }

    @Test
    void testLinearGradcheck() {
        Linear layer = new Linear(3, 2);
        Tensor inputTensor = new Tensor(new double[][] {{1.0, 0.0, -1.0}}, true);
        Gradcheck.Result result = Gradcheck.checkTensorGradient(tensor -> layer.forward(tensor).sum(), inputTensor);

        assertTrue(result.passed, Double.toString(result.relativeError));
    }

    @Test
    void testActivationModules() {
        Tensor inputTensor = new Tensor(new double[][] {{-1.0, 0.5, 2.0}}, true);
        Gradcheck.Result passedRelu = Gradcheck.checkTensorGradient(tensor -> new ReLU().forward(tensor).power(2).sum(), inputTensor);

        assertTrue(passedRelu.passed, Double.toString(passedRelu.relativeError));

        Tensor logits = new Tensor(new double[][] {{1.0, 2.0, 3.0}, {0.1, -0.2, 0.3}}, true);
        Tensor probabilities = new Softmax(1).forward(logits);
        double row0 = probabilities.data[0] + probabilities.data[1] + probabilities.data[2];
        double row1 = probabilities.data[3] + probabilities.data[4] + probabilities.data[5];
        assertTrue(Math.abs(row0 - 1.0) < 1e-9);
        assertTrue(Math.abs(row1 - 1.0) < 1e-9);
    }

    @Test
    void testSequentialForward() {
        Sequential model = new Sequential(new Linear(2, 5), new ReLU(), new Linear(5, 1));
        Tensor output = model.forward(new Tensor(new double[][] {{0.1, -0.2}, {0.3, 0.4}}));
        assertEquals(2, output.shape[0]);
        assertEquals(1, output.shape[1]);
        assertEquals(4, model.parameters().size());
    }

    @Test
    void testSequentialGradcheck() {
        Sequential model = new Sequential(new Linear(2, 8), new ReLU(), new Linear(8, 1));
        Tensor inputTensor = new Tensor(new double[][] {{0.5, -1.0}}, true);
        Gradcheck.Result result = Gradcheck.checkTensorGradient(tensor -> model.forward(tensor).sum(), inputTensor);

        assertTrue(result.passed, Double.toString(result.relativeError));
    }

    @Test
    void testMseDecreasesWithSgd() {
        Sequential model = new Sequential(new Linear(1, 8), new ReLU(), new Linear(8, 1));
        double[][] inputs = new double[40][1];
        double[][] targets = new double[40][1];
        for (int index = 0; index < 40; index++) {
            inputs[index][0] = -1.0 + 2.0 * index / 39.0;
            targets[index][0] = 0.5 * inputs[index][0] + 0.1;
        }

        SGD optimizer = new SGD(model.parameters(), 0.05);
        MSELoss lossFunction = new MSELoss();
        double first = Double.NaN;
        double last = Double.NaN;
        for (int step = 0; step < 80; step++) {
            optimizer.zeroGrad();
            Tensor prediction = model.forward(new Tensor(inputs, false));
            Tensor loss = lossFunction.forward(prediction, targets);
            loss.backward();
            optimizer.step();
            if (step == 0) {
                first = loss.data[0];
            }

            last = loss.data[0];
        }

        assertTrue(last < first * 0.5);
    }

    @Test
    void testBceAndCeFinite() {
        Tensor logits = new Tensor(new double[][] {{0.2}, {-0.5}, {1.0}}, true);
        double[][] target = {{1.0}, {0.0}, {1.0}};
        Tensor loss = new BCEWithLogitsLoss().forward(logits, target);
        loss.backward();
        assertTrue(Double.isFinite(loss.data[0]));

        Tensor logits2 = new Tensor(new double[][] {{0.1, 0.2, 0.3}, {0.5, -0.1, 0.0}}, true);
        int[] targetIndex = {2, 0};
        Tensor loss2 = new CrossEntropyLoss().forward(logits2, targetIndex);
        loss2.backward();
        assertTrue(Double.isFinite(loss2.data[0]));
    }

    @Test
    void testLossGradcheck() {
        Tensor prediction = new Tensor(new double[][] {{0.5}, {-0.2}}, true);
        double[][] target = {{1.0}, {0.0}};
        Gradcheck.Result passedMse = Gradcheck.checkTensorGradient(tensor -> new MSELoss().forward(tensor, target), prediction);

        assertTrue(passedMse.passed, Double.toString(passedMse.relativeError));

        Tensor logits = new Tensor(new double[][] {{0.2}, {-0.5}, {1.0}}, true);
        Gradcheck.Result passedBce = Gradcheck.checkTensorGradient(tensor -> new BCEWithLogitsLoss().forward(tensor, new double[][] {{1.0}, {0.0}, {1.0}}), logits);

        assertTrue(passedBce.passed, Double.toString(passedBce.relativeError));
    }

    @Test
    void testAdamSteps() {
        Tensor parameter = new Tensor(new double[] {1.0, 2.0}, true);
        parameter.grad = new double[] {0.5, -0.25};
        double[] before = parameter.data.clone();
        new Adam(List.of(parameter), 0.1).step();
        assertTrue(Math.abs(parameter.data[0] - before[0]) > 1e-12 || Math.abs(parameter.data[1] - before[1]) > 1e-12);
    }
}
