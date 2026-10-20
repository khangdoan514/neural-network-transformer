package tests;

import org.junit.jupiter.api.Test;
import src.Adam;
import src.BCEWithLogitsLoss;
import src.Data;
import src.Linear;
import src.ReLU;
import src.Sequential;
import src.Training;
import src.Transformer;
import src.CrossEntropyLoss;
import java.util.Random;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class TestLearning {
    @Test
    void testXorLearns() {
        double[][][] data = Data.makeXor(40, 0.05, 0);
        Sequential model = new Sequential(
                new Linear(2, 16),
                new ReLU(),
                new Linear(16, 16),
                new ReLU(),
                new Linear(16, 1)
        );
        Training.TrainingHistory history = Training.train(
                model,
                new BCEWithLogitsLoss(),
                new Adam(model.parameters(), 0.05),
                data[0],
                data[1],
                120,
                32,
                Training::accuracyBinary,
                new Random(0),
                false,
                10
        );
        assertTrue(history.metric.get(history.metric.size() - 1) >= 0.90);
        assertTrue(history.loss.get(history.loss.size() - 1) < history.loss.get(0));
    }

    @Test
    void testCopyLearns() {
        int[][][] data = Data.makeCopy(48, 4, 4, 0);
        Transformer model = new Transformer(4, 16, 4, 1, 4, new Random(0));
        Training.TrainingHistory history = Training.trainLanguage(
                model,
                new CrossEntropyLoss(),
                new Adam(model.parameters(), 0.05),
                data[0],
                data[1],
                40,
                16,
                false,
                10,
                new Random(0)
        );
        assertTrue(history.metric.get(history.metric.size() - 1) >= 0.70);
        assertTrue(history.loss.get(history.loss.size() - 1) < history.loss.get(0));
    }
}
