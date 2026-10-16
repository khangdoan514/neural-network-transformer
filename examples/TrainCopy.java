package examples;

import src.Adam;
import src.CrossEntropyLoss;
import src.Data;
import src.Training;
import src.Transformer;

import java.util.Random;

public class TrainCopy {
    public static void main(String[] args) {
        int[][][] data = Data.makeCopy(64, 8, 8, 0);
        Transformer model = new Transformer(8, 32, 4, 2, 8, new Random(0));
        Training.TrainingHistory history = Training.trainLanguage(
                model,
                new CrossEntropyLoss(),
                new Adam(model.parameters(), 0.03),
                data[0],
                data[1],
                80,
                16,
                true,
                10,
                new Random(0)
        );
        System.out.printf("final accuracy: %.3f%n", history.metric.get(history.metric.size() - 1));
    }
}
