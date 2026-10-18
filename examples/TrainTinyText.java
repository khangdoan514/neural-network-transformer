package examples;

import src.Adam;
import src.CrossEntropyLoss;
import src.Data;
import src.Training;
import src.Transformer;

import java.util.Random;

public class TrainTinyText {
    public static void main(String[] args) {
        int[][][] data = Data.makeTinyText(64, 12, 0);
        Transformer model = new Transformer(27, 32, 4, 2, 12, new Random(0));
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
