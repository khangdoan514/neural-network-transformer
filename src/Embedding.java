package src;

import java.util.Random;

public class Embedding extends Module {
    public Tensor weight;

    public Embedding(int vocabularySize, int channels, Random randomGenerator) {
        this.weight = Tensor.randn(new int[] {vocabularySize, channels}, true, randomGenerator);
        for (int index = 0; index < weight.data.length; index++) {
            weight.data[index] *= 0.02;
        }
    }

    public Tensor forward(int[][] tokens) {
        return weight.embeddingLookup(tokens);
    }
}
