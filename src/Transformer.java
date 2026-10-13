package src;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class Transformer extends Module {
    public Embedding tokenEmbedding;
    public Embedding positionEmbedding;
    public List<TransformerBlock> blocks;
    public LayerNorm finalNorm;
    public Linear languageHead;
    public int contextLength;

    public Transformer(int vocabularySize, int channels, int headCount, int layerCount, int contextLength, Random randomGenerator) {
        this.contextLength = contextLength;
        this.tokenEmbedding = new Embedding(vocabularySize, channels, randomGenerator);
        this.positionEmbedding = new Embedding(contextLength, channels, randomGenerator);
        this.blocks = new ArrayList<>();
        for (int index = 0; index < layerCount; index++) {
            blocks.add(new TransformerBlock(channels, headCount, randomGenerator));
        }

        this.finalNorm = new LayerNorm(channels);
        this.languageHead = new Linear(channels, vocabularySize, true, randomGenerator);
    }

    public Tensor forward(int[][] tokens) {
        int batch = tokens.length;
        int sequence = tokens[0].length;
        int[][] positions = new int[batch][sequence];
        for (int batchIndex = 0; batchIndex < batch; batchIndex++) {
            for (int time = 0; time < sequence; time++) {
                positions[batchIndex][time] = time;
            }
        }

        Tensor hidden = tokenEmbedding.forward(tokens).add(positionEmbedding.forward(positions));
        for (TransformerBlock block : blocks) {
            hidden = block.forward(hidden);
        }

        Tensor normalized = finalNorm.forward(hidden);
        int channels = normalized.shape[2];
        int vocabularySize = languageHead.weight.shape[1];

        return languageHead.forward(normalized.reshape(batch * sequence, channels)).reshape(batch, sequence, vocabularySize);
    }
}
