package src;

import java.util.Random;

public class TransformerBlock extends Module {
    public LayerNorm norm1;
    public LayerNorm norm2;
    public MultiHeadAttention attention;
    public FeedForward feedForward;

    public TransformerBlock(int channels, int headCount, Random randomGenerator) {
        this.norm1 = new LayerNorm(channels);
        this.norm2 = new LayerNorm(channels);
        this.attention = new MultiHeadAttention(channels, headCount, randomGenerator);
        this.feedForward = new FeedForward(channels, randomGenerator);
    }

    @Override
    public Tensor forward(Tensor input) {
        int batch = input.shape[0];
        int sequence = input.shape[1];
        int channels = input.shape[2];
        Tensor normalized = norm1.forward(input);
        Tensor attended = attention.forward(normalized).reshape(batch, sequence, channels);
        Tensor residual = input.add(attended);
        Tensor fed = feedForward.forward(norm2.forward(residual));

        return residual.add(fed);
    }
}
