package src;

import java.util.Arrays;
import java.util.Random;

public class MultiHeadAttention extends Module {
    public Linear queryProjection;
    public Linear keyProjection;
    public Linear valueProjection;
    public Linear outputProjection;
    public int headCount;
    public int headDimension;
    public int channels;

    public MultiHeadAttention(int channels, int headCount, Random randomGenerator) {
        if (channels % headCount != 0) {
            throw new IllegalArgumentException("channels must divide headCount");
        }

        this.channels = channels;
        this.headCount = headCount;
        this.headDimension = channels / headCount;
        this.queryProjection = new Linear(channels, channels, false, randomGenerator);
        this.keyProjection = new Linear(channels, channels, false, randomGenerator);
        this.valueProjection = new Linear(channels, channels, false, randomGenerator);
        this.outputProjection = new Linear(channels, channels, false, randomGenerator);
    }

    @Override
    public Tensor forward(Tensor input) {
        int batch = input.shape[0];
        int sequence = input.shape[1];
        Tensor flat = input.reshape(batch * sequence, channels);
        Tensor query = splitHeads(queryProjection.forward(flat), batch, sequence);
        Tensor key = splitHeads(keyProjection.forward(flat), batch, sequence);
        Tensor value = splitHeads(valueProjection.forward(flat), batch, sequence);
        Tensor scores = query.matmul(key.permute(0, 1, 3, 2)).multiply(1.0 / Math.sqrt(headDimension));
        scores = applyCausalMask(scores);
        Tensor weights = scores.softmax(-1);
        Tensor merged = mergeHeads(weights.matmul(value), batch, sequence);

        return outputProjection.forward(merged);
    }

    private Tensor splitHeads(Tensor projected, int batch, int sequence) {
        return projected.reshape(batch, sequence, headCount, headDimension).permute(0, 2, 1, 3);
    }

    private Tensor mergeHeads(Tensor headed, int batch, int sequence) {
        return headed.permute(0, 2, 1, 3).reshape(batch * sequence, channels);
    }

    private static Tensor applyCausalMask(Tensor scores) {
        int batchSize = scores.shape[0];
        int heads = scores.shape[1];
        int sequence = scores.shape[2];
        double[] masked = Arrays.copyOf(scores.data, scores.data.length);
        for (int batchIndex = 0; batchIndex < batchSize; batchIndex++) {
            for (int head = 0; head < heads; head++) {
                for (int query = 0; query < sequence; query++) {
                    for (int key = 0; key < sequence; key++) {
                        if (key > query) {
                            int index = ((batchIndex * heads + head) * sequence + query) * sequence + key;
                            masked[index] = -1e9;
                        }
                    }
                }
            }
        }

        Tensor out = new Tensor(masked, scores.shape, scores.requiresGrad, new Tensor[] {scores}, "mask");
        out.gradientFunction = () -> {
            if (scores.requiresGrad) {
                for (int index = 0; index < scores.grad.length; index++) {
                    scores.grad[index] += out.grad[index];
                }
            }
        };

        return out;
    }
}
