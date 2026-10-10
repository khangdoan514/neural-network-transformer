package src;

import java.util.Random;

public class FeedForward extends Module {
    public Linear expand;
    public Linear project;
    public GELU activation;

    public FeedForward(int channels, Random randomGenerator) {
        this.expand = new Linear(channels, 4 * channels, true, randomGenerator);
        this.project = new Linear(4 * channels, channels, true, randomGenerator);
        this.activation = new GELU();
    }

    @Override
    public Tensor forward(Tensor input) {
        int batch = input.shape[0];
        int sequence = input.shape[1];
        int channels = input.shape[2];
        Tensor flat = input.reshape(batch * sequence, channels);
        Tensor hidden = project.forward(activation.forward(expand.forward(flat)));

        return hidden.reshape(batch, sequence, channels);
    }
}
