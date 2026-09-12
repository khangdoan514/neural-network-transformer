package src;

public class LayerNorm extends Module {
    public Tensor gain;
    public Tensor bias;
    public double epsilon;

    public LayerNorm(int channels, double epsilon) {
        this.gain = Tensor.ones(new int[] {channels}, true);
        this.bias = Tensor.zeros(new int[] {channels}, true);
        this.epsilon = epsilon;
    }

    public LayerNorm(int channels) {
        this(channels, 1e-5);
    }

    @Override
    public Tensor forward(Tensor input) {
        Tensor mean = input.mean(-1, true);
        Tensor centered = input.subtract(mean);
        Tensor variance = centered.power(2).mean(-1, true);
        Tensor normalized = centered.divide(variance.add(epsilon).power(0.5));

        return normalized.multiply(gain).add(bias);
    }
}
