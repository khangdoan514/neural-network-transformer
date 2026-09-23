package src;

public class GELU extends Module {
    @Override
    public Tensor forward(Tensor input) {
        return input.gelu();
    }
}
