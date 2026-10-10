# **Transformer**

Stack a pre-norm attention residual with a positionwise feedforward residual. `FeedForward` is the channel mixer; `TransformerBlock` is one full decoder block.

> *`src/FeedForward.java` · `src/TransformerBlock.java`*

---

## **Math**

$$
h = x + \mathrm{MultiHeadAttention}(\mathrm{LayerNorm}(x))
$$

$$
y = h + \mathrm{FeedForward}(\mathrm{LayerNorm}(h))
$$

$$
\mathrm{FeedForward}(z) = W_2\,\mathrm{GELU}(W_1 z + b_1) + b_2
$$

The feedforward hidden width is $4C$. Tokens stay independent inside the feedforward network; mixing across positions is left to attention.

---

## **The `FeedForward` Class**

```java
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
```

---

## **The `TransformerBlock` Class**

```java
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
```

Explanation:

1. Pre-norm: `LayerNorm` before attention and before the feedforward network.
2. Residuals add the block input (or post-attention state) back unchanged.
3. The feedforward network expands to $4C$, applies GELU, then projects back to $C$.

---

## **Check**

```java
TransformerBlock block = new TransformerBlock(8, 4, new Random(0));
Tensor hidden = Tensor.ones(new int[] {2, 4, 8}, false);
Tensor output = block.forward(hidden);
```

`output.shape` is `{2, 4, 8}`.
