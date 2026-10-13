# **Transformer**

Stack embeddings, pre-norm blocks, and a vocabulary head into a decoder only language model.

> *`src/FeedForward.java` · `src/TransformerBlock.java` · `src/Transformer.java`*

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

Token and position embeddings:

$$
x_t = E_{\mathrm{tok}}[w_t] + E_{\mathrm{pos}}[t]
$$

The language head is a `Linear` from $C$ to $V$. The feedforward hidden width is $4C$.

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

## **The `Transformer` Class**

```java
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
```

---

## **Check**

```java
Transformer model = new Transformer(8, 32, 4, 2, 8, new Random(0));
int[][] tokens = new int[][] {{0, 1, 2, 3, 4, 5, 6, 7}, {1, 2, 3, 4, 5, 6, 7, 0}};
Tensor logits = model.forward(tokens);
```

`logits.shape` is `{2, 8, 8}`.
