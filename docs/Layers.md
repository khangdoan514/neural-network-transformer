# **Layers**

Parameterize the affine map $y = xW + b$. A `Module` owns learnable `Tensor` parameters; `Linear` is the first layer that uses them. `Embedding` and `LayerNorm` are the extra building blocks a transformer needs.

> *`src/Module.java` · `src/Linear.java` · `src/Embedding.java` · `src/LayerNorm.java`*

---

## **Math**

Affine map for a batch of inputs:

$$
y = xW + b
$$

He-style initialization:

$$
W_{ij} \sim \mathcal{N}\left(0, \sqrt{\frac{2}{d_{\mathrm{in}}}}\right)
$$

Embedding lookup: row $t$ of $E \in \mathbb{R}^{V \times C}$ for token id $t$.

Layer norm over the last axis:

$$
\hat{x} = \frac{x - \mu}{\sqrt{\sigma^2 + \varepsilon}}, \qquad y = \gamma \odot \hat{x} + \beta
$$

---

## **The `Module` Class**

| Method | Role |
|-------|------|
| `parameters()` | Collect every `Tensor` with `requiresGrad` on this module |
| `zeroGrad()` | Zero `.grad` on all parameters |
| `trainMode()` / `evalMode()` | Toggle training flag |
| `forward()` | Define the forward pass |
| `apply()` | Calls `forward()` |

### **`parameters()`**

```java
public List<Tensor> parameters() {
    List<Tensor> parameters = new ArrayList<>();
    collect(this, parameters);
    return parameters;
}
```

Explanation:

1. Reflection walks fields the way Python walks `__dict__`.
2. Nested `Module` values and `List`s of modules are collected recursively.

---

## **The `Linear` Class**

```java
public class Linear extends Module {
    public Tensor weight;
    public Tensor bias;

    public Linear(int inFeatures, int outFeatures, boolean bias, Random randomGenerator) {
        this.weight = kaiming(inFeatures, outFeatures, randomGenerator);
        this.bias = bias ? Tensor.zeros(new int[] {outFeatures}, true) : null;
    }

    @Override
    public Tensor forward(Tensor input) {
        Tensor output = input.matmul(weight);
        if (bias != null) {
            output = output.add(bias);
        }

        return output;
    }
}
```

---

## **The `Embedding` Class**

```java
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
```

---

## **The `LayerNorm` Class**

```java
@Override
public Tensor forward(Tensor input) {
    Tensor mean = input.mean(-1, true);
    Tensor centered = input.subtract(mean);
    Tensor variance = centered.power(2).mean(-1, true);
    Tensor normalized = centered.divide(variance.add(epsilon).power(0.5));

    return normalized.multiply(gain).add(bias);
}
```

---

## **Check**

```java
int[][][] data = Data.makeShift(16, 4, 4, 0);
Transformer model = new Transformer(4, 16, 4, 1, 4, new Random(0));
Training.TrainingHistory history = Training.trainLanguage(
    model,
    new CrossEntropyLoss(),
    new Adam(model.parameters(), 0.05),
    data[0],
    data[1],
    5,
    4,
    false,
    1,
    new Random(0)
);
```

`history.loss` and `history.metric` each have one entry per epoch.
