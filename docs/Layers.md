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

## **Check**

```java
Linear layer = new Linear(4, 1);
Tensor logits = layer.forward(Tensor.ones(new int[] {2, 4}, true));
Tensor bce = new BCEWithLogitsLoss().forward(logits, new double[][] {{1.0}, {0.0}});
bce.backward();
```

`logits.shape` is `{2, 1}` and `bce` is a scalar. `layer.parameters()` carry gradients after `backward()`.
