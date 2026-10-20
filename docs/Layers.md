# **Layers**

Parameterize the affine map $y = xW + b$. A `Module` owns learnable `Tensor` parameters; `Linear` is the first layer that uses them. `Embedding` and `LayerNorm` are the extra building blocks a transformer needs. `Sequential` stacks modules for the tabular `train()` path.

> *`src/Module.java` · `src/Linear.java` · `src/Embedding.java` · `src/LayerNorm.java` · `src/Sequential.java`*

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

## **The `Sequential` Class**

Chains modules left to right. Overrides `parameters()` so nested `Linear` weights are visible to `SGD` / `Adam`.

```java
public class Sequential extends Module {
    public List<Module> layers = new ArrayList<>();

    public Sequential(Module... modules) {
        layers.addAll(List.of(modules));
    }

    @Override
    public Tensor forward(Tensor input) {
        Tensor output = input;
        for (Module layer : layers) {
            output = layer.forward(output);
        }

        return output;
    }

    @Override
    public List<Tensor> parameters() {
        List<Tensor> parameters = new ArrayList<>();
        for (Module layer : layers) {
            parameters.addAll(layer.parameters());
        }

        return parameters;
    }
}
```

Explanation:

1. `forward` is a left fold: each layer's output is the next layer's input.
2. `parameters()` walks `layers` so nested weights reach the optimizer.
3. Activations have no learnable tensors; they still participate in the graph.

---

## **Check**

```java
Sequential model = new Sequential(new Linear(2, 5), new ReLU(), new Linear(5, 1));
Tensor output = model.forward(new Tensor(new double[][] {{0.1, -0.2}, {0.3, 0.4}}));
```

`output.shape` is `{2, 1}` and `model.parameters().size()` is `4`. `tests/TestModules.java` also gradchecks this stack.
