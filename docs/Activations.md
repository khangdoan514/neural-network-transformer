# **Activations**

Elementwise nonlinearities. Each module delegates to the matching `Tensor` op.

> *`src/ReLU.java` · `src/Tanh.java` · `src/Sigmoid.java`*

---

## **Math**

$$
\mathrm{ReLU}(x) = \max(x, 0)
$$

$$
\sigma(x) = \frac{1}{1 + e^{-x}}
$$

$$
\tanh x = \frac{e^{x} - e^{-x}}{e^{x} + e^{-x}}
$$

The ReLU subgradient at zero is taken to be zero. Sigmoid squashes to $(0, 1)$. Tanh squashes to $(-1, 1)$.

---

## **Modules**

```java
public class ReLU extends Module {
    @Override
    public Tensor forward(Tensor input) {
        return input.relu();
    }
}
```

`Tanh` and `Sigmoid` are the same forward method onto `input.tanh()` and `input.sigmoid()`.

---

## **Check**

```java
Tanh tanh = new Tanh();
Sigmoid sigmoid = new Sigmoid();
Tensor centered = tanh.forward(new Tensor(new double[] {-2, 0, 2}, false));
Tensor probability = sigmoid.forward(new Tensor(new double[] {0}, false));
```

`centered` is odd around zero. `probability.data[0]` is `0.5`.
