# **Activations**

Elementwise nonlinearities. Each module delegates to the matching `Tensor` op.

> *`src/ReLU.java`*

---

## **Math**

$$
\mathrm{ReLU}(x) = \max(x, 0)
$$

The subgradient at zero is taken to be zero.

---

## **The `ReLU` Class**

```java
public class ReLU extends Module {
    @Override
    public Tensor forward(Tensor input) {
        return input.relu();
    }
}
```

---

## **Check**

```java
ReLU relu = new ReLU();
Tensor y = relu.forward(new Tensor(new double[] {-1, 0, 2}, false));
```

`y` is `{0, 0, 2}`.
