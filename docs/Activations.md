# **Activations**

Elementwise nonlinearities and softmax over a chosen axis. Each module delegates to the matching `Tensor` operation.

> *`src/ReLU.java` · `src/Tanh.java` · `src/Sigmoid.java` · `src/GELU.java` · `src/Softmax.java`*

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

$$
\mathrm{GELU}(x) = x \cdot \Phi(x)
$$

$$
\mathrm{softmax}(z)_i = \frac{e^{z_i}}{\sum_j e^{z_j}}
$$

GELU uses the Gaussian CDF. Softmax subtracts a max for stability, then `exp` / `sum`.

---

## **Modules**

### **`ReLU`**

```java
public class ReLU extends Module {
    @Override
    public Tensor forward(Tensor input) {
        return input.relu();
    }
}
```

### **`Tanh`**

```java
public class Tanh extends Module {
    @Override
    public Tensor forward(Tensor input) {
        return input.tanh();
    }
}
```

### **`Sigmoid`**

```java
public class Sigmoid extends Module {
    @Override
    public Tensor forward(Tensor input) {
        return input.sigmoid();
    }
}
```

### **`GELU`**

```java
public class GELU extends Module {
    @Override
    public Tensor forward(Tensor input) {
        return input.gelu();
    }
}
```

### **`Softmax`**

```java
public class Softmax extends Module {
    public int axis;

    public Softmax() {
        this(-1);
    }

    public Softmax(int axis) {
        this.axis = axis;
    }

    @Override
    public Tensor forward(Tensor input) {
        return input.softmax(axis);
    }
}
```

---

## **Check**

```java
GELU gelu = new GELU();
Tensor y = gelu.forward(new Tensor(new double[] {-1.0, 0.0, 1.0}, false));
```

`y.data[1]` is `0`. Positive inputs stay positive. Negative inputs shrink toward zero.
