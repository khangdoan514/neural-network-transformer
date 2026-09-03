# **Tensors**

Represent data as a differentiable array: values now, gradients later. A `Tensor` wraps a `double[]` plus a `shape` and records how it was created so later ops can attach reverse mode rules.

> *`src/Tensor.java`*

---

## **Math**

A tensor here is a multidimensional array $x \in \mathbb{R}^{n_1 \times \cdots \times n_d}$ stored in row major order.

Common transformer shapes:

- Token batch: $(B, T)$ integer ids (looked up, not a `Tensor` until embedding)
- Hidden states: $(B, T, C)$
- Attention scores: $(B, H, T, T)$
- Weights: $(C_{\mathrm{in}}, C_{\mathrm{out}})$

Elementwise ops broadcast like NumPy:

$$
c = a + b, \qquad \frac{\partial L}{\partial a} = \frac{\partial L}{\partial c}, \qquad \frac{\partial L}{\partial b} = \frac{\partial L}{\partial c}
$$

---

## **The `Tensor` Class**

A differentiable multidimensional array.

| Field | Role |
|-------|------|
| `data` | Row major payload (`double[]`) |
| `shape` | Dimensions of `data` |
| `requiresGrad` | If true, accumulate gradients into `.grad` during backward |
| `grad` | Accumulator for $\partial L / \partial(\text{this tensor})$ |
| `parents` | Upstream tensors that produced this node |
| `operation` | Debug name of the creating operation |
| `gradientFunction` | Local reverse mode rule for this node |

```java
public final class Tensor {
    public double[] data;
    public int[] shape;
    public boolean requiresGrad;
    public double[] grad;
    public Tensor[] parents;
    public String operation;
    public Runnable gradientFunction;

    public Tensor(double[] data, int[] shape, boolean requiresGrad, Tensor[] children, String operation) {
        this.data = Arrays.copyOf(data, data.length);
        this.shape = Arrays.copyOf(shape, shape.length);
        boolean childRequiresGrad = false;
        if (children != null) {
            for (Tensor child : children) {
                if (child.requiresGrad) {
                    childRequiresGrad = true;
                    break;
                }
            }
        }

        this.requiresGrad = requiresGrad || childRequiresGrad;
        this.grad = this.requiresGrad ? new double[this.data.length] : null;
        this.parents = children == null ? new Tensor[0] : children;
        this.operation = operation == null ? "" : operation;
        this.gradientFunction = () -> {};
    }
}
```

Explanation:

1. Payload and `shape` are copied so later in-place optimizer steps do not alias constructor arguments.
2. `requiresGrad` is sticky: if any parent needs a gradient, the output does too.
3. `gradientFunction` starts as a no-op; each op replaces it with a local reverse rule.

Also available:

- `detach()` — copy values without gradient tracking
- `shape()`, `ndim()`, `size()` — shape helpers
- `zeroGrad()` — zero the `.grad` buffer
- `representation()` — debug string

---

## **Construction Functions**

Factory methods for common initial tensors:

```java
public static Tensor zeros(int[] shape, boolean requiresGrad) {
    return new Tensor(new double[product(shape)], shape, requiresGrad, null, "");
}

public static Tensor ones(int[] shape, boolean requiresGrad) {
    double[] data = new double[product(shape)];
    Arrays.fill(data, 1.0);
    return new Tensor(data, shape, requiresGrad, null, "");
}

public static Tensor randn(int[] shape, boolean requiresGrad, Random randomGenerator) {
    Random generator = randomGenerator == null ? new Random() : randomGenerator;
    double[] data = new double[product(shape)];
    for (int index = 0; index < data.length; index++) {
        data[index] = generator.nextGaussian();
    }

    return new Tensor(data, shape, requiresGrad, null, "");
}
```

Vector and matrix constructors wrap the same constructor after flattening:

```java
public Tensor(double[][] matrix, boolean requiresGrad) {
    this(flatten(matrix), new int[] {matrix.length, matrix[0].length}, requiresGrad, null, "");
}

public Tensor(double[] vector, boolean requiresGrad) {
    this(vector, new int[] {vector.length}, requiresGrad, null, "");
}
```

---

## **Utility Functions**

### **`product()`**

Element count for a shape vector:

```java
public static int product(int[] shape) {
    int total = 1;
    for (int size : shape) {
        total *= size;
    }

    return total;
}
```

### **`flatten()`**

Convert a `double[][]` matrix into row major storage for the constructor:

```java
public static double[] flatten(double[][] matrix) {
    double[] flat = new double[matrix.length * matrix[0].length];
    int index = 0;
    for (double[] row : matrix) {
        for (double value : row) {
            flat[index++] = value;
        }
    }

    return flat;
}
```

### **`allClose()` and `norm()`**

Used later by gradient checks:

```java
public static boolean allClose(double[] left, double[] right, double absoluteTolerance, double relativeTolerance)
public static double norm(double[] values)
```

---

## **Arithmetic**

Elementwise ops broadcast like NumPy. `binary()` handles broadcasting and wires local `gradientFunction` closures on the output tensor.

```java
public Tensor add(Tensor other)
public Tensor subtract(Tensor other)
public Tensor multiply(Tensor other)
public Tensor divide(Tensor other)
public Tensor power(double exponent)
public Tensor negate()
```

Reverse mode for these ops is covered in [Autograd.md](Autograd.md).

---

## **Check**

```java
Tensor a = new Tensor(new double[] {2.0, 3.0}, true);
Tensor b = new Tensor(new double[] {4.0, 5.0}, true);
Tensor output = a.multiply(b).add(a);
output.backward(new double[] {1.0, 1.0});
```

`a.grad` should be `{5, 6}` and `b.grad` should be `{2, 3}`.

```java
Tensor left = new Tensor(new double[][] {{1.0, 2.0}, {3.0, 4.0}}, true);
Tensor right = new Tensor(new double[] {10.0, 20.0}, true);
Tensor broadcastOutput = left.add(right);
broadcastOutput.backward(new double[] {1.0, 1.0, 1.0, 1.0});
```

`left.grad` is all ones and `right.grad` is `{2, 2}`.
