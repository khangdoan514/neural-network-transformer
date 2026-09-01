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

---

## **The `Tensor` Class**

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

```java
public static Tensor zeros(int[] shape, boolean requiresGrad)
public static Tensor ones(int[] shape, boolean requiresGrad)
public static Tensor randn(int[] shape, boolean requiresGrad, Random randomGenerator)
```

Vector and matrix constructors wrap the same constructor after flattening:

```java
public Tensor(double[][] matrix, boolean requiresGrad)
public Tensor(double[] vector, boolean requiresGrad)
```

---

## **Utility Functions**

```java
public static int product(int[] shape)
public static double[] flatten(double[][] matrix)
public static boolean allClose(double[] left, double[] right, double absoluteTolerance, double relativeTolerance)
public static double norm(double[] values)
```

---

## **Check**

```java
Tensor matrix = new Tensor(new double[][] {{1.0, 2.0}, {3.0, 4.0}}, false);
Tensor filled = Tensor.ones(new int[] {2, 2}, false);
```

`matrix.shape` is `{2, 2}` and `filled.size()` is `4`.
