# **Losses**

Scalar objectives $L(\hat y, y)$ that turn network outputs into something you can call `.backward()` on. Each loss is a `Module` whose `forward` returns a scalar `Tensor`.

> *`src/MSELoss.java` · `src/BCEWithLogitsLoss.java`*

---

## **Math**

**MSE** (regression)

$$
L = \frac{1}{N}\sum_i (\hat{y}_i - y_i)^2
$$

**BCE with logits** (binary classification, stable form)

For logits $z$ and targets $y \in \{0,1\}$:

$$
L = \frac{1}{N}\sum_i \Big(\max(z_i,0) - z_i y_i + \log(1 + e^{-|z_i|})\Big)
$$

---

## **`MSELoss`**

Mean squared error: `mean((prediction - target)^2)`.

```java
public class MSELoss extends Module {
    public Tensor forward(Tensor prediction, Tensor target) {
        return prediction.subtract(target).power(2).mean();
    }

    public Tensor forward(Tensor prediction, double[][] target) {
        return forward(prediction, new Tensor(target, false));
    }
}
```

---

## **`BCEWithLogitsLoss`**

Binary cross entropy on raw logits. Uses the stable rewrite above so large positive or negative logits do not overflow.

```java
public class BCEWithLogitsLoss extends Module {
    public Tensor forward(Tensor logits, Tensor target) {
        Tensor absoluteLogits = logits.relu().add(logits.negate().relu());
        Tensor loss = logits.relu().subtract(logits.multiply(target)).add(absoluteLogits.negate().exp().add(1.0).log());

        return loss.mean();
    }

    public Tensor forward(Tensor logits, double[][] target) {
        return forward(logits, new Tensor(target, false));
    }
}
```

Explanation:

1. `|z|` is built as `relu(z) + relu(-z)`.
2. Targets should be floats in $\{0,1\}$ with the same shape as `logits`.
3. Prefer this over `Sigmoid` + BCE when training binary classifiers.

---

## **Check**

```java
Tensor prediction = new Tensor(new double[][] {{0.5}, {-0.2}}, true);
Tensor mse = new MSELoss().forward(prediction, new double[][] {{1.0}, {0.0}});

Tensor logits = new Tensor(new double[][] {{0.2}, {-0.5}, {1.0}}, true);
Gradcheck.Result bceResult = Gradcheck.checkTensorGradient(tensor -> new BCEWithLogitsLoss().forward(tensor, new double[][] {{1.0}, {0.0}, {1.0}}), logits);
```

`mse` is a scalar. `bceResult.passed` should be true.
