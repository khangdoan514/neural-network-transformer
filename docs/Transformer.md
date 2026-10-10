# **Transformer**

The positionwise feedforward network expands each token, applies GELU, then projects back.

> *`src/FeedForward.java`*

---

## **Math**

$$
\mathrm{FFN}(z) = W_2\,\mathrm{GELU}(W_1 z + b_1) + b_2
$$

The hidden width is $4C$. Tokens stay independent inside the FFN; mixing across positions is left to attention.

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

## **Check**

```java
FeedForward feedForward = new FeedForward(8, new Random(0));
Tensor hidden = Tensor.ones(new int[] {2, 4, 8}, false);
Tensor output = feedForward.forward(hidden);
```

`output.shape` is `{2, 4, 8}`.
