# **Attention**

Causal multi head self attention is the mixing step that lets every token look at earlier tokens in the same sequence.

> *`src/MultiHeadAttention.java`*

---

## **Math**

$$
\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\left(\frac{QK^{\top}}{\sqrt{d_k}} + M\right) V
$$

$Q, K, V$ come from linear maps of the same input $x$. Heads split the channel axis, attend independently, then concatenate:

$$
\mathrm{MultiHeadAttention}(x) = \mathrm{Concat}(\mathrm{head}_1,\ldots,\mathrm{head}_h)W^O
$$

Causal mask: $M_{ij} = 0$ if $j \le i$, else a large negative number so softmax mass on the future is $\approx 0$.

Shapes inside one call:

- Input: $(B, T, C)$
- After split: $(B, H, T, d_k)$ with $C = H \cdot d_k$
- Scores: $(B, H, T, T)$
- Output: $(B T, C)$ (flattened for the output projection)

---

## **The `MultiHeadAttention` Class**

```java
@Override
public Tensor forward(Tensor input) {
    int batch = input.shape[0];
    int sequence = input.shape[1];
    Tensor flat = input.reshape(batch * sequence, channels);
    Tensor query = splitHeads(queryProjection.forward(flat), batch, sequence);
    Tensor key = splitHeads(keyProjection.forward(flat), batch, sequence);
    Tensor value = splitHeads(valueProjection.forward(flat), batch, sequence);
    Tensor scores = query.matmul(key.permute(0, 1, 3, 2)).multiply(1.0 / Math.sqrt(headDimension));
    scores = applyCausalMask(scores);
    Tensor weights = scores.softmax(-1);
    Tensor merged = mergeHeads(weights.matmul(value), batch, sequence);

    return outputProjection.forward(merged);
}
```

Explanation:

1. Flatten $(B, T, C)$ to $(BT, C)$ so `Linear` stays a matrix multiply.
2. `splitHeads` permutes to $(B, H, T, d_k)$.
3. Scale by $1/\sqrt{d_k}$, mask the upper triangle, softmax over keys.
4. Merge heads and project back to $C$.

---

## **Check**

```java
MultiHeadAttention attention = new MultiHeadAttention(8, 4, new Random(0));
Tensor hidden = Tensor.ones(new int[] {2, 4, 8}, false);
Tensor mixed = attention.forward(hidden);
```

`mixed.shape` is `{8, 8}`, reshapeable to `{2, 4, 8}`.
