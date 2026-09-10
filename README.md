# **Neural Network Transformer**

**Transformer** implemented from scratch with **Java** only. The project covers tensors, reverse mode autograd, backpropagation, linear layers, embeddings, layer norm, activations (ReLU, Tanh, Sigmoid, GELU, Softmax), multi head attention, transformer blocks, losses (MSE, BCE with logits, Cross Entropy), optimizers (SGD with momentum, Adam), minibatch training, finite difference gradient checks, and demos on copy, shift, and tiny text. No ND4J, DJL, PyTorch, TensorFlow, or JAX.

> *The repository is organized as an ordered, step by step curriculum. Each step includes math notes and maps to matching source files.*

---

## **Mathematical Overview**

### **1. Chain rule**

How a scalar loss $L$ pushes gradients backward through $y=g(x)$:

$$
\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} \frac{\partial y}{\partial x}
$$

### **2. Finite differences**

Numerical check against analytic `.grad`:

$$
\partial_i f(x) \approx \frac{f(x+\varepsilon e_i) - f(x-\varepsilon e_i)}{2\varepsilon}
$$

### **3. Affine map**

Forward pass of every `Linear` module:

$$
y = xW + b
$$

with $x \in \mathbb{R}^{N \times d_{\mathrm{in}}}$, $W \in \mathbb{R}^{d_{\mathrm{in}} \times d_{\mathrm{out}}}$, $b \in \mathbb{R}^{d_{\mathrm{out}}}$.

### **4. Scaled dot product attention**

$$
\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\left(\frac{QK^{\top}}{\sqrt{d_k}} + M\right) V
$$

Causal mask $M_{ij} = 0$ if $j \le i$, else $-\infty$, so token $i$ cannot attend to the future.

### **5. Multi head attention**

$$
\mathrm{MHA}(x) = \mathrm{Concat}(\mathrm{head}_1,\ldots,\mathrm{head}_h)W^O
$$

$$
\mathrm{head}_i = \mathrm{Attention}(xW_i^Q, xW_i^K, xW_i^V)
$$

### **6. Transformer block**

Pre-norm residual block:

$$
h = x + \mathrm{MHA}(\mathrm{LN}(x))
$$

$$
y = h + \mathrm{FFN}(\mathrm{LN}(h))
$$

$$
\mathrm{FFN}(z) = W_2\,\mathrm{GELU}(W_1 z + b_1) + b_2
$$

### **7. Activations**

Elementwise nonlinearities (and softmax over a class axis):

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
\mathrm{GELU}(x) \approx 0.5\,x\left(1 + \tanh\left[\sqrt{\frac{2}{\pi}}(x + 0.044715 x^3)\right]\right)
$$

$$
\mathrm{softmax}(z)_i = \frac{e^{z_i}}{\sum_j e^{z_j}}
$$

### **8. MSE**

Regression objective:

$$
L = \frac{1}{N}\sum_i (\hat{y}_i - y_i)^2
$$

### **9. Cross entropy**

Classification objective from logits:

$$
L = -\log p_c \quad \text{(class index } c\text{)}
$$

### **10. Adam**

Adaptive moments:

$$
\theta \leftarrow \theta - \eta \frac{\hat{m}}{\sqrt{\hat{v}}+\varepsilon}
$$

---

## **Step-by-step Roadmap**

| Step | Topic | Code | Guide |
|------|-------|------|-------|
| 01 | Tensors and ops | `src/Tensor.java` | [docs/Tensors.md](docs/Tensors.md) |
| 02 | Autograd / chain rule | `src/Tensor.java` (`backward`) | [docs/Autograd.md](docs/Autograd.md) |
| 03 | Gradient checking | `src/Gradcheck.java` | [docs/Gradcheck.md](docs/Gradcheck.md) |
| 04 | Modules, `Linear`, Embedding, LayerNorm | `src/Module.java` · `src/Linear.java` · `src/Embedding.java` · `src/LayerNorm.java` | [docs/Layers.md](docs/Layers.md) |
| 05 | Activations | `ReLU`, `Tanh`, `Sigmoid`, `GELU`, `Softmax` | [docs/Activations.md](docs/Activations.md) |
| 06 | Losses | `MSELoss`, `BCEWithLogitsLoss`, `CrossEntropyLoss` | [docs/Losses.md](docs/Losses.md) |
| 07 | Optimizers | `src/SGD.java`, `src/Adam.java` | [docs/Optimizers.md](docs/Optimizers.md) |
| 08 | Attention and Transformer | `src/MultiHeadAttention.java` · `src/Transformer.java` | [docs/Attention.md](docs/Attention.md), [docs/Transformer.md](docs/Transformer.md) |
| 09 | Training loop | `src/Training.java` | [docs/Training.md](docs/Training.md) |
| 10 | Experiments | `examples/` | [docs/Experiments.md](docs/Experiments.md) |

---

## **Getting Started**

```bash
git clone https://github.com/khangdoan514/neural-network-transformer
cd neural-network-transformer

mvn test
```

### **Run Demos**

```bash
mvn -q compile
java -cp target/classes examples.TrainCopy
java -cp target/classes examples.TrainShift
java -cp target/classes examples.TrainTinyText
```

### **Minimal Usage**

```java
import src.CrossEntropyLoss;
import src.Data;
import src.Tensor;
import src.Transformer;

int[][][] data = Data.makeCopy(4, 8, 8, 0);
Transformer model = new Transformer(8, 32, 4, 2, 8, null);
Tensor logits = model.forward(data[0]);
Tensor loss = new CrossEntropyLoss().forward(logits, data[1]);
loss.backward();
```
