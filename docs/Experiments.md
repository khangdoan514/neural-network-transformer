# **Experiments**

Toy datasets and demos that prove the transformer stack works end to end. Data helpers live in `src/Data.java`; runnable scripts live under `examples/`.

> *`src/Data.java` · `examples/TrainCopy.java` · `examples/TrainShift.java` · `examples/TrainTinyText.java`*

---

## **Datasets**

### **`makeCopy()`**

Random token sequences. Targets equal the inputs, so causal attention can copy the current token.

```java
public static int[][][] makeCopy(int sampleCount, int sequenceLength, int vocabularySize, int seed) {
    Random randomGenerator = new Random(seed);
    int[][] inputs = new int[sampleCount][sequenceLength];
    int[][] targets = new int[sampleCount][sequenceLength];
    for (int sample = 0; sample < sampleCount; sample++) {
        for (int time = 0; time < sequenceLength; time++) {
            int token = randomGenerator.nextInt(vocabularySize);
            inputs[sample][time] = token;
            targets[sample][time] = token;
        }
    }

    return new int[][][] {inputs, targets};
}
```

### **`makeShift()`**

Next token prediction on random ids: target at time $t$ is input at $t+1$.

```java
public static int[][][] makeShift(int sampleCount, int sequenceLength, int vocabularySize, int seed) {
    Random randomGenerator = new Random(seed);
    int[][] inputs = new int[sampleCount][sequenceLength];
    int[][] targets = new int[sampleCount][sequenceLength];
    for (int sample = 0; sample < sampleCount; sample++) {
        for (int time = 0; time < sequenceLength; time++) {
            inputs[sample][time] = randomGenerator.nextInt(vocabularySize);
        }

        for (int time = 0; time < sequenceLength; time++) {
            targets[sample][time] = time + 1 < sequenceLength ? inputs[sample][time + 1] : inputs[sample][time];
        }
    }

    return new int[][][] {inputs, targets};
}
```

### **`makeTinyText()`**

Character level windows over a short English string (`a` to `z` plus space). Targets are the next character. Vocabulary size is $27$.

```java
public static int[][][] makeTinyText(int sampleCount, int sequenceLength, int seed) {
    String text = "the cat sat on the mat the dog sat on the log the cat sat";
    int vocabularySize = 27;
    Random randomGenerator = new Random(seed);
    int[] encoded = new int[text.length()];
    for (int index = 0; index < text.length(); index++) {
        char character = text.charAt(index);
        encoded[index] = character == ' ' ? 26 : character - 'a';
    }

    int[][] inputs = new int[sampleCount][sequenceLength];
    int[][] targets = new int[sampleCount][sequenceLength];
    for (int sample = 0; sample < sampleCount; sample++) {
        int start = randomGenerator.nextInt(encoded.length - sequenceLength - 1);
        for (int time = 0; time < sequenceLength; time++) {
            inputs[sample][time] = encoded[start + time];
            targets[sample][time] = encoded[start + time + 1];
        }
    }

    return new int[][][] {inputs, targets, new int[][] {{vocabularySize}}};
}
```

### **`makeXor()`**

Dense XOR pairs for the tabular `train()` path and learning tests.

```java
public static double[][][] makeXor(int samplesPerClass, double noise, int seed) {
    Random randomGenerator = new Random(seed);
    double[][] centers = {{0.0, 0.0}, {0.0, 1.0}, {1.0, 0.0}, {1.0, 1.0}};
    double[] centerLabels = {0.0, 1.0, 1.0, 0.0};
    int total = samplesPerClass * 4;
    double[][] inputs = new double[total][2];
    double[][] targets = new double[total][1];
    int write = 0;
    for (int centerIndex = 0; centerIndex < 4; centerIndex++) {
        for (int sample = 0; sample < samplesPerClass; sample++) {
            inputs[write][0] = centers[centerIndex][0] + noise * randomGenerator.nextGaussian();
            inputs[write][1] = centers[centerIndex][1] + noise * randomGenerator.nextGaussian();
            targets[write][0] = centerLabels[centerIndex];
            write += 1;
        }
    }

    return shuffleDense(inputs, targets, randomGenerator);
}
```

---

## **Demos**

| Script | Task | Loss | Target |
|--------|------|------|--------|
| `examples/TrainCopy.java` | copy tokens | `CrossEntropyLoss` | accuracy ≥ 0.70 |
| `examples/TrainShift.java` | next token (random) | `CrossEntropyLoss` | loss decreases |
| `examples/TrainTinyText.java` | tiny character language model | `CrossEntropyLoss` | loss decreases |

```bash
mvn -q compile
java -cp target/classes examples.TrainCopy
java -cp target/classes examples.TrainShift
java -cp target/classes examples.TrainTinyText
```

```java
int[][][] data = Data.makeTinyText(64, 12, 0);
Transformer model = new Transformer(27, 32, 4, 2, 12, new Random(0));
Training.TrainingHistory history = Training.trainLanguage(
    model,
    new CrossEntropyLoss(),
    new Adam(model.parameters(), 0.03),
    data[0],
    data[1],
    80,
    16,
    true,
    10,
    new Random(0)
);
```

---

## **Check**

```java
int[][][] data = Data.makeCopy(48, 4, 4, 0);
Transformer model = new Transformer(4, 16, 4, 1, 4, new Random(0));
Training.TrainingHistory history = Training.trainLanguage(
    model,
    new CrossEntropyLoss(),
    new Adam(model.parameters(), 0.05),
    data[0],
    data[1],
    40,
    16,
    false,
    10,
    new Random(0)
);
```

`tests/TestLearning.java` checks that copy accuracy reaches at least $0.70$ and that loss falls below the first epoch.
