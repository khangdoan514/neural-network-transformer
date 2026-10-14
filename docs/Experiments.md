# **Experiments**

Toy datasets that prove the transformer stack works end to end. Data helpers live in `src/Data.java`.

> *`src/Data.java`*

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

Final metric should rise and final loss should fall below the first epoch.
