# **Training**

Minibatch loop that ties the model, loss, and optimizer together: shuffle, forward, backward, step, repeat.

> *`src/Training.java`*

---

## **One Step**

For each minibatch:

1. `optimizer.zeroGrad()`
2. Forward: `prediction = model.forward(batch)`
3. `loss = lossFunction.forward(prediction, targets)`
4. `loss.backward()`
5. `optimizer.step()`

An epoch is one full pass over the shuffled dataset. With `batchSize = null`, the whole set is one batch (full-batch gradient descent).

---

## **Metrics**

### **`accuracyTokens()`**

```java
public static double accuracyTokens(Tensor logits, int[][] target) {
    int batch = logits.shape[0];
    int sequence = logits.shape[1];
    int vocabularySize = logits.shape[2];
    int count = 0;
    int total = batch * sequence;
    for (int batchIndex = 0; batchIndex < batch; batchIndex++) {
        for (int time = 0; time < sequence; time++) {
            int offset = (batchIndex * sequence + time) * vocabularySize;
            int best = 0;
            double bestValue = logits.data[offset];
            for (int token = 1; token < vocabularySize; token++) {
                double value = logits.data[offset + token];
                if (value > bestValue) {
                    bestValue = value;
                    best = token;
                }
            }

            if (best == target[batchIndex][time]) {
                count += 1;
            }
        }
    }

    return (double) count / total;
}
```

Token accuracy takes the index of the maximum over the vocabulary axis at each position.

---

## **`trainLanguage()`**

Token batches for the decoder only transformer. `inputs` and `targets` are `int[][]` with shape $(N, T)$.

```java
public static TrainingHistory trainLanguage(Transformer model, CrossEntropyLoss lossFunction, Optimizer optimizer, int[][] inputs, int[][] targets, int epochs, Integer batchSize, boolean verbose, int logEvery, Random randomGenerator) {
    Random generator = randomGenerator == null ? new Random(0) : randomGenerator;
    int sampleCount = inputs.length;
    int resolvedBatchSize = batchSize == null ? sampleCount : batchSize;
    TrainingHistory history = new TrainingHistory();
    for (int epoch = 1; epoch <= epochs; epoch++) {
        int[] permutation = permutation(sampleCount, generator);
        int[][] shuffledInputs = gatherTokenRows(inputs, permutation);
        int[][] shuffledTargets = gatherTokenRows(targets, permutation);
        List<Double> batchLosses = new ArrayList<>();
        List<Double> batchMetrics = new ArrayList<>();
        for (int start = 0; start < sampleCount; start += resolvedBatchSize) {
            int end = Math.min(start + resolvedBatchSize, sampleCount);
            int[][] batchInputs = sliceTokenRows(shuffledInputs, start, end);
            int[][] batchTargets = sliceTokenRows(shuffledTargets, start, end);
            optimizer.zeroGrad();
            Tensor prediction = model.forward(batchInputs);
            Tensor loss = lossFunction.forward(prediction, batchTargets);
            loss.backward();
            optimizer.step();
            batchLosses.add(loss.data[0]);
            batchMetrics.add(accuracyTokens(prediction.detach(), batchTargets));
        }

        double epochLoss = mean(batchLosses);
        double epochMetric = mean(batchMetrics);
        history.loss.add(epochLoss);
        history.metric.add(epochMetric);
        if (verbose && (epoch % logEvery == 0 || epoch == 1 || epoch == epochs)) {
            System.out.printf("epoch %4d | loss %.6f | metric %.4f%n", epoch, epochLoss, epochMetric);
        }
    }

    return history;
}
```

Explanation:

1. Shuffle indices once per epoch.
2. Slice contiguous token minibatches after the shuffle.
3. `accuracyTokens` uses `prediction.detach()` so evaluation does not grow the graph.
4. `TrainingHistory` stores per-epoch mean loss and token accuracy.

---

## **Check**

```java
int[][][] data = Data.makeShift(64, 8, 8, 0);
Transformer model = new Transformer(8, 32, 4, 2, 8, new Random(0));
Training.TrainingHistory history = Training.trainLanguage(
    model,
    new CrossEntropyLoss(),
    new Adam(model.parameters(), 0.03),
    data[0],
    data[1],
    80,
    16,
    false,
    10,
    new Random(0)
);
```

Final loss should fall below the first epoch on the shift task.
