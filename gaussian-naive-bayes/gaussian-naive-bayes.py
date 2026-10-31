import math

def gaussian_naive_bayes(X_train, y_train, X_test):
    """
    Returns a predicted class label for every test sample.
    """
    classes = set(y_train)
    n, d = len(y_train), len(X_train[0])

    stats = {}
    for c in classes:
        data = [X_train[i] for i in range(n) if y_train[i] == c]
        mean = [sum(row[j] for row in data) / len(data) for j in range(d)]
        var = [sum((row[j] - mean[j]) ** 2 for row in data) / len(data) for j in range(d)]
        stats[c] = (len(data) / n, mean, var)

    preds = []
    for x in X_test:
        scores = {}
        for c in classes:
            prior, mean, var = stats[c]
            score = math.log(prior)

            for j in range(d):
                v = var[j] + 1e-9
                score += -0.5 * math.log(2 * math.pi * v) - (x[j] - mean[j]) ** 2 / (2 * v)

            scores[c] = score

        preds.append(max(scores, key=scores.get))

    return preds