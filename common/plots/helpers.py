import json


def load_metrics(path):
    with path.open() as file:
        return [json.loads(line) for line in file if line.strip()]


def running_average(values, window):
    averages = []
    total = 0.0

    for index, value in enumerate(values):
        total += value
        if index >= window:
            total -= values[index - window]
        averages.append(total / min(index + 1, window))

    return averages
