import csv
from pathlib import Path


def save_predictions(predictions, save_path=Path("../submission.csv")):
    with open(save_path, "w") as pred:
        csv_out = csv.writer(pred)
        csv_out.writerow(['id', 'label'])
        for i, row in enumerate(predictions):
            csv_out.writerow([i, row])
