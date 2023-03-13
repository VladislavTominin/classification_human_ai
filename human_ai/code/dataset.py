import pandas as pd
from pathlib import Path
from datasets import load_dataset
import datasets
from datasets import Dataset


def get_train_dataset(data_path=Path.home() / 'classification_human_ai/Students_Data/train_set.json', test_size=None):
    df = pd.read_json(data_path)
    dataset = Dataset.from_pandas(df)

    if test_size is not None:
        return dataset.class_encode_column('label').train_test_split(
            test_size=test_size, seed=42, stratify_by_column='label'
        )
    return dataset


def get_test_dataset(data_path=Path.home() / 'classification_human_ai/Students_Data/test_set.json'):
    df = pd.read_json(data_path)
    dataset = Dataset.from_pandas(df)
    return dataset
