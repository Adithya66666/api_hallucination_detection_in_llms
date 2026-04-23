from datasets import load_dataset

def load_truthfulqa():
    ds = load_dataset("domenicrosati/TruthfulQA")
    return ds["validation"]