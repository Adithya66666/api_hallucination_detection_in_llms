from app.dataset_loader import load_fever_dataset

# Load 5 lines for testing
data = load_fever_dataset("data/train.jsonl", limit=5)

for i, item in enumerate(data):
    print(f"Claim {i+1}: {item['claim']}")
    print(f"Label: {item['label']}")
    print(f"Evidence: {item['evidence']}")
    print("------")