import json
from typing import List, Dict, Optional

def load_fever_dataset(path: str, limit: Optional[int] = None) -> List[Dict]:
    data = []

    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            item = json.loads(line)

            data.append({
                "claim": item.get("claim", ""),
                "label": item.get("label", ""),
                "evidence": item.get("evidence", [])
            })

            if limit and i + 1 >= limit:
                break

    return data