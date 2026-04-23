import re

def extract_entities(text: str):
    # simple heuristic: extract capitalized phrases
    matches = re.findall(r'([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)', text)

    # remove common words
    blacklist = ["The", "It", "This"]
    entities = [m for m in matches if m not in blacklist]

    return list(set(entities))