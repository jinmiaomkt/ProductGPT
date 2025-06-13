import json, itertools
first = json.loads(open("/Users/jxm190071/Dropbox/Mac/Desktop/E2 Genshim Impact/Data/lstm_predictions.jsonl").readline())
print(len(first["probs"][0]))   # → 10   ← should be 9