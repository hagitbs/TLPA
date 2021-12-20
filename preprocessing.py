import ijson
import simplejson as json

with open("data/LOCO.json") as f:
    data = ijson.items(f, 'item')
    items = [v for v in data]  
    for i in range(0, len(items), 5000):
        with open(f"data/LOCO_{i}.json", "w") as fp:
            json.dump(items[i:i+5000], fp)