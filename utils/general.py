import json

def load_json(json_path):
    content = json.load(open(json_path, "r"))
    return content