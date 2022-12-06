from config import DICTIONARIES

import os
import json

def read_dicts():
    dictionary = dict()

    for filename in os.listdir(DICTIONARIES):
        if filename.endswith(".json"):
            with open(os.path.join(DICTIONARIES, filename), "r") as file:
                dictionary.update(json.loads(file.read()))
    
    return dictionary
