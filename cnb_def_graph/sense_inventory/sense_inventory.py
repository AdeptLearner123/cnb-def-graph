from collections import defaultdict

class SenseInventory:
    def __init__(self, sense_inventory_data):
        self._key_to_senses = defaultdict(lambda: [])
        
        for lemma_pos, entry in sense_inventory_data.items():
            _, pos = lemma_pos.split("|")
            lemma_tokens = tuple(entry["tokens"])
            senses = entry["senses"]

            keys = [ (lemma_tokens, None), (lemma_tokens, pos) ]
            
            for key in keys:
                self._key_to_senses[key] += senses
        
    def get_senses(self, tokens, pos=None):
        key = (tuple(tokens), pos)
        return self._key_to_senses[key]
