from cnb_def_graph.sense_inventory.sense_inventory import SenseInventory
from config import SENSE_INVENTORY

import json
from collections import defaultdict

class SenseProposer:
    def __init__(self):
        with open(SENSE_INVENTORY, "r") as file:
            self._sense_inventory = SenseInventory(json.loads(file.read()))
    
    def _assign_multi_word_possibilities(
        self, token_tags, token_senses, length
    ):
        for i in range(len(token_tags) - length + 1):
            span = token_tags[i : i + length]
            span_tokens = [ token for token, _ in span ]
            senses_from_tokens = self._sense_inventory.get_senses(span_tokens)

            for _, senses in token_senses[i : i + length]:
                senses += [ (sense, i, i + length) for sense in senses_from_tokens ]

    def _assign_single_word_possibilities(self, token_tags, token_senses):
        for i, (token, tag) in enumerate(token_tags):
            if tag is None:
                continue
            _, senses = token_senses[i]
            senses += [ (sense, i, i + 1) for sense in self._sense_inventory.get_senses([token], tag) ]

    def _remove_duplicates(self, proposals):
        # Remove duplicate proposals, keeping longest spans
        sense_indices = defaultdict(lambda: (0, 0))
        
        for sense, start, end in proposals:
            previous_start, previous_end = sense_indices[sense]
            if previous_end - previous_start < end - start:
                sense_indices[sense] = (start, end)
        
        return [ (sense, start, end) for sense, (start, end) in sense_indices.items() ]


    def propose_senses(self, token_tags):
        token_proposals = [ (token, []) for token, _ in token_tags ]

        for length in range(2, 5):
            self._assign_multi_word_possibilities(
                token_tags, token_proposals, length
            )
        self._assign_single_word_possibilities(
            token_tags, token_proposals
        )
        
        # Tokens that are stop words should not have any senses.
        # Otherwise compound words that conatin stop words will always have the stop word assigned to that sense.
        # Ex: The sense for "come to" will always have "to" disambiguated to that sense, even if "come to" is not the right sense for "come".
        for i, (token, tag) in enumerate(token_tags):
            if tag is None:
                token_proposals[i] = (token, [])

        # Remove duplicates, keeping longest compound
        for i, (token, proposals) in enumerate(token_proposals):
            token_proposals[i] = (token, self._remove_duplicates(proposals))

        # Sort sense ids so that proposed senses are deterministic.
        token_proposals = [(token, sorted(list(proposals))) for token, proposals in token_proposals]

        return token_proposals
