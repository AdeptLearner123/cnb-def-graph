from cnb_def_graph.disambiguator.disambiguator import Disambiguator
from cnb_def_graph.sense_proposer.sense_proposer import SenseProposer
from config import TEST_LABELS

import os
import json

def lists_to_tuples(list):
    return [ tuple(x) for x in list ]


def test_disambiguator():
    disambiguator = Disambiguator()

    for filename in os.listdir(TEST_LABELS):
        with open(os.path.join(TEST_LABELS, filename), "r") as file:
            print("Testing", filename)
            labels = json.loads(file.read())

            token_proposals = [ (item["token"], lists_to_tuples(item["proposals"]) if "proposals" in item else []) for item in labels["tokens"] ]
            tokens = [ token for token, _ in token_proposals ]
            print("Token proposals")

            senses_list = disambiguator.batch_disambiguate([ token_proposals ])
            senses = senses_list[0]
            print("Expected senses")

            expected_senses = [ lists_to_tuples(item["sense"]) if "sense" in item else None for item in labels["tokens"] ]

            for token, sense, expected_sense_options in zip(tokens, senses, expected_senses):
                if expected_sense_options is None:
                    assert sense is None, f"Expected {token} to have no sense but was {sense}"
                else:
                    assert sense in expected_sense_options, f"Expected {token} to have sense in {expected_sense_options} but was {sense}"