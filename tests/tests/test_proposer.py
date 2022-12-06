from cnb_def_graph.sense_proposer.sense_proposer import SenseProposer
from config import TEST_LABELS

import os
import json

def lists_to_tuples(list):
    return [ tuple(x) for x in list ]

def test_tokenizer():
    sense_proposer = SenseProposer()

    for filename in os.listdir(TEST_LABELS):
        with open(os.path.join(TEST_LABELS, filename), "r") as file:
            print("Testing", filename)
            labels = json.loads(file.read())

            token_tags = [ (item["token"], item["tag"]) for item in labels["tokens"] ]
            expected_proposals = [ item["proposals"] if "proposals" in item else [] for item in labels["tokens"] ]
            expected_proposals = [ lists_to_tuples(item) for item in expected_proposals ]

            token_proposals = sense_proposer.propose_senses(token_tags)
            proposals_list = [ proposals for _, proposals in token_proposals ]

            proposals_list = [ sorted(proposals) for proposals in proposals_list ]
            expected_proposals = [ sorted(proposals) for proposals in expected_proposals ]
            print(proposals_list)
            print(expected_proposals)
            assert proposals_list == expected_proposals, f"Expected proposals {expected_proposals} but was {proposals_list}"
