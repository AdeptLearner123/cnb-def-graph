from cnb_def_graph.token_tagger.token_tagger import TokenTagger
from cnb_def_graph.utils.read_dicts import read_dicts
from config import TEST_LABELS

import os
import json

def test_tokenizer():
    dictionary = read_dicts()
    token_tagger = TokenTagger()

    for filename in os.listdir(TEST_LABELS):
        with open(os.path.join(TEST_LABELS, filename), "r") as file:
            print("Testing", filename)
            labels = json.loads(file.read())
            sense_id = labels["sense_id"]
            sentence_idx = int(labels["sentence"])

            expected_token_tags = [ (item["token"], item["tag"]) for item in labels["tokens"] ]
            token_tags = token_tagger.tokenize_tag(dictionary[sense_id]["sentences"][sentence_idx])

            expected_tokens = [ token for token, _ in expected_token_tags ]
            tokens = [ token for token, _ in token_tags ]

            print(expected_tokens)
            print(tokens)

            assert expected_tokens == tokens, f"Expected tokens {expected_tokens} but was {tokens}"

            for (token, tag), (_, expected_tag) in zip(token_tags, expected_token_tags):
                assert tag == expected_tag, f"Expected token {token} to have tag {expected_tag} but was {tag}"