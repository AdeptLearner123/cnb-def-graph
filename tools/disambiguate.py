from cnb_def_graph.utils.read_dicts import read_dicts
from cnb_def_graph.token_tagger.token_tagger import TokenTagger
from cnb_def_graph.sense_proposer.sense_proposer import SenseProposer
from cnb_def_graph.disambiguator.disambiguator import Disambiguator
from argparse import ArgumentParser

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("sense_id", type=str)
    args = parser.parse_args()
    return args.sense_id


def main():
    dictionary = read_dicts()
    sense_id = parse_args()

    token_tagger = TokenTagger()
    sense_proposer = SenseProposer()
    disambiguator = Disambiguator()

    definition = dictionary[sense_id]["definition"]
    token_tags = token_tagger.tokenize_tag(definition)

    print("Token tags", token_tags)

    token_proposals, compound_indices = sense_proposer.propose_senses(token_tags)
    sense_indices = disambiguator.disambiguate(sense_id, token_proposals, compound_indices)

    print("Senses", sense_indices)


if __name__ == "__main__":
    main()