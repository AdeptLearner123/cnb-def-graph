from config import DRY_RUN_SENSES
from cnb_def_graph.consec.disambiguation_instance import ConsecDisambiguationInstance
from cnb_def_graph.consec.tokenizer import ConsecTokenizer
from cnb_def_graph.utils.read_dicts import read_dicts
from cnb_def_graph.token_tagger.token_tagger import TokenTagger
from cnb_def_graph.sense_proposer.sense_proposer import SenseProposer

MAX_INPUTS_LENGTH = 750

def test_input_lengths():
    with open(DRY_RUN_SENSES, "r") as file:
        senses = file.read().splitlines()

    dictionary = read_dicts()
    tokenizer = ConsecTokenizer()
    token_tagger = TokenTagger()
    sense_proposer = SenseProposer()

    for sense in senses:
        for sentence in dictionary[sense]["sentences"]:
            token_tags = token_tagger.tokenize_tag(sentence)
            token_proposals, compound_indices = sense_proposer.propose_senses(token_tags)
            disambiguation_instance = ConsecDisambiguationInstance(dictionary, tokenizer, sense, token_proposals, compound_indices)

            while not disambiguation_instance.is_finished():
                inputs, candidate_senses = disambiguation_instance.get_next_input()
                _, candidate_definitions = disambiguation_instance._get_candidate_definitions()

                print(sentence)
                print(disambiguation_instance._get_context_definitions())
                print(candidate_definitions)
                curr_idx = disambiguation_instance._disambiguation_order[disambiguation_instance._current]
                curr_token, _ = disambiguation_instance._token_senses[curr_idx]

                assert len(inputs[0]) < MAX_INPUTS_LENGTH, f"Input lengths exceeded for {sense}, {curr_token}, actual length: {len(inputs[0])}"

                # Select the sense with the longest definition
                selected_sense, _ = max(zip(candidate_senses, candidate_definitions), key=lambda sense_definition: len(sense_definition[1]))
                disambiguation_instance.set_result(selected_sense)