from cnb_def_graph.token_tagger.token_tagger import TokenTagger
from cnb_def_graph.sense_proposer.sense_proposer import SenseProposer
from cnb_def_graph.disambiguator.disambiguator import Disambiguator
from cnb_def_graph.utils.read_dicts import read_dicts

from time import time
import os
import json
from config import DISAMBIGUATION_BATCHES, DRY_RUN_SENSES
from argparse import ArgumentParser

SAVE_INTERVAL = 2
CHUNK_SIZE = 10000

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--use-amp", action="store_true")
    args = parser.parse_args()
    return args.use_amp


def get_sentence_id(sense, idx):
    return f"{sense}|{idx}"


def parse_sentence_id(sentence_id):
    sense, idx = sentence_id.split("|")
    return sense, idx


def get_disambiguated_sentence_ids():
    sentence_ids = set()

    if not os.path.exists(DISAMBIGUATION_BATCHES):
        os.mkdir(DISAMBIGUATION_BATCHES)

    for filename in os.listdir(DISAMBIGUATION_BATCHES):
        if filename.endswith(".json"):
            with open(os.path.join(DISAMBIGUATION_BATCHES, filename), "r") as file:
                file_json = json.loads(file.read())
                sentence_ids.update(file_json.keys())
    return sentence_ids


def save(batch_id, batch_text_senses):
    print("Saving", len(batch_text_senses))
    filename = f"batch_{batch_id}.json"
    path = os.path.join(DISAMBIGUATION_BATCHES, filename)

    if os.path.isfile(path):
        raise "Batch file path already exists: " + path

    with open(path, "w+") as file:
        file.write(json.dumps(batch_text_senses, sort_keys=True, indent=4, ensure_ascii=False))


def divide_chunks(sense_ids):
    return [ sense_ids[i : i + CHUNK_SIZE] for i in range(0, len(sense_ids), CHUNK_SIZE) ]


def disambiguate_defs(dictionary, sentence_ids, start_batch_id, should_save):
    use_amp = parse_args()
    
    token_tagger = TokenTagger()
    sense_proposer = SenseProposer()
    disambiguator = Disambiguator(use_amp=use_amp)

    batch_id = start_batch_id

    start = time()
    for i, chunk_sentence_ids in enumerate(divide_chunks(sentence_ids)):
        print("CHUNK", i)
        sense_idxs = [ parse_sentence_id(sentence_id) for sentence_id in chunk_sentence_ids ]
        sentence_list = [ dictionary[sense_id]["sentences"][int(idx)] for sense_id, idx in sense_idxs ]
        token_tags_list = [ token_tagger.tokenize_tag(definition) for definition in sentence_list ]
        proposals_list = [ sense_proposer.propose_senses(token_tags) for token_tags in token_tags_list ]

        senses_list = disambiguator.batch_disambiguate(proposals_list)
        batch_result = { sentence_id: {
            "sentence": sentence,
            "senses": senses
        } for sentence_id, sentence, senses in zip(chunk_sentence_ids, sentence_list, senses_list) }

        if should_save:
            save(batch_id, batch_result)
            batch_id += CHUNK_SIZE
    print(f"Time taken: {time() - start}")

    """
    for i, sense_id in tqdm(list(enumerate(sense_ids))):
        definition = dictionary[sense_id]["definition"]
        token_tags = token_tagger.tokenize_tag(definition)
        token_proposals, compound_indices = sense_proposer.propose_senses(token_tags)
        senses = disambiguator.disambiguate(sense_id, token_proposals, compound_indices)
        #senses = disambiguator.batch_disambiguate([ sense_id ], [ token_proposals ], [ compound_indices ])[0]
        senses = [ sense for sense in senses if sense is not None ]
        batch_result[sense_id] = list(set(senses))

        if should_save and (i + 1) % SAVE_INTERVAL == 0:
            save(batch_id, batch_result)
            batch_result = dict()
            batch_id += SAVE_INTERVAL
    if should_save:
        save(batch_id, batch_result)    
    """


def get_sentence_ids(dictionary, sense_ids):
    sentence_ids = []
    for sense_id in sense_ids:
        sentence_ids += [ get_sentence_id(sense_id, i) for i in range(len(dictionary[sense_id]["sentences"])) ]
    return sentence_ids


def disambiguate_all():
    dictionary = read_dicts()
    sentence_ids = get_sentence_ids(dictionary, list(dictionary.keys()))
    disambiguated_sentence_ids = get_disambiguated_sentence_ids()
    missing_sentence_ids = set(sentence_ids).difference(disambiguated_sentence_ids)

    print("Sentence ids:", len(missing_sentence_ids), "/", len(dictionary))

    disambiguate_defs(dictionary, missing_sentence_ids, len(disambiguated_sentence_ids), True)


def dry_run():
    with open(DRY_RUN_SENSES, "r") as file:
        sense_ids = file.read().splitlines()
    
    dictionary = read_dicts()
    sentence_ids = get_sentence_ids(dictionary, sense_ids)

    print("Sentence ids", len(sentence_ids))

    disambiguate_defs(dictionary, sentence_ids, 0, False)


def create_dry_run():
    import random

    dictionary = read_dicts()
    dry_run_senses = random.sample(dictionary.keys(), 512)
    with open(DRY_RUN_SENSES, "w+") as file:
        file.write("\n".join(dry_run_senses))
