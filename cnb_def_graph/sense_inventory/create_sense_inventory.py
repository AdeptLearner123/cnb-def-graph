import json
from tqdm import tqdm

from cnb_def_graph.utils.read_dicts import read_dicts
from cnb_def_graph.token_tagger.token_tagger import TokenTagger
from config import SENSE_INVENTORY

def main():
    print("Status:", "reading")
    dictionary = read_dicts()

    print("Status:", "compiling map")
    token_tagger = TokenTagger()
    lemma_senses = dict()

    for sense_id, entry in tqdm(list(dictionary.items())):
        lemma_forms = entry["wordForms"]
        pos = entry["pos"]
        for lemma_form in lemma_forms:
            lemma_key = f"{lemma_form}|{pos}"
            if lemma_key not in lemma_senses:
                lemma_tokens = [ token for token, _ in token_tagger.tokenize_tag(lemma_form) ]
                lemma_senses[lemma_key] = {
                    "senses": [],
                    "tokens": lemma_tokens
                }
            lemma_senses[lemma_key]["senses"].append(sense_id)

    print("Status:", "writing")
    with open(SENSE_INVENTORY, "w+") as file:
        file.write(json.dumps(lemma_senses, sort_keys=True, ensure_ascii=False, indent=4))


if __name__ == "__main__":
    main()
