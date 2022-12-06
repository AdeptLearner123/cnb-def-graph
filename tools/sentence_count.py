from cnb_def_graph.utils.read_dicts import read_dicts

def main():
    dictionary = read_dicts()

    total_sents = 0
    for entry in dictionary.values():
        total_sents += len(entry["sentences"])
    
    print("Total", total_sents)