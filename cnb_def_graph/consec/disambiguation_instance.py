class ConsecDisambiguationInstance:
    """
    Represents a single piece of text being disambiguated by Consec.
    Generates the Consec inputs sequentially in order of increasing polysemy.
    """

    MAX_CONTEXT_DEFS = 5

    def __init__(self, dictionary, tokenizer, token_proposals):
        self._dictionary = dictionary
        self._proposals = [ proposals for _, proposals in token_proposals ]
        self._set_disambiguation_order()
        self._tokenizer = tokenizer
        self._tokens = [ token for token, _ in token_proposals ]

    def _get_definition(self, sense):
        return self._dictionary[sense]["definition"]
    
    def _set_disambiguation_order(self):
        disambiguated_senses = [None] * len(self._proposals)

        token_polysemy = []
        for i, proposals in enumerate(self._proposals):
            if len(proposals) == 1:
                disambiguated_senses[i] = proposals[0]
            elif len(proposals) > 1:
                token_polysemy.append((i, len(proposals)))

        token_polysemy = sorted(token_polysemy, key=lambda item:item[1])
        disambiguation_order = [ i for i, _ in token_polysemy]
        self._disambiguated_senses = disambiguated_senses
        self._disambiguation_order = disambiguation_order
    
    def is_finished(self):
        return len(self._disambiguation_order) == 0
    
    def get_next_input(self):
        idx = self._disambiguation_order[0]
        context_definitions = self._get_context_definitions()
        candidate_senses, candidate_definitions = self._get_candidate_definitions()
        
        tokenizer_result = self._tokenizer.tokenize(self._tokens, idx, candidate_definitions, context_definitions)
        return tokenizer_result, candidate_senses

    def _get_candidate_definitions(self):
        idx = self._disambiguation_order[0]
        candidate_senses = [ sense for sense, _, _ in self._proposals[idx] ]
        return candidate_senses, [ self._get_definition(sense) for sense in candidate_senses ]

    def _get_context_definitions(self):
        context_sense_spans = [ (i, item) for i, item in enumerate(self._disambiguated_senses) if item is not None ]
        context_sense_spans = context_sense_spans[:self.MAX_CONTEXT_DEFS]
        return [ (i, self._get_definition(sense)) for i, (sense, _, _) in context_sense_spans ]

    def set_result(self, selected_idx):
        token_idx = self._disambiguation_order[0]
        sense, start, end = self._proposals[token_idx][selected_idx]

        for token_idx in range(start, end):
            self._disambiguated_senses[token_idx] = (sense, start, end)

            if token_idx in self._disambiguation_order:
                self._disambiguation_order.remove(token_idx)
    
    def get_disambiguated_senses(self):
        return self._disambiguated_senses