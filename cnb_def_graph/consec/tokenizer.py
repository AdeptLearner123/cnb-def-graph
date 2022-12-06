import torch
from transformers import AutoTokenizer
from tokenizers import AddedToken


class ConsecTokenizer():
    TARGET_MARKER = ("<d>", "</d>")
    DEF_MARKER = ("DEF_SEP", "DEF_END")
    TOKENIZER_MODEL = "microsoft/deberta-large"

    def __init__(self):
        self._tokenizer = AutoTokenizer.from_pretrained(self.TOKENIZER_MODEL)

        additional_special_tokens = [
            *[AddedToken(t, single_word=True, lstrip=True) for t in self.TARGET_MARKER],
            *[AddedToken(t, single_word=True, lstrip=True) for t in self.DEF_MARKER],
        ]
        self._tokenizer.add_special_tokens({"additional_special_tokens": additional_special_tokens})

    def _get_marked_tokens(self, tokens, target_idx):
        marked_target = f"{self.TARGET_MARKER[0]} {tokens[target_idx]} {self.TARGET_MARKER[1]}"
        return [ token if idx != target_idx else marked_target for idx, token in enumerate(tokens) ]

    def _get_marked_candidate_definitions(self, target, candidate_definitions):
        return [ f"{definition.capitalize()}. {self.DEF_MARKER[0]} {target} {self.DEF_MARKER[1]}" for definition in candidate_definitions]

    def _get_marked_context_definitions(self, tokens, context_definitions):
        return [ (idx, f"{definition.capitalize()}. {self.DEF_MARKER[0]} {tokens[idx]} {self.DEF_MARKER[1]}") for idx, definition in context_definitions ]

    def _deberta_tokenize(self, text):
        # The tokenizer uses the GPT2 tokenizer, which expects words to include a leading space.
        # Refer to this: https://discuss.huggingface.co/t/bpe-tokenizers-and-spaces-before-words/475.
        # Although GPT2 tokenizer should not include a leading space for the first word in a sentence, Consec was trained using a leading space for all words.
        return self._tokenizer(f" {text}", return_attention_mask=False, return_token_type_ids=False, add_special_tokens=False)["input_ids"]

    def _add_to_input_ids(self, input_ids, texts):
        offsets = []

        for text in texts:
            text_ids = self._deberta_tokenize(text)
            offsets.append((len(input_ids), len(input_ids) + len(text_ids)))
            input_ids += text_ids
        
        return offsets

    def _get_input_tokens(self, tokens, candidate_definitions, context_definitions):
        input_ids = [self._tokenizer.cls_token_id]
        token_offsets = self._add_to_input_ids(input_ids, tokens)
        input_ids.append(self._tokenizer.sep_token_id)
        text_ids_len = len(input_ids)
        token_types = [0] * text_ids_len

        candidate_def_offsets = self._add_to_input_ids(input_ids, candidate_definitions)
        context_def_offsets = self._add_to_input_ids(input_ids, [ context_definition for _, context_definition in context_definitions ])
        input_ids.append(self._tokenizer.sep_token_id)

        token_types += [1] * (len(input_ids) - len(token_types))

        return torch.tensor(input_ids), torch.tensor(token_types), text_ids_len, token_offsets, candidate_def_offsets, context_def_offsets

    def _get_relative_positions(self, total_input_ids, text_ids_len, token_offsets, candidate_def_offsets, context_def_offsets, target_idx, context_token_idxs):
        # Relative position at (R, C) is the relative position of R from the perspective of C.
        relative_positions = torch.zeros((total_input_ids, total_input_ids), dtype=torch.long)

        # Each token in text should see other text tokens as a continuous sequence.
        for i in range(text_ids_len):
            relative_positions[:text_ids_len, i] = torch.arange(0, text_ids_len) - i

            # Every text input token should see each definition as appended right behind the text.
            for begin_idx, end_idx in candidate_def_offsets + context_def_offsets:
                def_len = end_idx - begin_idx
                def_relative_pos = torch.arange(text_ids_len - i, text_ids_len - i + def_len)
                relative_positions[begin_idx:end_idx, i] = def_relative_pos
                relative_positions[i, begin_idx:end_idx] = -def_relative_pos

        # Each token in definition should see every token in all other definitions as a continuous sequence.
        all_def_ids_len = total_input_ids - text_ids_len - 1
        for i in range(all_def_ids_len):
            relative_positions[text_ids_len:-1, text_ids_len + i] = torch.arange(all_def_ids_len) - i

        # Every token should see its definition as right behind it
        candidate_token_def_offsets = [ (token_offsets[target_idx], def_offset) for def_offset in candidate_def_offsets ]
        context_token_def_offsets = [ (token_offsets[token_idx], def_offset) for token_idx, def_offset in zip(context_token_idxs, context_def_offsets) ]

        for (token_begin_idx, token_end_idx), (def_begin_idx, def_end_idx) in candidate_token_def_offsets + context_token_def_offsets:
            for i in range(token_begin_idx, token_end_idx):
                relative_positions[def_begin_idx:def_end_idx, i] = (token_end_idx - i) + torch.arange(def_end_idx - def_begin_idx)

        # Every context definition sees its token as right before it
        for (token_begin_idx, token_end_idx), (def_begin_idx, def_end_idx) in context_token_def_offsets:
            for i in range(token_begin_idx, token_end_idx):
                relative_positions[i, def_begin_idx:def_end_idx] = -relative_positions[def_begin_idx:def_end_idx, i]
        
        # The final <SEP> token should see each token from the perspective of the token furthest after it.
        # EX: ["I", "have", "a", "beautiful", "dog"] --- "I" has relative position -4 to "dog",
        # so it should have relative position -5 to the final <SEP> token
        last_token_positions = torch.min(relative_positions[:-1, :-1], dim=-1)[0] - 1
        relative_positions[-1, :-1] = -last_token_positions
        relative_positions[:-1, -1] = last_token_positions

        return relative_positions

    def _get_definitions_mask(self, input_ids, definition_positions):
        definitions_mask = torch.ones_like(input_ids, dtype=torch.float)
        for idx in definition_positions:
            definitions_mask[idx] = 0.0
        return definitions_mask

    def tokenize(self, tokens, target_idx, candidate_definitions, context_definitions):
        marked_tokens = self._get_marked_tokens(tokens, target_idx)
        marked_candidate_defs = self._get_marked_candidate_definitions(tokens[target_idx], candidate_definitions)
        marked_context_defs = self._get_marked_context_definitions(tokens, context_definitions)

        input_ids, token_types, text_ids_len, token_offsets, candidate_def_offsets, context_def_offsets = self._get_input_tokens(marked_tokens, marked_candidate_defs, marked_context_defs)
        
        context_token_idxs = [ idx for idx, _ in context_definitions ]
        relative_positions = self._get_relative_positions(len(input_ids), text_ids_len, token_offsets, candidate_def_offsets, context_def_offsets, target_idx, context_token_idxs)

        definition_positions = [ start for start, _ in candidate_def_offsets ]
        definitions_mask = self._get_definitions_mask(input_ids, definition_positions)
        attention_mask = torch.ones_like(input_ids)
        return input_ids, attention_mask, token_types, relative_positions, definitions_mask, definition_positions
