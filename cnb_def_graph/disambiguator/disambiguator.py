import torch

from config import CONSEC_MODEL_STATE
from cnb_def_graph.utils.read_dicts import read_dicts
from cnb_def_graph.consec.disambiguation_instance import ConsecDisambiguationInstance
from cnb_def_graph.consec.sense_extractor import SenseExtractor
from cnb_def_graph.consec.tokenizer import ConsecTokenizer

from tqdm import tqdm

class Disambiguator:
    BATCH_SIZE = 1

    def __init__(self, debug_mode=False, use_amp=False):
        self._dictionary = read_dicts()
        self._debug_mode = debug_mode
        state_dict = torch.load(CONSEC_MODEL_STATE)
        self._sense_extractor = SenseExtractor(use_amp=use_amp)
        self._sense_extractor.load_state_dict(state_dict)
        self._sense_extractor.eval()
        self._tokenizer = ConsecTokenizer()

        if torch.cuda.is_available():
            self._sense_extractor.cuda()

    """
    def _disambiguate_tokens(self, sense_id, token_senses, compound_indices):
        disambiguation_instance = ConsecDisambiguationInstance(self._dictionary, self._tokenizer, sense_id, token_senses, compound_indices)

        while not disambiguation_instance.is_finished():
            input, senses = disambiguation_instance.get_next_input()
            if torch.cuda.is_available():
                input = self._send_inputs_to_cuda(input)

            probs = self._sense_extractor.extract(*input)

            if self._debug_mode:
                sense_idxs = torch.tensor(probs).argsort(descending=True)
                for sense_idx in sense_idxs:
                    print(f"{senses[sense_idx]}:  {probs[sense_idx]}")

            sense_idx = torch.argmax(torch.tensor(probs))
            disambiguation_instance.set_result(senses[sense_idx])

        return disambiguation_instance.get_disambiguated_senses()
    """

    def _send_inputs_to_cuda(self, inputs):
        (input_ids, attention_mask, token_types, relative_pos, def_mask, def_pos) = inputs

        input_ids = input_ids.cuda()
        attention_mask = attention_mask.cuda()
        token_types = token_types.cuda()
        relative_pos = relative_pos.cuda()
        def_mask = def_mask.cuda()

        return (input_ids, attention_mask, token_types, relative_pos, def_mask, def_pos)

    def disambiguate(self, sense_id, token_senses, compound_indices):
        return self._disambiguate_tokens(sense_id, token_senses, compound_indices)

    def _divide_batches(self, active_instances, input_senses_list):
        instance_batches = [ active_instances[i : i + self.BATCH_SIZE] for i in range(0, len(active_instances), self.BATCH_SIZE) ]
        input_senses_batches = [ input_senses_list[i : i + self.BATCH_SIZE] for i in range(0, len(input_senses_list), self.BATCH_SIZE) ]
        return list(zip(instance_batches, input_senses_batches))

    def batch_disambiguate(self, token_proposals_list):
        disambiguation_instances = [
            ConsecDisambiguationInstance(self._dictionary, self._tokenizer, token_proposals)
            for token_proposals in token_proposals_list
        ]

        while any([ not instance.is_finished() for instance in disambiguation_instances ]):
            active_instances = [ instance for instance in disambiguation_instances if not instance.is_finished() ]

            inputs_senses_list = [ instance.get_next_input() for instance in active_instances ]

            for batch_instances, batch_inputs_senses in tqdm(self._divide_batches(active_instances, inputs_senses_list)):
                inputs_list = [ inputs for inputs, _ in batch_inputs_senses ]

                if torch.cuda.is_available():
                    inputs_list = [ self._send_inputs_to_cuda(inputs) for inputs in inputs_list ]
                
                inputs_list = list(zip(*inputs_list))
                probs_list = self._sense_extractor.batch_extract(*inputs_list)

                idx_list = [ torch.argmax(torch.tensor(probs)) for probs in probs_list ]

                [ instance.set_result(selected_idx) for instance, selected_idx in zip(batch_instances, idx_list) ]
        
        return [ instance.get_disambiguated_senses() for instance in disambiguation_instances ]