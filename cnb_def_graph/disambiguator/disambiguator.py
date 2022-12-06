import torch
import bisect 

from config import CONSEC_MODEL_STATE
from cnb_def_graph.utils.read_dicts import read_dicts
from cnb_def_graph.consec.disambiguation_instance import ConsecDisambiguationInstance
from cnb_def_graph.consec.sense_extractor import SenseExtractor
from cnb_def_graph.consec.tokenizer import ConsecTokenizer

from tqdm import tqdm

class Disambiguator:
    BATCH_SIZE = 4
    
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

    
    def _divide_batches(self, active_instances, inputs_list):
        instance_inputs = list(zip(active_instances, inputs_list))
        instance_inputs.sort(key=lambda item: len(item[1][0]), reverse=True)

        sorted_instances = [ instance for instance, _ in instance_inputs ]
        sorted_inputs = [ inputs for _, inputs in instance_inputs ]

        batch_instances = [ sorted_instances[i : i + self.BATCH_SIZE] for i in range(0, len(active_instances), self.BATCH_SIZE) ]
        batch_inputs = [ sorted_inputs[i : i + self.BATCH_SIZE] for i in range(0, len(active_instances), self.BATCH_SIZE) ]

        return list(zip(batch_instances, batch_inputs))


    def batch_disambiguate(self, token_proposals_list):
        disambiguation_instances = [
            ConsecDisambiguationInstance(self._dictionary, self._tokenizer, token_proposals)
            for token_proposals in token_proposals_list
        ]

        while any([ not instance.is_finished() for instance in disambiguation_instances ]):
            active_instances = [ instance for instance in disambiguation_instances if not instance.is_finished() ]

            inputs_list = [ instance.get_next_input() for instance in active_instances ]

            for batch_instances, batch_inputs in tqdm(self._divide_batches(active_instances, inputs_list)):
                #for inputs, instance in zip(inputs_list, batch_instances):
                #    if len(inputs[0]) == 712:
                #        print("Found", instance._tokens, instance._get_context_definitions(), instance._get_candidate_definitions())

                if torch.cuda.is_available():
                    batch_inputs = [ self._send_inputs_to_cuda(inputs) for inputs in batch_inputs ]
                
                batch_inputs = list(zip(*batch_inputs))
                probs_list = self._sense_extractor.batch_extract(*batch_inputs)

                idx_list = [ torch.argmax(torch.tensor(probs)) for probs in probs_list ]

                [ instance.set_result(selected_idx) for instance, selected_idx in zip(batch_instances, idx_list) ]
        
        return [ instance.get_disambiguated_senses() for instance in disambiguation_instances ]