import uuid
from pytorch_pretrained_bert.tokenization import BertTokenizer

class EvaluatorTokenizer:
    def __init__(self, max_sequence_length=20):
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", "bert-base-uncased")
        self.max_sequence_length = max_sequence_length

    def build_data(self, samples, targets):
        input_ids = []
        segment_ids = []
        for sample, target in zip(samples, targets):
            if len(samples) <= 5:
                print("sample length and target length:", len(sample), len(target))
            input_id, _, segment_id = self.bert_feature_extractor(
                target, sample, max_seq_length=self.max_sequence_length, tokenize_fn=self.tokenizer)
            if len(samples) <= 5:
                print("finished bert feature extractor")
            input_ids.append(input_id)
            segment_ids.append(segment_id)
        return input_ids, segment_ids

    def bert_feature_extractor(
            self, text_a, text_b=None, max_seq_length=512, tokenize_fn=None):
        text_a = text_a[:500]
        text_b = text_b[:500]
        tokens_a = tokenize_fn.tokenize(text_a)
        tokens_b = None
        if text_b:
            tokens_b = tokenize_fn.tokenize(text_b)
        if tokens_b:
            self._truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for one [SEP] & one [CLS] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:max_seq_length - 2]
        if tokens_b:
            input_ids = tokenize_fn.convert_tokens_to_ids(
                ['[CLS]'] + tokens_b + ['[SEP]'] + tokens_a + ['[SEP]'])
            segment_ids = [0] * (len(tokens_b) + 2) + [1] * (len(tokens_a) + 1)
        else:
            input_ids = tokenize_fn.convert_tokens_to_ids(
                ['[CLS]'] + tokens_a + ['[SEP]'])
            segment_ids = [0] * len(input_ids)
        input_mask = None
        return input_ids, input_mask, segment_ids

    def _truncate_seq_pair(self, tokens_a, tokens_b, max_length):
        """Truncates a sequence pair in place to the maximum length.
        Copyed from https://github.com/huggingface/pytorch-pretrained-BERT
        """
        # This is a simple heuristic which will always truncate the longer sequence
        # one token at a time. This makes more sense than truncating an equal percent
        # of tokens from each, since if one sequence is very short then each token
        # that's truncated likely contains more information than a longer sequence.
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            return
        tokens_a = tokens_a[:int(max_length/2)]
        tokens_b = tokens_b[:int(max_length/2)]
        """
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_length:
                break
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()
        """