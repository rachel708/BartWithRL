import torch
import uuid
from mt_dnn.model import MTDNNModel

class EvaluatorModel:
    def __init__(self, checkpoint):
        #torch.cuda.set_device(0)
        self.cuda = torch.cuda.is_available()
        if self.cuda:
            state_dict = torch.load(checkpoint)
        else:
            state_dict = torch.load(checkpoint, map_location="cpu")
        config = state_dict['config']
        config["cuda"] = self.cuda
        self.model = MTDNNModel(config, state_dict=state_dict)
        self.model.load(checkpoint)
        if self.cuda:
            self.model.cuda()

    def make_baches(self, data, batch_size=32):
        return [data[i:i + batch_size] for i in range(0, len(data), batch_size)]

    def predict(self, input_ids, segment_ids, do_print=False):
        scores = []
        preds = []
        data = self.load_data(input_ids, segment_ids)
        if do_print:
            print("data:", data)
        #data = self.make_baches(self.load_data(input_ids, segment_ids))
        #for batch in data:
        batch_data, batch_info = self.prepare_model_input(data)
        if do_print:
            print("batch_info:", batch_info)
        score, pred, gold = self.model.predict(batch_info, batch_data)
        if do_print:
            print("pred:", pred)
        scores.extend(score)
        preds.extend(pred)
        return scores, preds

    def prepare_model_input(self, batch):
        batch_size = len(batch)
        tok_len = max(len(x['token_id']) for x in batch)
        hypothesis_len = max(len(x['type_id']) - sum(x['type_id']) for x in batch)
        token_ids = torch.LongTensor(batch_size, tok_len).fill_(0)
        type_ids = torch.LongTensor(batch_size, tok_len).fill_(0)
        masks = torch.LongTensor(batch_size, tok_len).fill_(0)
        premise_masks = torch.ByteTensor(batch_size, tok_len).fill_(1)
        hypothesis_masks = torch.ByteTensor(batch_size, hypothesis_len).fill_(1)
        for i, sample in enumerate(batch):
            select_len = len(sample['token_id'])
            token_ids[i, :select_len] = torch.LongTensor(sample['token_id'][:select_len])
            type_ids[i, :select_len] = torch.LongTensor(sample['type_id'][:select_len])
            masks[i, :select_len] = torch.LongTensor([1] * select_len)
            hlen = len(sample['type_id']) - sum(sample['type_id'])
            hypothesis_masks[i, :hlen] = torch.LongTensor([0] * hlen)
            for j in range(hlen, select_len):
                premise_masks[i, j] = 0
        batch_data = [token_ids, type_ids, masks, premise_masks, hypothesis_masks]
        batch_info = {
                'token_id': 0,
                'segment_id': 1,
                'mask': 2,
                'premise_mask': 3,
                'hypothesis_mask': 4,
                'task_id':0,
                'input_len':len(batch_data),
                'task_type':"Classification",
                'pairwise_size':1,
                'label':[sample['label'] for sample in batch],
                'uids':[sample['uid'] for sample in batch]  
                }
        if self.cuda:
            for i, item in enumerate(batch_data):
                batch_data[i] = self.patch(item.pin_memory())
        return batch_data, batch_info

    def load_data(self, input_ids, segment_ids):
        data = []
        for input_id, segment_id in zip(input_ids, segment_ids):
            sample = {'uid': uuid.uuid1(),
                      'label': 'entailment',
                      'token_id': input_id,
                      'type_id': segment_id,
                      'factor': 1.0}
            data.append(sample)
        return data

    def patch(self, v):
        v = v.cuda(non_blocking=True)
        return v