import math
from typing import Dict, List, Optional
from torch import Tensor
import torch
import torch.nn as nn

from fairseq import metrics, utils, search
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.models import FairseqIncrementalDecoder
from fairseq.models.fairseq_encoder import EncoderOut
from fairseq.data import data_utils

from fairseq.data.encoders.gpt2_bpe import GPT2BPE
import numpy as np

from fairseq.criterions.evaluator_tokenizer import EvaluatorTokenizer
from fairseq.criterions.evaluator_model import EvaluatorModel

def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None, reduce=True):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.)
        smooth_loss.masked_fill_(pad_mask, 0.)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)

    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    eps_i = epsilon / lprobs.size(-1)
    loss = (1. - epsilon) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss

@register_criterion('label_smoothed_cross_entropy_with_RL')
class LabelSmoothedCrossEntropyWithRLCriterion(FairseqCriterion):
    def __init__(self, task, sentence_avg, label_smoothing, lambda_ratio, rl_k, max_decoding_len, rl_len_penalty,
                 gpt2_encoder_json, gpt2_vocab_bpe, mtdnn_max_len, mtdnn_path):
        super().__init__(task)
        assert hasattr(task, 'target_dictionary'), "only support task with attr target_dictionary"
        self.tgt_dict = task.target_dictionary
        self.src_dict = task.source_dictionary
        self.sentence_avg = sentence_avg
        self.eps = label_smoothing
        self.rl_k = rl_k
        self.max_decoding_len = max_decoding_len
        self.lambda_ratio = lambda_ratio
        self.len_penalty = rl_len_penalty

        self.eos = self.tgt_dict.eos()
        self.pad = self.tgt_dict.pad()
        self.unk = self.tgt_dict.unk()
        self.vocab_size = len(self.tgt_dict)

        kwargs = {"gpt2_encoder_json": gpt2_encoder_json, "gpt2_vocab_bpe": gpt2_vocab_bpe}
        self.bpe = GPT2BPE(kwargs)
        print("bpe:", self.bpe)

        self.greedy_searcher = search.BeamSearch(self.tgt_dict)
        self.sampling_searcher = search.Sampling(self.tgt_dict)

        self.evaluator_tokenizer = EvaluatorTokenizer(mtdnn_max_len)
        self.evaluator_model = EvaluatorModel(mtdnn_path)

    @staticmethod
    def add_args(parser):
        parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')

        parser.add_argument('--lambda-ratio', default=0., type=float,
                            help='ratio for balancing RL and SL. 0 will lead to pure SL training. 1 will lead to pure RL training.')

        parser.add_argument('--rl-k', default=5, type=int,
                            help='hyper-parameter k of rl training')

        parser.add_argument('--max_decoding_len', default=10, type=int,
                            help='max decoding step of rl training')

        parser.add_argument('--rl-len-penalty', default=0., type=float,
                            help='length penalty of rl training. 0 means no penalty')

        parser.add_argument('--gpt2-encoder-json', default="/home/work/tianqi/fairseq/fairseq/encoder.json", type=str,
                            help='path of gpt2 encoder file')

        parser.add_argument('--gpt2-vocab-bpe', default="/home/work/tianqi/fairseq/fairseq/vocab.bpe", type=str,
                            help='path of gpt2 encoder file')

        parser.add_argument('--mtdnn-max-len', default=10, type=int,
                            help='max length of mtdnn model')

        parser.add_argument('--mtdnn-path', default="/home/work/tianqi/paraphrase/mtdnn/mt-dnn/scripts/checkpoints/adsnli_adamax_answer_opt0_gc0_ggc1_2019-10-22T0240/model_6.pt", type=str,
                            help='path of mtdnn model')


    def rl_forward_core(self, model, sample, searcher, rl_k, bsz, src_tokens, src_lengths, encoder_outs, print_for_debug=False):
        max_len = self.max_decoding_len
        new_order = torch.arange(bsz).view(-1,1).repeat(1, rl_k).view(-1) #[batch_size*rl_k][0,0,1,1,2,2...]
        new_order = new_order.to(src_tokens.device).long()
        encoder_outs = model.reorder_encoder_out(encoder_outs, new_order) #[bsz*rl_k, src_len, hidden_size]
        assert encoder_outs is not None

        #initialize buffers
        scores = (
            torch.zeros(bsz*rl_k, max_len + 1).to(src_tokens).float()
            )
        tokens = (
            torch.zeros(bsz*rl_k, max_len + 2)
            .to(src_tokens)
            .long()
            .fill_(self.pad)
            )
        tokens[:, 0] = self.eos 
        if hasattr(self.tgt_dict, "bos_index"):
           tokens[:, 0] = self.tgt_dict.bos_index

        blacklist = (
            torch.zeros(bsz, rl_k).to(src_tokens).eq(-1)
            )

        # list of completed sentences
        finalized = torch.jit.annotate(
            List[List[Dict[str, Tensor]]],
            [torch.jit.annotate(List[Dict[str, Tensor]], []) for i in range(bsz)],
        )  # contains lists of dictionaries of infomation about the hypothesis being finalized at each step

        finished = [
            False for i in range(bsz)
        ]  # a boolean array indicating if the sentence at the index is finished or not
        num_remaining_sent = bsz  # number of sentences remaining

        cand_bbsz_offsets = (torch.arange(0, bsz) * rl_k).unsqueeze(1).type_as(tokens)
        cand_offsets = torch.arange(0, 2*rl_k).type_as(tokens)

        reorder_state: Optional[Tensor] = None
        batch_idxs: Optional[Tensor] = None

        for step in range(max_len + 1):
            # reorder decoder internal states based on the prev choice of beams
            # print(f'step: {step}')
            # print("batch_size is {} at step {}".format(bsz, step))
            if reorder_state is not None:
                if batch_idxs is not None:
                    # update beam indices to take into account removed sentences
                    corr = batch_idxs - torch.arange(batch_idxs.numel()).type_as(
                        batch_idxs
                    )
                    reorder_state.view(-1, rl_k).add_(
                        corr.unsqueeze(-1) * rl_k
                    )
                model.reorder_incremental_state(reorder_state)
                encoder_outs = model.reorder_encoder_out(
                    encoder_outs, reorder_state
                )

            lprobs, _ = model.forward_decoder(
                tokens[:, : step+1], encoder_outs
                )
            if print_for_debug:
                print("step ", step, ": lprobs before assign", lprobs[0])
            lprobs[lprobs != lprobs] = torch.tensor(-math.inf).to(lprobs)

            if step >= max_len:
                lprobs[:, : self.eos] = -math.inf
                lprobs[:, self.eos+1:] = -math.inf

            scores = scores.type_as(lprobs)

            eos_bbsz_idx = torch.empty(0).to(tokens)
            eos_scores = torch.empty(0).to(scores)

            searcher.set_src_lengths(src_lengths)

            cand_scores, cand_indices, cand_beams = searcher.step(
                step,
                lprobs.view(bsz, -1, self.vocab_size),
                scores.view(bsz, rl_k, -1)[:, :, :step]
                )
            if print_for_debug:
                print("step ", step, ":", lprobs[0], scores[0])

            cand_bbsz_idx = cand_beams.add(cand_bbsz_offsets)#[bsz, rl_k*2]
            eos_mask = cand_indices.eq(self.eos) & cand_scores.ne(-math.inf)#[bsz, rl_k*2]
            eos_mask[:, :rl_k][blacklist] = torch.tensor(0).to(eos_mask)

            eos_bbsz_idx = torch.masked_select(
                cand_bbsz_idx[:,:rl_k], mask=eos_mask[:,:rl_k])
            
            finalized_sents: List[int] = []
            if eos_bbsz_idx.numel() > 0:
                eos_scores = torch.masked_select(
                    cand_scores[:, :rl_k], mask=eos_mask[:, :rl_k])
                finalized_sents = self.finalize_hypos(
                    step,
                    eos_bbsz_idx,
                    eos_scores,
                    tokens,
                    scores,
                    finalized,
                    finished,
                    rl_k,
                    max_len)
                num_remaining_sent -= len(finalized_sents)

            assert num_remaining_sent >= 0
            if num_remaining_sent == 0:
                break
            assert step < max_len

            if len(finalized_sents) > 0:
                new_bsz = bsz - len(finalized_sents)
                batch_mask = torch.ones(bsz).to(cand_indices)
                batch_mask[
                    torch.tensor(finalized_sents).to(cand_indices)
                    ] = torch.tensor(0).to(batch_mask)
                batch_idxs = batch_mask.nonzero().squeeze(-1)

                eos_mask = eos_mask[batch_idxs]
                cand_beams = cand_beams[batch_idxs]
                cand_bbsz_offsets.resize_(new_bsz, 1)
                cand_bbsz_idx = cand_beams.add(cand_bbsz_offsets)
                cand_scores = cand_scores[batch_idxs]
                cand_indices = cand_indices[batch_idxs]
                src_lengths = src_lengths[batch_idxs]
                blacklist = blacklist[batch_idxs]
                scores = scores.view(bsz, -1)[batch_idxs].view(new_bsz * rl_k, -1)
                tokens = tokens.view(bsz, -1)[batch_idxs].view(new_bsz * rl_k, -1)
                bsz = new_bsz
            else:
                batch_idxs = None

            # print("step {}: batch_idxs {}".format(step, batch_idxs))
            # print("step {}: cand_bbsz_idx {}".format(step, cand_bbsz_idx))
            eos_mask[:, :rl_k] = ~((~blacklist) & (~eos_mask[:, :rl_k]))
            active_mask = torch.add(
                eos_mask.type_as(cand_offsets) * (2 * rl_k),
                cand_offsets[: eos_mask.size(1)]
                )

            new_blacklist, active_hypos = torch.topk(
                active_mask, k=rl_k, dim=1, largest=False)

            #update blacklist to ignore any finalized hypos
            blacklist = new_blacklist.ge(2*rl_k)[:, :rl_k]
            assert (~blacklist).any(dim=1).all()

            active_bbsz_idx = torch.gather(cand_bbsz_idx, dim=1, index=active_hypos)
            active_scores = torch.gather(cand_scores, dim=1, index=active_hypos)

            active_bbsz_idx = active_bbsz_idx.view(-1)
            active_scores = active_scores.view(-1)

            tokens[:, :step+1] = torch.index_select(
                tokens[:, :step+1], dim=0, index=active_bbsz_idx
                )
            tokens.view(bsz, rl_k, -1)[:, :, step+1] = torch.gather(
                cand_indices, dim=1, index=active_hypos)

            if step > 0:
                scores[:, :step] = torch.index_select(
                    scores[:, :step], dim=0, index=active_bbsz_idx
                )
            scores.view(bsz, rl_k, -1)[:, :, step] = torch.gather(
                cand_scores, dim=1, index=active_hypos
            )

            reorder_state = active_bbsz_idx

        finalized_tokens = []
        finalized_scores = []
         # sort by score descending
        for i in range(rl_k):
            _finalized_tokens = []
            _finalized_scores = []
            for sent in range(len(finalized)):
                beam = finalized[sent][i]
                _finalized_tokens.append(beam["tokens"])
                _finalized_scores.append(beam["score"])
            finalized_tokens.append(_finalized_tokens)
            finalized_scores.append(_finalized_scores)

        #finalized_tokens [rl_k, bsz]
        return finalized_tokens, finalized_scores

    def rl_forward(self, model, sample):
        encoder_input: Dict[str, Tensor] = {}
        for k,v in sample["net_input"].items():
            if k!= "prev_output_tokens":
                encoder_input[k] = v

        src_tokens = encoder_input["src_tokens"]
        src_lengths = (
            (src_tokens.ne(self.eos) & src_tokens.ne(self.pad)).long().sum(dim=1)
            )#[batch_size, 1]

        input_size= src_tokens.size()
        bsz, src_len = input_size[0], input_size[1] #batch_size, max_source_length

        rl_model = EnsembleModel([model.module])
        #rl_model.cuda()
        rl_model.reset_incremental_state()

        encoder_outs = rl_model.forward_encoder(
            src_tokens=encoder_input["src_tokens"],
            src_lengths=encoder_input["src_lengths"],
            )#[bsz, src_len, hidden_size]

        greedy_finalized_tokens, greedy_finalized_scores = self.rl_forward_core(rl_model, 
                                                sample, 
                                                self.greedy_searcher,
                                                1,
                                                bsz, 
                                                src_tokens, 
                                                src_lengths, 
                                                encoder_outs,
                                                print_for_debug=False)

        rl_model.reset_incremental_state()

        sample_finalized_tokens, sample_finalized_scores = self.rl_forward_core(rl_model, 
                                                sample, 
                                                self.sampling_searcher,
                                                self.rl_k,
                                                bsz, 
                                                src_tokens, 
                                                src_lengths, 
                                                encoder_outs,
                                                print_for_debug=False)
        rl_model.reset_incremental_state()

        #sample_finalized = greedy_finalized

        rl_reward = self.cal_reward(encoder_input["src_tokens"], 
                                    greedy_finalized_tokens,
                                    greedy_finalized_scores,
                                    sample_finalized_tokens,
                                    sample_finalized_scores)
        #print("rl_reward:", rl_reward)
        return rl_reward

    def cal_reward(self, src_tokens, 
                   greedy_finalized_tokens, greedy_finalized_scores,
                   sample_finalized_tokens, sample_finalized_scores):
        _sampling_rewards = []
        _greedy_rewards = []
        _rewards_diff = []
        _rl_losses = []

        #print("src_tokens:", src_tokens[0])
        #print("greedy_finalized_tokens:", greedy_finalized_tokens[0][0])
        #print("sample_finalized_tokens:", sample_finalized_tokens[0][0])


        bsz = src_tokens.size()[0]
        greedy_reward = self.cal_rl_reward_core(src_tokens, greedy_finalized_tokens[0])
        for _ in range(self.rl_k):
            sampling_reward = self.cal_rl_reward_core(src_tokens, sample_finalized_tokens[_])
            reward_diff = greedy_reward - sampling_reward
            sample_losses = torch.FloatTensor(sample_finalized_scores[_])
            rl_losses = torch.FloatTensor(sample_losses * reward_diff)
            _rl_losses.append(rl_losses)

        _rl_losses = torch.stack(_rl_losses, dim=0)
        rl_loss = _rl_losses.mean()
        return rl_loss
        #return 0

    def cal_rl_reward_core(self, inputs, sample_outputs):
        """
        arguments:
          input:bsz of tokens
          sample_output: bsz of tokens
        output:
          bsz of float
        """
        sources = []
        targets = []
        for _ in range(len(inputs)):
            input = inputs[_]
            output = sample_outputs[_]
            source = self.decode(input, self.src_dict)
            target = self.decode(output, self.tgt_dict)
            sources.append(source)
            targets.append(target)

        input_ids, segment_ids = self.evaluator_tokenizer.build_data(samples, targets)
        similarity_score, _ = self.evaluator_model.predict(input_ids, segment_ids)
        #print("similarity_score:",similarity_score)
        return torch.FloatTensor(similarity_score)

    def forward(self, model, sample, reduce=True):
        net_output = model(**sample['net_input'])
        sl_loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)

        rl_reward = self.rl_forward(model, sample)
        loss = (1-self.lambda_ratio) * sl_loss + self.lambda_ratio *(-rl_reward)
        sample_size = sample['target'].size(0) if self.sentence_avg else sample['ntokens']
        logging_output = {
            'loss': loss.data,
            'nll_loss': nll_loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
            'rl_reward': rl_reward,
            'sl_loss': sl_loss
            }
        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = model.get_targets(sample, net_output).view(-1, 1)
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs, target, self.eps, ignore_index=self.padding_idx, reduce=reduce,
            )
        return loss, nll_loss

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        nll_loss_sum = sum(log.get('nll_loss', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)

        sl_loss_sum = sum(log.get('sl_loss', 0) for log in logging_outputs)
        rl_reward_sum = sum(log.get('rl_reward', 0) for log in logging_outputs)

        metrics.log_scalar('loss', loss_sum / sample_size / math.log(2), sample_size, round=3)
        metrics.log_scalar('nll_loss', nll_loss_sum / ntokens / math.log(2), ntokens, round=3)
        metrics.log_derived('ppl', lambda meters: utils.get_perplexity(meters['nll_loss'].avg))

        metrics.log_scalar('sl_loss', sl_loss_sum / sample_size / math.log(2), sample_size, round=3)
        metrics.log_scalar('rl_reward', rl_reward_sum / sample_size / math.log(2), sample_size, round=3)

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True

    def finalize_hypos(
        self,
        step:int,
        bbsz_idx,
        eos_scores,
        tokens,
        scores,
        finalized: List[List[Dict[str, Tensor]]],
        finished: List[bool],
        beam_size:int,
        max_len:int
        ):
        assert bbsz_idx.numel() == eos_scores.numel() #[bsz, rl_k], [bsz, rl_k]

        #clone relevant token and attention tensors
        tokens_clone = tokens.index_select(0, bbsz_idx)[
            :, 1:step+2
            ]# skip the first index, which is EOS
        tokens_clone[:, step] = self.eos

        #if self.normalize_scores:
        eos_scores /= (step+1) ** self.len_penalty

        cum_unfin: List[int] = []
        prev = 0
        for f in finished:
            if f:
                prev += 1
            else:
                cum_unfin.append(prev)

        sents_seen:Dict[str, Optional[Tensor]] = {}
        for i in range(bbsz_idx.size()[0]):
            idx = bbsz_idx[i]
            score = eos_scores[i]
            unfin_idx = idx//beam_size
            sent = unfin_idx + cum_unfin[unfin_idx]

            seen = str(sent.item()) + "_" + str(unfin_idx.item())
            if seen not in sents_seen:
                sents_seen[seen] = None

            if len(finalized[sent]) < beam_size:
                finalized[sent].append(
                    {
                        "tokens": tokens_clone[i],
                        "score": score
                        }
                    )

        newly_finished: List[int] = []
        for seen in sents_seen.keys():
            sent: int = int(float(seen.split("_")[0]))
            unfin_idx: int = int(float(seen.split("_")[1]))
            if not finished[sent] and self.is_finished(
                step, unfin_idx, max_len, len(finalized[sent]), beam_size
            ):
                finished[sent] = True
                newly_finished.append(unfin_idx)
        return newly_finished

    def is_finished(
        self,
        step: int,
        unfin_idx: int,
        max_len: int,
        finalized_sent_len: int,
        beam_size: int,
    ):
        """
        Check whether we've finished generation for a given sentence, by
        comparing the worst score among finalized hypotheses to the best
        possible score among unfinalized hypotheses.
        """
        assert finalized_sent_len <= beam_size
        if finalized_sent_len == beam_size or step == max_len:
            return True
        return False

    def decode(self, tokens: torch.LongTensor, dict):
        assert tokens.dim() == 1
        tokens = tokens.cpu().numpy()
        if tokens[0] == self.src_dict.bos():
            tokens = tokens[1:]  # remove <s>
        eos_mask = (tokens == self.src_dict.eos())
        doc_mask = eos_mask[1:] & eos_mask[:-1]
        sentences = np.split(tokens, doc_mask.nonzero()[0] + 1)
        new_sentences = []
        for s in sentences:
            _s = dict.string(s, extra_symbols_to_ignore=[0, 1, 2, 3, 50262, 50263,50264,50265])
            #print(s, _s)
            _s = self.bpe.decode(_s)
            new_sentences.append(_s)
        sentences = new_sentences
        #sentences = [self.bpe.decode(self.src_dict.string(s)) for s in sentences]
        if len(sentences) == 1:
            return sentences[0]
        return sentences

class EnsembleModel(nn.Module):
    """A wrapper around an ensemble of models."""

    incremental_states: List[Dict[str, Dict[str, Optional[Tensor]]]]

    def __init__(self, models):
        super().__init__()
        self.models_size = len(models)
        # method '__len__' is not supported in ModuleList for torch script
        self.single_model = models[0]
        self.models = nn.ModuleList(models)

        self.incremental_states = torch.jit.annotate(
            List[Dict[str, Dict[str, Optional[Tensor]]]],
            [
                torch.jit.annotate(Dict[str, Dict[str, Optional[Tensor]]], {})
                for i in range(self.models_size)
            ],
        )
        self.has_incremental: bool = False
        if all(
            hasattr(m, "decoder") and isinstance(m.decoder, FairseqIncrementalDecoder)
            for m in models
        ):
            self.has_incremental = True

    def forward(self):
        pass

    def reset_incremental_state(self):
        if self.has_incremental_states():
            self.incremental_states = torch.jit.annotate(
                List[Dict[str, Dict[str, Optional[Tensor]]]],
                [
                    torch.jit.annotate(Dict[str, Dict[str, Optional[Tensor]]], {})
                    for i in range(self.models_size)
                ],
            )
        return

    def has_encoder(self):
        return hasattr(self.single_model, "encoder")

    def has_incremental_states(self):
        return self.has_incremental

    def max_decoder_positions(self):
        return min([m.max_decoder_positions() for m in self.models])

    @torch.jit.export
    def forward_encoder(self, src_tokens, src_lengths):
        if not self.has_encoder():
            return None
        return [
            model.encoder(src_tokens=src_tokens, src_lengths=src_lengths)
            for model in self.models
        ]

    @torch.jit.export
    def forward_decoder(
        self, tokens, encoder_outs: List[EncoderOut], temperature: float = 1.0
    ):
        log_probs = []
        avg_attn: Optional[Tensor] = None
        encoder_out: Optional[EncoderOut] = None
        for i, model in enumerate(self.models):
            if self.has_encoder():
                encoder_out = encoder_outs[i]
            # decode each model
            if self.has_incremental_states():
                decoder_out = model.decoder.forward(
                    tokens,
                    encoder_out=encoder_out,
                    incremental_state=self.incremental_states[i]
                )
            else:
                decoder_out = model.decoder.forward(tokens, encoder_out=encoder_out)

            attn: Optional[Tensor] = None
            decoder_len = len(decoder_out)
            if decoder_len > 1 and decoder_out[1] is not None:
                if isinstance(decoder_out[1], Tensor):
                    attn = decoder_out[1]
                else:
                    attn_holder = decoder_out[1]["attn"]
                    if isinstance(attn_holder, Tensor):
                        attn = attn_holder
                    elif attn_holder is not None:
                        attn = attn_holder[0]
                if attn is not None:
                    attn = attn[:, -1, :]

            decoder_out_tuple = (
                decoder_out[0][:, -1:, :].div_(temperature),
                None if decoder_len <= 1 else decoder_out[1],
            )

            probs = model.get_normalized_probs(
                decoder_out_tuple, log_probs=True, sample=None
            )
            probs = probs[:, -1, :]
            if self.models_size == 1:
                return probs, attn

            log_probs.append(probs)
            if attn is not None:
                if avg_attn is None:
                    avg_attn = attn
                else:
                    avg_attn.add_(attn)
        avg_probs = torch.logsumexp(torch.stack(log_probs, dim=0), dim=0) - math.log(
            self.models_size
        )
        if avg_attn is not None:
            avg_attn.div_(self.models_size)
        return avg_probs, avg_attn

    @torch.jit.export
    def reorder_encoder_out(self, encoder_outs: Optional[List[EncoderOut]], new_order):
        """
        Reorder encoder output according to *new_order*.
        Args:
            encoder_out: output from the ``forward()`` method
            new_order (LongTensor): desired order
        Returns:
            *encoder_out* rearranged according to *new_order*
        """
        new_outs: List[EncoderOut] = []
        if not self.has_encoder():
            return new_outs
        for i, model in enumerate(self.models):
            assert encoder_outs is not None
            new_outs.append(
                model.encoder.reorder_encoder_out(encoder_outs[i], new_order)
            )
        return new_outs

    @torch.jit.export
    def reorder_incremental_state(self, new_order):
        #print("new_order tensor content:", new_order)
        #print("incremental states:", self.incremental_states[0].keys())
        if not self.has_incremental_states():
            return
        for i, model in enumerate(self.models):
            model.decoder.reorder_incremental_state(
                self.incremental_states[i], new_order
            )


@torch.jit.script
class BeamContainer(object):
    def __init__(self, score: float, elem: Dict[str, Tensor]):
        self.score = score
        self.elem = elem

    def __lt__(self, other):
        # type: (BeamContainer) -> bool
        # this has to use old style type annotations
        # Match original behavior of sorted function when two scores are equal.
        return self.score <= other.score
