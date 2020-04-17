from fairseq.tasks import FairseqTask, register_task
from fairseq.tasks.translation import TranslationTask
from fairseq.criterions.label_smoothed_cross_entropy_with_RL import LabelSmoothedCrossEntropyWithRLCriterion


@register_task('translation_rl')
class TranslationWithRLTask(TranslationTask):
    def build_criterion(self, args):
        #only criterion label_smoothed_cross_entory_with_RL is supported in this task

        """
        Build the :class:`~fairseq.criterions.FairseqCriterion` instance for
        this task.
        Args:
            args (argparse.Namespace): parsed command-line arguments
        Returns:
            a :class:`~fairseq.criterions.FairseqCriterion` instance
        """
        return LabelSmoothedCrossEntropyWithRLCriterion(task=args.task, 
                                                        sentence_avg=args.sentence_avg,
                                                        label_smoothing=args.label_smoothing,
                                                        lambda_ratio=args.lambda_ratio,
                                                        rl_k=args.rl_k,
                                                        max_decoding_len=args.max_decoding_len,
                                                        tgt_dict=self.target_dictionary)