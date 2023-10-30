from typing import List, Optional, Tuple, Union
from tqdm import tqdm
import random

import deepsparse

from lm_eval import utils
from lm_eval.base import BaseLM


class DeepSparseLM(BaseLM):
    _DEFAULT_MAX_LENGTH = 2048

    def __init__(
        self,
        pretrained: str,
        tokenizer: Optional[str] = None,
        batch_size: Optional[Union[int, str]] = 1,
        max_gen_toks: Optional[int] = 256,
        max_length: Optional[int] = None,
        trust_remote_code: Optional[bool] = False,
    ):
        super().__init__()

        # Initialize new model and tokenizer instances
        self.model = deepsparse.Pipeline.create(
            task="text-generation",
            model_path=pretrained,
            sequence_length=max_length or self._DEFAULT_MAX_LENGTH,
            prompt_sequence_length=16,
            trust_remote_code=trust_remote_code,
            batch_size=batch_size,
        )
        self.tokenizer = tokenizer if tokenizer else self.model.tokenizer

        self.vocab_size = self.tokenizer.vocab_size

        self._batch_size = int(batch_size)
        self._max_length = max_length
        self._max_gen_toks = max_gen_toks

    @property
    def eot_token(self) -> str:
        return self.tokenizer.eos_token

    @property
    def eot_token_id(self) -> int:
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        if self._max_length:  # if max length manually set, return it
            return self._max_length
        # seqlen_config_attrs = ("n_positions", "max_position_embeddings", "n_ctx")
        # for attr in seqlen_config_attrs:
        #     if hasattr(self.model.config, attr):
        #         return getattr(self.model.config, attr)
        # if hasattr(self.tokenizer, "model_max_length"):
        #     if self.tokenizer.model_max_length == 1000000000000000019884624838656:
        #         return self._DEFAULT_MAX_LENGTH
        #     return self.tokenizer.model_max_length
        return self._DEFAULT_MAX_LENGTH

    @property
    def max_gen_toks(self):
        return self._max_gen_toks

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def device(self):
        return "cpu"

    def tok_encode(self, string: str):
        return self.tokenizer.encode(string, add_special_tokens=False)

    def tok_decode(self, tokens):
        return self.tokenizer.decode(tokens)

    def greedy_until(
        self, requests: List[Tuple[str, Union[List[str], str]]]
    ) -> List[str]:
        def _collate(x):
            tokens = self.tok_encode(x[0])
            return len(tokens), x[0]

        results = []
        reorder = utils.Reorderer(requests, _collate)

        # adaptive_batch_size = None
        # if self.batch_size == "auto":
        #     # using rolling window with maximum context
        #     print("Passed argument batch_size = auto. Detecting largest batch size")
        #     batch_size = self._detect_batch_size()
        #     print(f"Determined Largest batch size: {batch_size}")
        #     adaptive_batch_size = batch_size

        for chunk in utils.chunks(
            tqdm(reorder.get_reordered(), disable=False),
            self.batch_size,
        ):
            context = [c[0] for c in chunk]
            request_args = chunk[0][1]
            stop = request_args.get("until", None)
            stop_sequences = stop if isinstance(stop, list) else [stop]
            max_generation_length = request_args.get("max_length", None)

            assert (
                isinstance(max_generation_length, int) or max_generation_length is None
            )
            assert isinstance(stop_sequences, list) or stop_sequences is None

            # TODO: Find a better way to handle stop sequences for 0-shot.
            if stop_sequences is None:
                until = [self.eot_token]
            else:
                until = stop_sequences + [self.eot_token]

            if max_generation_length is None:
                max_tokens = self.max_gen_toks
            else:
                max_tokens = max_generation_length

            # token_context = self.tok_encode_batch(context)

            responses = self.model(
                sequences=context,
                max_new_tokens=max_tokens,
                stop=until,
                do_sample=False,
            )

            responses = responses if type(responses) is list else [responses]

            for response in responses:
                response = response.generations[0].text
                # Ensure the generated responses do not contain the stop sequences.
                for term in until:
                    response = response.split(term)[0]
                # partial caching
                self.cache_hook.add_partial("greedy_until", (context, until), response)
                results.append(response)

        return reorder.get_original(results)

    def loglikelihood(self, requests):
        loglikelihoods = []
        for context, continuation in requests:
            tokens = self.tokenizer.encode(context + continuation, add_special_tokens=False)
            context_tokens = self.tokenizer.encode(context, add_special_tokens=False)

            # Compute the score (logits) for the continuation tokens
            scores = self.model(
                sequences=[context],
                max_new_tokens=len(tokens) - len(context_tokens),
                output_scores=True,
                stop=None,
                do_sample=False,
            ).generations[0].score

            # Calculate log probabilities from logits
            log_probs = log_softmax(scores, axis=-1)

            # Sum the log probabilities for the continuation tokens
            continuation_log_probs = np.sum(
                [log_probs[i, token] for i, token in enumerate(tokens[len(context_tokens):], start=len(context_tokens))]
            )
            # Check if the continuation is greedy
            is_greedy = np.all(np.argmax(log_probs, axis=-1) == tokens[len(context_tokens):])

            loglikelihoods.append((continuation_log_probs, is_greedy))
        
        return loglikelihoods

    def loglikelihood_rolling(self, requests):
        rolling_loglikelihoods = []
        for string in requests:
            tokens = self.tokenizer.encode(string, add_special_tokens=False)
            log_prob_sum = 0
            is_greedy = True

            for i in range(1, len(tokens)):
                # Prepare the context and continuation
                context = self.tokenizer.decode(tokens[:i])
                continuation = self.tokenizer.decode(tokens[i:i+1])

                # Compute loglikelihood for each token
                scores = self.model(
                    sequences=[context],
                    max_new_tokens=1,
                    output_scores=True,
                    stop=None,
                    do_sample=False,
                ).generations[0].score

                log_probs = log_softmax(scores, axis=-1)
                log_prob_sum += log_probs[0, tokens[i]]
                is_greedy = is_greedy and (np.argmax(log_probs, axis=-1)[0] == tokens[i])

            rolling_loglikelihoods.append((log_prob_sum, is_greedy))
        
        return rolling_loglikelihoods

    def _loglikelihood_tokens(self, requests, disable_tqdm=False):
        raise NotImplementedError("No support for logits.")

    def _model_call(self, inps):
        # Isn't used because we override _loglikelihood_tokens
        raise NotImplementedError()

    def _model_generate(self, context, max_length, eos_token_id):
        # Isn't used because we override greedy_until
        raise NotImplementedError()
