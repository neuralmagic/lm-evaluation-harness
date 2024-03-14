from lm_eval.models.huggingface import AutoCausalLM
from typing import Optional, Union
import os
import torch
import transformers

class AutoGPTQLM(AutoCausalLM):
    def _create_auto_model(
        self,
        *,
        pretrained: str,
        trust_remote_code: Optional[bool] = False,
        torch_dtype: Optional[Union[str, torch.dtype]] = None,
        **kwargs,
    ) -> transformers.AutoModel:

        try:
            from auto_gptq import AutoGPTQForCausalLM
        except ModuleNotFoundError:
            raise Exception(
                "package `auto_gptq` is not installed. "
                "Please install it via `pip install auto-gptq`"
            )

        
        model = AutoGPTQForCausalLM.from_quantized(pretrained,
                                                   trust_remote_code=trust_remote_code,
                                                   torch_dtype=torch_dtype)
        return model
