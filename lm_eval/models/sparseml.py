from lm_eval.models.huggingface import AutoCausalLM
from typing import Optional, Union
import torch
import transformers
from sparseml.transformers.utils.sparse_model import SparseAutoModel


class SparseML(AutoCausalLM):
    def _create_auto_model(
        self,
        *,
        pretrained: str,
        trust_remote_code: Optional[bool] = False,
        torch_dtype: Optional[Union[str, torch.dtype]] = None,
        **kwargs,
    ) -> transformers.AutoModel:

        recipe_file = os.path.join(pretrained, "recipe.yaml")

        model = SparseAutoModel.text_generation_from_pretrained(
            model_name_or_path=pretrained,
            config=self._config,
            recipe=recipe_file,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch_dtype,
        )
        return model
