from lm_eval.models.huggingface import AutoCausalLM
from typing import Optional, Union
import os
import torch
import transformers


class SparseML(AutoCausalLM):
    def _create_auto_model(
        self,
        *,
        pretrained: str,
        trust_remote_code: Optional[bool] = False,
        torch_dtype: Optional[Union[str, torch.dtype]] = None,
        **kwargs,
    ) -> transformers.AutoModel:

        try:
            import sparseml
        except ModuleNotFoundError:
            raise Exception(
                "package `sparseml` is not installed. "
                "Please install it via `pip install sparseml[transformers]`"
            )

        recipe_file = os.path.join(pretrained, "recipe.yaml")
        if not os.path.isfile(recipe_file):
            recipe_file = None

        from sparseml.transformers import SparseAutoModel
        model = SparseAutoModel.text_generation_from_pretrained(
            model_name_or_path=pretrained,
            config=self._config,
            recipe=recipe_file,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch_dtype,
        )

        return model