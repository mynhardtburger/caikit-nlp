# Copyright The Caikit Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# Standard
from functools import cached_property
from typing import Iterable, List, Optional, Tuple, Union
import os

# Third Party
import numpy as np

# First Party
from caikit.core.exceptions import error_handler
from caikit.core.module_backends import BackendBase, backend_types
from caikit.core.modules import ModuleBase, ModuleConfig, ModuleSaver, module
from caikit.interfaces.nlp.data_model import (
    GeneratedTextResult,
    GeneratedTextStreamResult,
    TokenizationResults,
)
from caikit.interfaces.nlp.tasks import TextGenerationTask, TokenizationTask
from caikit.interfaces.runtime.data_model import RuntimeServerContextType
from caikit_tgis_backend import TGISBackend
import alog

# Local
from ...data_model import ExponentialDecayLengthPenalty
from ...resources.pretrained_model import (
    HFAutoCausalLM,
    HFAutoSeq2SeqLM,
    PretrainedModelBase,
)
from ...toolkit.text_generation.tgis_utils import (
    GENERATE_FUNCTION_TGIS_ARGS,
    TGISGenerationClient,
)
from .text_generation_local import TextGeneration

log = alog.use_channel("TXT_GEN")
error = error_handler.get(log)

# pylint: disable=too-many-instance-attributes


@module(
    backend_type=TGISBackend.backend_type,
    base_module=TextGeneration,
    tasks=[TextGenerationTask, TokenizationTask],
)
class TextGenerationTGIS(ModuleBase):
    """Module to provide text generation capabilities"""

    SUPPORTED_LOAD_BACKENDS = [TGISBackend.backend_type, backend_types.LOCAL]

    supported_resources = [HFAutoCausalLM, HFAutoSeq2SeqLM]

    def __init__(
        self,
        model_name: str,
        model: Optional[PretrainedModelBase] = None,
        bos_token: Optional[str] = None,
        sep_token: Optional[str] = None,
        eos_token: Optional[str] = None,
        pad_token: Optional[str] = None,
        tgis_backend: Optional[TGISBackend] = None,
    ):
        super().__init__()

        error.type_check("<NLP00609194E>", str, allow_none=True, bos_token=bos_token)
        error.type_check("<NLP72469403E>", str, allow_none=True, sep_token=sep_token)
        error.type_check("<NLP48137045E>", str, allow_none=True, eos_token=eos_token)
        error.type_check("<NLP53511308E>", str, allow_none=True, pad_token=pad_token)
        self.model = model
        self.model_name = model_name

        # Set _model_loaded as False by default. This will only get set to True if
        # we enable the tgis_backend and we are able to fetch the client successfully.
        self._model_loaded = False
        if tgis_backend:
            self.tgis_backend = tgis_backend

        self._tgis_backend = tgis_backend
        self._bos_token = bos_token
        self._sep_token = sep_token
        self._eos_token = eos_token
        self._pad_token = pad_token

    def __del__(self):
        # nothing to unload if we didn't finish loading
        if self._model_loaded and self._tgis_backend:
            self._tgis_backend.unload_model(self.model_name)

    @cached_property
    def _client(self):
        # Lazily configure/create the internal tgis backend client
        if self._tgis_backend:
            return self._tgis_backend.get_client(self.model_name)

    @cached_property
    def tgis_generation_client(self):
        # Lazily create the generation client
        # This in turn calls self._client which also lazily gets the tgis backend client
        return TGISGenerationClient(
            self.model_name, self._eos_token, self._client, self.PRODUCER_ID
        )

    @classmethod
    def bootstrap(cls, model_path: str, load_backend: Union[BackendBase, None] = None):
        """Function to bootstrap a pre-trained transformers model and
        get a caikit text-generation 'model'.

        Args:
            base_model_path: str
                Path to transformers model
                NOTE: Model path needs to contain tokenizer as well
            load_backend: BackendBase
                Backend object to be used to run inference with.
                NOTE: this is required for inferencing. It is
                made optional to support the model conversion use-case
        Returns:
            caikit_nlp.blocks.text_generation.TextGeneration
                Object of TextGeneration class (model)
        """

        text_generation_inst = TextGeneration.bootstrap(model_path)
        bos_token = text_generation_inst.model._tokenizer.bos_token
        sep_token = text_generation_inst.model._tokenizer.sep_token
        eos_token = text_generation_inst.model._tokenizer.eos_token or None
        pad_token = text_generation_inst.model._tokenizer.pad_token

        return cls(
            text_generation_inst.model_name,
            text_generation_inst.model,
            bos_token=bos_token,
            sep_token=sep_token,
            eos_token=eos_token,
            pad_token=pad_token,
            tgis_backend=load_backend,
        )

    @classmethod
    def load(cls, model_path: str, load_backend: BackendBase) -> "TextGeneration":
        """Function to load text-generation model. Note, this only loads
        "remote" style model, i.e the cakit-model that doesn't
        necessarily required to have actual artifacts in it
        and thus only saves them in "remote" format.

        Args:
            model_path: str
                Path to the model to be loaded.
            load_backend: BackendBase
                Backend object to be used to run inference with.
        Returns:
            TextGeneration
                Instance of this class built from the on disk model.
        """
        error.type_check("<NLP03521359E>", TGISBackend, load_backend=load_backend)

        config = ModuleConfig.load(model_path)
        tgis_backend = config.tgis_backend or load_backend
        artifacts_path = config.artifact_path
        if artifacts_path:
            model_name = os.path.join(model_path, artifacts_path)
            error.dir_check("<NLP01983374E>", model_name)
            log.debug("Loading with on-disk artifacts: %s", model_name)
        else:
            model_name = config.model_name
            error.type_check("<NLP90686335E>", str, model_name=model_name)
            log.debug("Loading with model name: %s", model_name)
        return cls(
            model_name,
            bos_token=config.bos_token,
            sep_token=config.sep_token,
            eos_token=config.eos_token,
            pad_token=config.pad_token,
            tgis_backend=tgis_backend,
        )

    def save(self, model_path: str):
        """Export the config for this model.
        This saves the model in "remote" style
        and does not store the actual model artifacts
        along with the caikit-model.

        model_path: str
            Path to which we should write our model.
        """
        # pylint: disable=duplicate-code
        saver = ModuleSaver(
            self,
            model_path=model_path,
        )
        with saver:
            saver.update_config(
                {
                    "model_name": self.model_name,
                    "bos_token": self._bos_token,
                    "sep_token": self._sep_token,
                    "eos_token": self._eos_token,
                    "pad_token": self._pad_token,
                }
            )

    # pylint: disable=duplicate-code
    @TextGenerationTask.taskmethod(context_arg="context")
    def run(
        self,
        text: str,
        max_new_tokens: Optional[int] = 20,
        min_new_tokens: Optional[int] = 0,
        truncate_input_tokens: Optional[int] = 0,
        decoding_method: Optional[str] = "GREEDY",
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        typical_p: Optional[float] = None,
        temperature: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        max_time: Optional[float] = None,
        exponential_decay_length_penalty: Optional[
            Union[Tuple[int, float], ExponentialDecayLengthPenalty]
        ] = None,
        stop_sequences: Optional[List[str]] = None,
        seed: Optional[np.uint64] = None,
        preserve_input_text: bool = False,
        input_tokens: bool = False,
        generated_tokens: bool = True,
        token_logprobs: bool = True,
        token_ranks: bool = True,
        include_stop_sequence: Optional[bool] = True,
        context: Optional[RuntimeServerContextType] = None,
    ) -> GeneratedTextResult:
        f"""Run inference against the model running in TGIS.

        Args:
           {GENERATE_FUNCTION_TGIS_ARGS}
        Returns:
            GeneratedTextResult
                Generated text result produced by TGIS.
        """
        self._register_model_connection_with_context(context)

        if self._model_loaded:
            return self.tgis_generation_client.unary_generate(
                text=text,
                preserve_input_text=preserve_input_text,
                input_tokens=input_tokens,
                generated_tokens=generated_tokens,
                token_logprobs=token_logprobs,
                token_ranks=token_ranks,
                include_stop_sequence=include_stop_sequence,
                max_new_tokens=max_new_tokens,
                min_new_tokens=min_new_tokens,
                truncate_input_tokens=truncate_input_tokens,
                decoding_method=decoding_method,
                top_k=top_k,
                top_p=top_p,
                typical_p=typical_p,
                temperature=temperature,
                seed=seed,
                repetition_penalty=repetition_penalty,
                max_time=max_time,
                exponential_decay_length_penalty=exponential_decay_length_penalty,
                stop_sequences=stop_sequences,
            )

    @TextGenerationTask.taskmethod(output_streaming=True, context_arg="context")
    def run_stream_out(
        self,
        text: str,
        max_new_tokens: Optional[int] = 20,
        min_new_tokens: Optional[int] = 0,
        truncate_input_tokens: Optional[int] = 0,
        decoding_method: Optional[str] = "GREEDY",
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        typical_p: Optional[float] = None,
        temperature: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        max_time: Optional[float] = None,
        exponential_decay_length_penalty: Optional[
            Union[Tuple[int, float], ExponentialDecayLengthPenalty]
        ] = None,
        stop_sequences: Optional[List[str]] = None,
        seed: Optional[np.uint64] = None,
        preserve_input_text: bool = False,
        input_tokens: bool = False,
        generated_tokens: bool = True,
        token_logprobs: bool = True,
        token_ranks: bool = True,
        include_stop_sequence: Optional[bool] = True,
        context: Optional[RuntimeServerContextType] = None,
    ) -> Iterable[GeneratedTextStreamResult]:
        f"""Run output stream inferencing for text generation module.

        Args:
            {GENERATE_FUNCTION_TGIS_ARGS}
        Returns:
            Iterable[GeneratedTextStreamResult]
        """
        self._register_model_connection_with_context(context)

        if self._model_loaded:
            return self.tgis_generation_client.stream_generate(
                text=text,
                preserve_input_text=preserve_input_text,
                input_tokens=input_tokens,
                generated_tokens=generated_tokens,
                token_logprobs=token_logprobs,
                token_ranks=token_ranks,
                include_stop_sequence=include_stop_sequence,
                max_new_tokens=max_new_tokens,
                min_new_tokens=min_new_tokens,
                truncate_input_tokens=truncate_input_tokens,
                decoding_method=decoding_method,
                top_k=top_k,
                top_p=top_p,
                typical_p=typical_p,
                temperature=temperature,
                seed=seed,
                repetition_penalty=repetition_penalty,
                max_time=max_time,
                exponential_decay_length_penalty=exponential_decay_length_penalty,
                stop_sequences=stop_sequences,
            )

    @TokenizationTask.taskmethod(context_arg="context")
    def run_tokenizer(
        self,
        text: str,
        context: Optional[RuntimeServerContextType] = None,
    ) -> TokenizationResults:
        """Run tokenization task against the model running in TGIS.

        Args:
           text: str
                Text to tokenize
        Returns:
            TokenizationResults
                The token count
        """
        self._register_model_connection_with_context(context)

        if self._model_loaded:
            return self.tgis_generation_client.unary_tokenize(
                text=text,
            )

    def _register_model_connection_with_context(
        self, context: Optional[RuntimeServerContextType]
    ):
        """
        Register a remote model connection with the configured TGISBackend if there is
        a context override provided.
        """
        if self._tgis_backend:
            self._tgis_backend.handle_runtime_context(self.model_name, context)
            self._model_loaded = True
