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
from typing import (
    TYPE_CHECKING,
    Callable,
    List,
    Literal,
    Optional,
    Tuple,
    TypeVar,
    Union,
    overload,
)
import importlib
import os
import time

# Third Party
from torch.backends import mps
import numpy as np
import torch

# First Party
from caikit import get_config
from caikit.core import ModuleBase, ModuleConfig, ModuleSaver, module
from caikit.core.data_model.json_dict import JsonDict
from caikit.core.exceptions import error_handler
from caikit.interfaces.common.data_model.vectors import ListOfVector1D, Vector1D
from caikit.interfaces.nlp.data_model import (
    EmbeddingResult,
    EmbeddingResults,
    RerankResult,
    RerankResults,
    RerankScore,
    RerankScores,
    SentenceSimilarityResult,
    SentenceSimilarityResults,
    SentenceSimilarityScores,
)
from caikit.interfaces.nlp.tasks import (
    EmbeddingTask,
    EmbeddingTasks,
    RerankTask,
    RerankTasks,
    SentenceSimilarityTask,
    SentenceSimilarityTasks,
)
import alog

# Local
from caikit_nlp.modules.text_embedding.utils import env_val_to_bool, env_val_to_int

if TYPE_CHECKING:
    # Third Party
    from sentence_transformers import SentenceTransformer

logger = alog.use_channel("TXT_EMB")
error = error_handler.get(logger)

embedding_cfg = get_config().get("embedding", {})

AUTOCAST = env_val_to_bool(val=embedding_cfg.get("autocast"))
IPEX = env_val_to_bool(val=embedding_cfg.get("ipex"))
PT2_COMPILE = env_val_to_bool(val=embedding_cfg.get("pt2_compile"))
RETRIES = env_val_to_int(val=embedding_cfg.get("retries"), default=0)
BATCH_SIZE = env_val_to_int(val=embedding_cfg.get("batch_size"), default=0)
DEVICE = embedding_cfg.get("device", "")

RT = TypeVar("RT")  # return type
EncodingResult = Union[List[torch.Tensor], np.ndarray, torch.Tensor]


@module(
    "eeb12558-b4fa-4f34-a9fd-3f5890e9cd3f",
    "EmbeddingModule",
    "0.0.1",
    tasks=[
        EmbeddingTask,
        EmbeddingTasks,
        SentenceSimilarityTask,
        SentenceSimilarityTasks,
        RerankTask,
        RerankTasks,
    ],
)
class EmbeddingModule(ModuleBase):
    # Retry count if enabled to try again (was for thread contention errors)
    RETRY_COUNT = max(RETRIES, 0)  # Ensure non-negative, before using in loop!

    _ARTIFACTS_PATH_KEY = "artifacts_path"
    _ARTIFACTS_PATH_DEFAULT = "artifacts"

    def __init__(
        self,
        model: "SentenceTransformer",
    ):
        super().__init__()
        self.model = model

    @classmethod
    def load(cls, model_path: str, *args, **kwargs) -> "EmbeddingModule":
        """Load model

        Args:
            model_path: str
                Path to the config dir under the model_id (where the config.yml lives)

        Returns:
            EmbeddingModule
                Instance of this class built from the model.
        """
        # Local
        from caikit_nlp.modules.text_embedding.sentence_transformer_truncate import (
            SentenceTransformerWithTruncate,
        )

        config = ModuleConfig.load(model_path)
        artifacts_path = config.get(cls._ARTIFACTS_PATH_KEY)

        error.value_check(
            "<NLP07391618E>",
            artifacts_path,
            ValueError(f"Model config missing '{cls._ARTIFACTS_PATH_KEY}'"),
        )

        artifacts_path = os.path.abspath(os.path.join(model_path, artifacts_path))
        error.dir_check("<NLP34197772E>", artifacts_path)

        ipex = cls._get_ipex(IPEX)
        device = cls._select_device(ipex, DEVICE)

        model = SentenceTransformerWithTruncate(
            model_name_or_path=artifacts_path, device=device
        )
        model.eval()  # required for IPEX at least
        if device is not None:
            model.to(torch.device(device))

        optimized_model: SentenceTransformerWithTruncate = EmbeddingModule._optimize(
            model, ipex, device, AUTOCAST, PT2_COMPILE
        )

        # Validate model with any encode test (simple and hardcoded for now).
        # This gets some of the first-time inference cost out of the way.
        # This avoids using the tokenizer (for truncation) before it is ready.
        optimized_model.encode_with_truncate("warmup")

        return cls(optimized_model)

    @classmethod
    def _get_ipex(cls, ipex_flag):
        """Get IPEX optimization library if enabled and available, else return False

        Returns ipex library or False
        """
        ret = False

        # Enabled by environment variable
        # When IPEX is not false, attempt to import the library and use it.
        if ipex_flag:
            try:
                ret = importlib.import_module("intel_extension_for_pytorch")
            except Exception as ie:  # pylint: disable=broad-exception-caught
                # We don't require the module so catch, log, proceed to return False
                msg = (
                    f"IPEX enabled in env, but skipping ipex.optimize() because "
                    f"import intel_extension_for_pytorch failed with exception: {ie}"
                )
                logger.warning(msg, exc_info=True)

        return ret

    @staticmethod
    def _select_device(use_ipex, device):
        """Use environment variables and availability to determine the device to use"""
        if use_ipex:
            # If enabled, use "xpu" (IPEX on GPU instead of IPEX on CPU)
            if device == "xpu":
                return "xpu"
        elif device == "mps" and mps.is_built() and mps.is_available():
            # Never use on ipex, but otherwise use mps if enabled and available
            return "mps"

        return "cuda" if torch.cuda.is_available() else None

    @staticmethod
    def _get_backend(use_ipex, use_device):
        """Determine the backend to use for torch compile.

        Considers global ipex if enabled first, next mps device, finally defaults.

        Returns the backend for torch.compile()
        """
        if use_ipex:
            return "ipex"
        if use_device == "mps":
            return mps
        return "inductor"  # default backend

    @staticmethod
    def _optimize(model, ipex, device, autocast, pt2_compile):
        if ipex:
            if autocast:  # IPEX performs best with autocast using bfloat16
                model = ipex.optimize(
                    model, dtype=torch.bfloat16, weights_prepack=False
                )
            else:
                model = ipex.optimize(model, weights_prepack=False)

        # torch.compile won't work everywhere, but when set we'll try it
        if pt2_compile:
            backend = EmbeddingModule._get_backend(ipex, device)
            try:
                model = torch.compile(model, backend=backend, mode="max-autotune")
            except Exception as e:  # pylint: disable=broad-exception-caught
                # Not always supported (e.g. in a python version) so catch, log, proceed.
                warn_msg = (
                    f"PT2_COMPILE enabled, but continuing without torch.compile() "
                    f"because it failed with exception: {e}"
                )
                logger.warning(warn_msg, exc_info=True)
        return model

    def _with_retry(self, fn: Callable[..., RT], *args, **kwargs) -> RT:
        first_exception = None
        for count in range(1 + self.RETRY_COUNT):  # try once plus retries (if needed)
            try:
                return fn(*args, **kwargs)
            except Exception as e:  # pylint: disable=broad-exception-caught
                if first_exception is None:
                    first_exception = e
                if self.RETRY_COUNT > 0:
                    warn_msg = f"Try {count + 1}: {fn} failed due to: {e}"
                    logger.warning("<NLP54902271W>", warn_msg, exc_info=True)
                    if count + 1 < self.RETRY_COUNT:
                        time.sleep(0.1 * (count * 2))

        # If above return did not happen, raise the first exception
        error.log_raise(
            log_code="<NLP13096081E>",
            exception=first_exception,
        )

    @overload
    def _encode_with_retry(
        self, *args, return_token_count: Literal[True], **kwargs
    ) -> Tuple[EncodingResult, int]:
        ...

    @overload
    def _encode_with_retry(
        self, *args, return_token_count: Literal[False] = False, **kwargs
    ) -> EncodingResult:
        ...

    def _encode_with_retry(
        self, *args, **kwargs
    ) -> Union[Tuple[EncodingResult, int], EncodingResult]:
        """All encode calls should use this for consistent param adding and retry loop"""

        # Add the batch_size kwarg if not passed in and given a usable BATCH_SIZE
        if BATCH_SIZE > 0:
            if kwargs is None:
                kwargs = {}
            if "batch_size" not in kwargs:
                kwargs["batch_size"] = BATCH_SIZE

        # Local
        from caikit_nlp.modules.text_embedding.sentence_transformer_truncate import (
            SentenceTransformerWithTruncate,
        )

        if isinstance(self.model, SentenceTransformerWithTruncate):
            return self._with_retry(self.model.encode_with_truncate, *args, **kwargs)

        # Else...
        # It's possible to init with a model that doesn't have the added kwargs.
        # E.g. a SentenceTransformer or other transformer model. Remove those kwargs!
        # This is not the normal use case but at least don't pass invalid kwargs, to encode()
        # and don't return the unexpected tuple (adding token count).
        del kwargs["truncate_input_tokens"]
        del kwargs["return_token_count"]
        return self._with_retry(self.model.encode, *args, **kwargs)

    @EmbeddingTask.taskmethod()
    def run_embedding(
        self,
        text: str,
        truncate_input_tokens: Optional[int] = 0,
    ) -> EmbeddingResult:
        """Get embedding for a string.
        Args:
            text: str
                Input text to be processed
            truncate_input_tokens: int
                Truncation length for input tokens.
                If less than zero, this is disabled (returns texts without processing).
                If zero or greater than the model's maximum, then this is a test
                to see if truncation is needed. If needed, an exception is thrown.
                Otherwise, we take this usable truncation limit to truncate the tokens and
                decode them to return truncated strings that can be used with this model.
        Returns:
            EmbeddingResult: the result vector nicely wrapped up
        """
        error.type_check("<NLP27491611E>", str, text=text)

        embeddings, input_token_count = self._encode_with_retry(
            text,
            truncate_input_tokens=truncate_input_tokens,
            return_token_count=True,
        )
        return EmbeddingResult(
            result=Vector1D.from_vector(embeddings),
            producer_id=self.PRODUCER_ID,
            input_token_count=input_token_count,
        )

    @EmbeddingTasks.taskmethod()
    def run_embeddings(
        self,
        texts: List[str],
        truncate_input_tokens: Optional[int] = 0,
    ) -> EmbeddingResults:
        """Get embedding vectors for texts.
        Args:
            texts: List[str]
                List of input texts to be processed
            truncate_input_tokens: int
                Truncation length for input tokens.
                If less than zero, this is disabled (returns texts without processing).
                If zero or greater than the model's maximum, then this is a test
                to see if truncation is needed. If needed, an exception is thrown.
                Otherwise, we take this usable truncation limit to truncate the tokens and then
                decode them to return truncated strings that can be used with this model.
        Returns:
            EmbeddingResults: List of vectors. One for each input text (in order).
             Each vector is a list of floats (supports various float types).
        """
        if isinstance(
            texts, str
        ):  # encode allows str, but the result would lack a dimension
            texts = [texts]

        embeddings, input_token_count = self._encode_with_retry(
            texts,
            truncate_input_tokens=truncate_input_tokens,
            return_token_count=True,
        )
        vectors = [Vector1D.from_vector(e) for e in embeddings]

        return EmbeddingResults(
            results=ListOfVector1D(vectors=vectors),
            producer_id=self.PRODUCER_ID,
            input_token_count=input_token_count,
        )

    @SentenceSimilarityTask.taskmethod()
    def run_sentence_similarity(
        self,
        source_sentence: str,
        sentences: List[str],
        truncate_input_tokens: Optional[int] = 0,
    ) -> SentenceSimilarityResult:
        """Get similarity scores for each of sentences compared to the source_sentence.
        Args:
            source_sentence: str
            sentences: List[str]
                Sentences to compare to source_sentence
            truncate_input_tokens: int
                Truncation length for input tokens.
                If less than zero, this is disabled (returns texts without processing).
                If zero or greater than the model's maximum, then this is a test
                to see if truncation is needed. If needed, an exception is thrown.
                Otherwise, we take this usable truncation limit to truncate the tokens and then
                decode them to return truncated strings that can be used with this model.
        Returns:
            SentenceSimilarityResult: Similarity scores for each sentence.
        """
        # Third Party
        from sentence_transformers.util import cos_sim

        source_embedding, source_token_count = self._encode_with_retry(
            source_sentence,
            truncate_input_tokens=truncate_input_tokens,
            return_token_count=True,
        )
        embeddings, sentences_token_count = self._encode_with_retry(
            sentences,
            truncate_input_tokens=truncate_input_tokens,
            return_token_count=True,
        )

        input_token_count = source_token_count + sentences_token_count
        res = cos_sim(source_embedding, embeddings)

        return SentenceSimilarityResult(
            result=SentenceSimilarityScores(scores=res.tolist()[0]),
            producer_id=self.PRODUCER_ID,
            input_token_count=input_token_count,
        )

    @SentenceSimilarityTasks.taskmethod()
    def run_sentence_similarities(
        self,
        source_sentences: List[str],
        sentences: List[str],
        truncate_input_tokens: Optional[int] = 0,
    ) -> SentenceSimilarityResults:
        """Run sentence-similarities on model.
        Args:
            source_sentences: List[str]
            sentences: List[str]
                Sentences to compare to source_sentences
            truncate_input_tokens: int
                Truncation length for input tokens.
                If less than zero, this is disabled (returns texts without processing).
                If zero or greater than the model's maximum, then this is a test
                to see if truncation is needed. If needed, an exception is thrown.
                Otherwise, we take this usable truncation limit to truncate the tokens and then
                decode them to return truncated strings that can be used with this model.
        Returns:
            SentenceSimilarityResults: Similarity scores for each source sentence in order.
                Each one contains the source-sentence's score for each sentence in order.
        """
        # Third Party
        from sentence_transformers.util import cos_sim

        source_embedding, source_token_count = self._encode_with_retry(
            source_sentences,
            truncate_input_tokens=truncate_input_tokens,
            return_token_count=True,
        )
        embeddings, sentences_token_count = self._encode_with_retry(
            sentences,
            truncate_input_tokens=truncate_input_tokens,
            return_token_count=True,
        )

        input_token_count = source_token_count + sentences_token_count
        res = cos_sim(source_embedding, embeddings)
        float_list_list = res.tolist()

        return SentenceSimilarityResults(
            results=[SentenceSimilarityScores(fl) for fl in float_list_list],
            producer_id=self.PRODUCER_ID,
            input_token_count=input_token_count,
        )

    @RerankTask.taskmethod()
    def run_rerank_query(
        self,
        query: str,
        documents: List[JsonDict],
        top_n: Optional[int] = None,
        truncate_input_tokens: Optional[int] = 0,
        return_documents: bool = True,
        return_query: bool = True,
        return_text: bool = True,
    ) -> RerankResult:
        """Rerank the documents returning the most relevant top_n in order for this query.
        Args:
            query: str
                Query is the source string to be compared to the text of the documents.
            documents:  List[JsonDict]
                Each document is a dict. The text value is used for comparison to the query.
                If there is no text key, then _text is used and finally default is "".
            top_n:  Optional[int]
                Results for the top n most relevant documents will be returned.
                If top_n is not provided or (not > 0), then all are returned.
            truncate_input_tokens: int
                Truncation length for input tokens.
                If less than zero, this is disabled (returns texts without processing).
                If zero or greater than the model's maximum, then this is a test
                to see if truncation is needed. If needed, an exception is thrown.
                Otherwise, we take this usable truncation limit to truncate the tokens and then
                decode them to return truncated strings that can be used with this model.
            return_documents:  bool
                Default True
                Setting to False will disable returning of the input document (index is returned).
            return_query:  bool
                Default True
                Setting to False will disable returning of the query (results are in query order)
            return_text:  bool
                Default True
                Setting to False will disable returning of document text string that was used.
        Returns:
            RerankResult
                Returns the (top_n) scores in relevance order (most relevant first).
                The results always include a score and index which may be used to find the document
                in the original documents list. Optionally, the results also contain the entire
                document with its score (for use in chaining) and for convenience the query and
                text used for comparison may be returned.

        """

        error.type_check(
            "<NLP05323654E>",
            str,
            query=query,
        )

        results = self.run_rerank_queries(
            queries=[query],
            documents=documents,
            top_n=top_n,
            truncate_input_tokens=truncate_input_tokens,
            return_documents=return_documents,
            return_queries=return_query,
            return_text=return_text,
        )

        if results.results:
            return RerankResult(
                result=results.results[0],
                producer_id=self.PRODUCER_ID,
                input_token_count=results.input_token_count,
            )

        RerankResult(
            result=RerankScore(
                scores=[],
                query=query if return_query else None,
            ),
            producer_id=self.PRODUCER_ID,
            input_token_count=results.input_token_count,
        )

    @RerankTasks.taskmethod()
    def run_rerank_queries(
        self,
        queries: List[str],
        documents: List[JsonDict],
        top_n: Optional[int] = None,
        truncate_input_tokens: Optional[int] = 0,
        return_documents: bool = True,
        return_queries: bool = True,
        return_text: bool = True,
    ) -> RerankResults:
        """Rerank the documents returning the most relevant top_n in order for each of the queries.
        Args:
            queries: List[str]
                Each of the queries will be compared to the text of each of the documents.
            documents:  List[JsonDict]
                Each document is a dict. The text value is used for comparison to the query.
                If there is no text key, then _text is used and finally default is "".
            top_n:  Optional[int]
                Results for the top n most relevant documents will be returned.
                If top_n is not provided or (not > 0), then all are returned.
            truncate_input_tokens: int
                Truncation length for input tokens.
                If less than zero, this is disabled (returns texts without processing).
                If zero or greater than the model's maximum, then this is a test
                to see if truncation is needed. If needed, an exception is thrown.
                Otherwise, we take this usable truncation limit to truncate the tokens and then
                decode them to return truncated strings that can be used with this model.
            return_documents:  bool
                Default True
                Setting to False will disable returning of the input document (index is returned).
            return_queries:  bool
                Default True
                Setting to False will disable returning of the query (results are in query order)
            return_text:  bool
                Default True
                Setting to False will disable returning of document text string that was used.
        Returns:
            RerankResults
                For each query in queries (in the original order)...
                Returns the (top_n) scores in relevance order (most relevant first).
                The results always include a score and index which may be used to find the document
                in the original documents list. Optionally, the results also contain the entire
                document with its score (for use in chaining) and for convenience the query and
                text used for comparison may be returned.
        """
        # Third Party
        from sentence_transformers.util import dot_score
        from sentence_transformers.util import (
            normalize_embeddings as normalize,  # avoid parameter shadowing
        )
        from sentence_transformers.util import semantic_search

        error.type_check(
            "<NLP09038249E>",
            list,
            queries=queries,
            documents=documents,
        )

        error.value_check(
            "<NLP24788937E>",
            queries and documents,
            "Cannot rerank without a query and at least one document",
        )

        if top_n is None or top_n < 1:
            top_n = len(documents)

        # Using input document dicts so get "text" else "_text" else default to ""
        def get_text(doc):
            return doc.get("text") or doc.get("_text", "")

        doc_texts = [get_text(doc) for doc in documents]

        doc_embeddings, doc_token_count = self._encode_with_retry(
            doc_texts,
            truncate_input_tokens=truncate_input_tokens,
            return_token_count=True,
            convert_to_tensor=True,
        )
        doc_embeddings = normalize(doc_embeddings.to(self.model.device))

        query_embeddings, query_token_count = self._encode_with_retry(
            queries,
            truncate_input_tokens=truncate_input_tokens,
            return_token_count=True,
            convert_to_tensor=True,
        )
        query_embeddings = normalize(query_embeddings.to(self.model.device))

        res = semantic_search(
            query_embeddings, doc_embeddings, top_k=top_n, score_function=dot_score
        )

        # Fixup result dicts
        for r in res:
            for x in r:
                # Renaming corpus_id to index
                corpus_id = x.pop("corpus_id")
                x["index"] = corpus_id
                # Optionally adding the original document and/or just the text that was used
                if return_documents:
                    x["document"] = documents[corpus_id]
                if return_text:
                    x["text"] = get_text(documents[corpus_id])

        def add_query(q):
            return queries[q] if return_queries else None

        results = [
            RerankScores(
                query=add_query(q),
                scores=[RerankScore(**x) for x in r],
            )
            for q, r in enumerate(res)
        ]
        input_token_count = doc_token_count + query_token_count

        return RerankResults(
            results=results,
            producer_id=self.PRODUCER_ID,
            input_token_count=input_token_count,
        )

    @classmethod
    def bootstrap(cls, model_name_or_path: str) -> "EmbeddingModule":
        """Bootstrap a sentence-transformers model

        Args:
            model_name_or_path: str
                Model name (Hugging Face hub) or path to model to load.
        """
        # Local
        from caikit_nlp.modules.text_embedding.sentence_transformer_truncate import (
            SentenceTransformerWithTruncate,
        )

        return cls(
            model=SentenceTransformerWithTruncate(model_name_or_path=model_name_or_path)
        )

    def save(self, model_path: str, *args, **kwargs):
        """Save model using config in model_path

        Args:
            model_path: str
                Path to model config
        """

        error.type_check("<NLP82314992E>", str, model_path=model_path)
        model_config_path = model_path.strip()
        error.value_check(
            "<NLP40145207E>",
            model_config_path,
            f"model_path '{model_config_path}' is invalid",
        )

        model_config_path = os.path.abspath(
            model_config_path.strip()
        )  # No leading/trailing spaces sneaky weirdness

        # Only allow new dirs because there are not enough controls to safely update in-place
        os.makedirs(model_config_path, exist_ok=False)

        saver = ModuleSaver(
            module=self,
            model_path=model_config_path,
        )
        artifacts_path = self._ARTIFACTS_PATH_DEFAULT
        saver.update_config({self._ARTIFACTS_PATH_KEY: artifacts_path})

        # Save the model
        self.model.save(os.path.join(model_config_path, artifacts_path))

        # Save the config
        ModuleConfig(saver.config).save(model_config_path)
