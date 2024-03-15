# To avoid dependency problems, make sentence-transformers an optional import and
# defer any ModuleNotFoundError until someone actually tries to init a model with this module.
# Standard
from typing import Dict, List, Literal, NamedTuple, Optional, Sized, Union, overload

# Third Party
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import batch_to_device
from transformers import BatchEncoding
import numpy as np
import torch

# First Party
from caikit import get_config
from caikit.core.exceptions import error_handler
import alog

# Local
from caikit_nlp.modules.text_embedding.utils import env_val_to_bool

logger = alog.use_channel("TXT_EMB")
error = error_handler.get(logger)
embedding_cfg = get_config().get("embedding", {})
AUTOCAST = env_val_to_bool(val=embedding_cfg.get("autocast"))


class EmbeddingResultTuple(NamedTuple):
    """Output of SentenceTransformerWithTruncate.encode()"""

    embedding: Union[List[torch.Tensor], np.ndarray, torch.Tensor]
    input_token_count: int


class TruncatedTokensTuple(NamedTuple):
    """Output of SentenceTransformerWithTruncate._truncate_input_tokens()"""

    tokenized: BatchEncoding
    input_token_count: int


def get_sample_start_indexes(tokenized: BatchEncoding) -> List[int]:
    """Returns a list containing the index for the first encoding of each sample
    contained in tokenized."""

    # When truncating occurs a sample is split across multiple encodings
    # ie. len(tokenized.encodings) > the number of text samples input for tokenization

    # Knowing the encoding index of where each sample's first encoding is located allows us to
    # access the encodings for individual samples

    # note: tokenized["overflow_to_sample_mapping"] is a torch.Tensor

    samples_start_indexes: Dict[int, int] = {}
    for i, tensor_sample in enumerate(tokenized["overflow_to_sample_mapping"]):
        int_sample = int(tensor_sample)
        if int_sample not in samples_start_indexes:
            samples_start_indexes[int_sample] = i

    return list(samples_start_indexes.values())


def sum_token_count(
    tokenized: BatchEncoding,
    truncate_only: bool,
) -> int:
    """Returns the number of non-special tokens.
    Args:
        tokenized: BatchEncoding
        truncate_only: bool
    Returns:
        Int total of all tokens contained in tokenized.
    """
    # Encoding objects have various attributes of note:
    # - tokens: list of tokens (sub-parts of the input strings after word/subword
    #       splitting and before conversion to integer indices)
    # - attention_mask: List of indices specifying which tokens should be attended to
    #       by the model. Note that [PAD] = 0, while [CLS] / [SEP] = 1
    # - special_tokens_mask: List of 0s and 1s, with 1 specifying added special tokens
    #       and 0 specifying regular sequence tokens

    error.type_check(
        "<NLP82314993E>",
        BatchEncoding,
        tokenized=tokenized,
    )
    error.value_check(
        "<NLP82314995E>",
        tokenized.encodings,
        "Number of tokenized encodings is only known when a non-python tokenizer is used",
    )

    token_count = 0

    if truncate_only:
        # Only sum the length for the 1st encoding of each sample
        samples_start_idx = get_sample_start_indexes(tokenized)

        token_count = sum(
            (
                x
                for idx in samples_start_idx
                for x in tokenized.encodings[idx].attention_mask
            )
        )
    else:
        # Sum the length of all encodings for all samples
        for encoding in tokenized.encodings:
            token_count += sum(encoding.attention_mask)

    return token_count


class SentenceTransformerWithTruncate(SentenceTransformer):
    def _truncate_input_tokens(
        self, truncate_input_tokens: int, texts: List[str]
    ) -> TruncatedTokensTuple:
        """Truncate input tokens
        Args:
            truncate_input_tokens: int
                Truncation length for input tokens.
                If less than zero, this truncation is left up to the tokenizer default (model max).
                If zero or greater than the model's maximum, then this is used as a test
                to see if truncation is needed. If needed is needed, an exception is thrown.
                Otherwise, we take this usable truncation limit to truncate the input tokens.
            texts: List[str]
                Input texts to be checked and optionally truncated.
        Returns:
            Tuple containing a dictionary of lists/arrays/tensors returned by the tokenizer, with
            proper truncation ('input_ids', 'attention_mask', etc.), and the input_token_count int.
        """

        max_tokens = self.max_seq_length

        # Do truncation if given a usable truncation value, else test for need to truncation
        if truncate_input_tokens < 0:
            okay_to_truncate = True
            max_length = max_tokens
        elif 0 < truncate_input_tokens <= max_tokens:
            okay_to_truncate = True
            max_length = truncate_input_tokens
        else:
            okay_to_truncate = False
            max_length = max_tokens

        assert len(texts) > 0, "Cannot truncate nothing"
        assert isinstance(texts[0], str), "Only str can be truncated"

        to_tokenize = [texts]
        to_tokenize = [[str(s).strip() for s in col] for col in to_tokenize]
        tokenized = self.tokenizer(
            *to_tokenize,
            return_attention_mask=True,
            return_token_type_ids=False,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            return_length=True,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=max_length,
        )

        # When truncation occurs multiple encodings are created for a single sample text
        was_truncated = len(tokenized.encodings) > len(to_tokenize[0])

        if not okay_to_truncate and was_truncated:
            # re-tokenize without truncation to eliminate the duplication of certain
            # special tokens (eg. [CLS] and [SEP]) with each overflow encoding.
            tokenized = self.tokenizer(
                *to_tokenize,
                return_attention_mask=True,
                return_token_type_ids=False,
                return_overflowing_tokens=True,
                return_offsets_mapping=True,
                return_length=True,
                return_tensors="pt",
                truncation=False,
                padding=True,
            )

            tokens = sum_token_count(tokenized, truncate_only=False)
            error.log_raise(
                "<NLP08391926E>",
                ValueError(
                    f"Token sequence length is longer than the specified "
                    f"maximum sequence length for this model ({tokens} > {max_tokens})."
                ),
            )

        input_token_count = sum_token_count(tokenized, truncate_only=True)

        return TruncatedTokensTuple(tokenized, input_token_count)

    @overload
    def encode_with_truncate(
        self,
        sentences: Union[str, List[str]],
        batch_size: int = 32,
        show_progress_bar: Optional[bool] = None,
        output_value: str = "sentence_embedding",
        convert_to_numpy: bool = True,
        convert_to_tensor: bool = False,
        device: Optional[str] = None,
        normalize_embeddings: bool = False,
        truncate_input_tokens: int = 0,
        return_token_count: Literal[False] = False,
    ) -> Union[List[torch.Tensor], np.ndarray, torch.Tensor]:
        ...

    @overload
    def encode_with_truncate(
        self,
        sentences: Union[str, List[str]],
        batch_size: int = 32,
        show_progress_bar: Optional[bool] = None,
        output_value: str = "sentence_embedding",
        convert_to_numpy: bool = True,
        convert_to_tensor: bool = False,
        device: Optional[str] = None,
        normalize_embeddings: bool = False,
        truncate_input_tokens: int = 0,
        return_token_count: Literal[True] = True,
    ) -> EmbeddingResultTuple:
        ...

    def encode_with_truncate(
        self,
        sentences: Union[str, List[str]],
        batch_size: int = 32,
        show_progress_bar: Optional[bool] = None,
        output_value: str = "sentence_embedding",
        convert_to_numpy: bool = True,
        convert_to_tensor: bool = False,
        device: Optional[str] = None,
        normalize_embeddings: bool = False,
        truncate_input_tokens: int = 0,
        return_token_count: bool = False,
    ) -> Union[EmbeddingResultTuple, List[torch.Tensor], np.ndarray, torch.Tensor]:
        """
        Computes sentence embeddings

        :param sentences: the sentences to embed
        :param batch_size: the batch size used for the computation
        :param show_progress_bar: Ignored here. Added for compatibility with super API.
        :param output_value: Ignored here. Added for compatibility with super API.
        :param convert_to_numpy: If true, the output is a list of numpy vectors. Else, it is a list
                of pytorch tensors.
        :param convert_to_tensor: If true, you get one large tensor as return. Overwrites any
                setting from convert_to_numpy
        :param device: Which torch.device to use for the computation
        :param normalize_embeddings: Ignored here. Added for compatibility with super API.
        :param truncate_input_tokens: Truncation length for input tokens.
                Truncation length for input tokens.
                If less than zero, this truncation is left up to the tokenizer default (model max).
                If zero or greater than the model's maximum, then this is used as a test
                to see if truncation is needed. If needed is needed, an exception is thrown.
                Otherwise, we take this usable truncation limit to truncate the input tokens.
        :param return_token_count: If true, a tuple is returned to add the input token count.

        :return:
        If return_token_count is true then return a tuple of the embedding and the input_token_count int.
        If return_token_count is false or not provided then return the embedding.
        """

        # These args are for API compatability, but are currently ignored in our version of encode()
        _ = (
            show_progress_bar,
            output_value,
            normalize_embeddings,
        )

        self.eval()

        if convert_to_tensor:
            convert_to_numpy = False

        input_was_string = False
        list_of_sentences = sentences
        if isinstance(list_of_sentences, str) or not isinstance(
            sentences, Sized
        ):  # Cast an individual sentence to a list with length 1
            list_of_sentences = [sentences]
            input_was_string = True

        error.type_check_all("<NLP82314994E>", str, sentences=list_of_sentences)

        if device is None:
            device = self.device

        self.to(device)

        all_embeddings = []

        # Sort sentences according to length, from longest to shortest
        # OOM errors then occurs at start of encoding
        length_sorted_idx = np.argsort(
            [-self._text_length(sen) for sen in list_of_sentences]
        )
        sentences_sorted: list[str] = [
            list_of_sentences[idx] for idx in length_sorted_idx
        ]

        input_token_count = 0

        for start_index in range(0, len(list_of_sentences), batch_size):
            sentences_batch = sentences_sorted[start_index : start_index + batch_size]
            features, token_count = self._truncate_input_tokens(
                truncate_input_tokens, sentences_batch
            )
            input_token_count += token_count

            features = batch_to_device(features, device)

            if AUTOCAST:
                with torch.no_grad(), torch.cpu.amp.autocast():
                    out_features = self.forward(features)
                    embeddings = out_features["sentence_embedding"]
                    if convert_to_numpy:
                        embeddings = embeddings.detach().cpu()
                    all_embeddings.extend(embeddings)
            else:
                with torch.no_grad():
                    out_features = self.forward(features)
                    embeddings = out_features["sentence_embedding"]
                    if convert_to_numpy:
                        embeddings = embeddings.detach().cpu()
                    all_embeddings.extend(embeddings)

        # Restore original order
        all_embeddings = [all_embeddings[idx] for idx in np.argsort(length_sorted_idx)]

        if convert_to_tensor:
            all_embeddings = torch.stack(all_embeddings)
        elif convert_to_numpy:
            all_embeddings = np.asarray([emb.numpy() for emb in all_embeddings])

        if input_was_string:
            all_embeddings = all_embeddings[0]

        if return_token_count:
            return EmbeddingResultTuple(all_embeddings, input_token_count)
        else:
            return all_embeddings
