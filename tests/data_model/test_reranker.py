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
"""Test for reranker
"""

# Standard
import random
import string

# Third Party
import pytest

# First Party
from caikit.interfaces.nlp.data_model import (
    RerankResult,
    RerankResults,
    RerankScore,
    RerankScores,
    SentenceSimilarityResult,
    SentenceSimilarityResults,
    SentenceSimilarityScores,
)

## Setup #########################################################################


@pytest.fixture
def input_document():
    return {
        "text": "this is the input text",
        "_text": "alternate _text here",
        "title": "some title attribute here",
        "anything": "another string attribute",
        "str_test": "test string",
        "int_test": 1234,
        "float_test": 9876.4321,
    }


@pytest.fixture
def input_random_document():
    return {
        "text": "".join(random.choices(string.printable, k=100)),
        "random_str": "".join(random.choices(string.printable, k=100)),
        "random_int": random.randint(-99999, 99999),
        "random_float": random.uniform(-99999, 99999),
    }


@pytest.fixture
def input_documents(input_document, input_random_document):
    return [input_document, input_random_document]


@pytest.fixture
def input_score(input_document):
    return {
        "document": input_document,
        "index": 1234,
        "score": 9876.54321,
        "text": "this is the input text",
    }


@pytest.fixture
def input_random_score(input_random_document):
    return {
        "document": input_random_document,
        "index": random.randint(-99999, 99999),
        "score": random.uniform(-99999, 99999),
        "text": "".join(random.choices(string.printable, k=100)),
    }


@pytest.fixture
def input_random_score_3():
    return {
        "document": {"text": "random foo3"},
        "index": random.randint(-99999, 99999),
        "score": random.uniform(-99999, 99999),
        "text": "".join(random.choices(string.printable, k=100)),
    }


@pytest.fixture
def input_scores(input_score, input_random_score):
    return [RerankScore(**input_score), RerankScore(**input_random_score)]


@pytest.fixture
def input_scores2(input_random_score, input_random_score_3):
    return [
        RerankScore(**input_random_score),
        RerankScore(**input_random_score_3),
    ]


@pytest.fixture
def rerank_result_1(input_scores):
    return {"result": RerankScores(query="foo", scores=input_scores)}


@pytest.fixture
def rerank_result_2(input_scores2):
    return {"result": RerankScores(query="bar", scores=input_scores2)}


@pytest.fixture
def input_sentence_similarity_scores_1():
    return {"scores": [random.uniform(-99999, 99999) for _ in range(10)]}


@pytest.fixture
def input_sentence_similarity_scores_2():
    return {"scores": [random.uniform(-99999, 99999) for _ in range(10)]}


@pytest.fixture
def sentence_similarity_result(input_sentence_similarity_scores_1):
    return {"result": SentenceSimilarityScores(**input_sentence_similarity_scores_1)}


@pytest.fixture
def sentence_similarity_results(
    input_sentence_similarity_scores_1, input_sentence_similarity_scores_2
):
    return {
        "results": [
            SentenceSimilarityScores(**input_sentence_similarity_scores_1),
            SentenceSimilarityScores(**input_sentence_similarity_scores_2),
        ]
    }


@pytest.fixture
def rerank_results(rerank_result_1, rerank_result_2):
    return {"results": [rerank_result_1["result"], rerank_result_2["result"]]}


@pytest.fixture
def input_sentence_list_scores(input_sentence_similarities_scores):
    return {"results": input_sentence_similarities_scores}


@pytest.fixture
def input_sentence_similarity_scores_2():
    return {"scores": [random.uniform(-99999, 99999) for _ in range(10)]}


@pytest.fixture
def input_sentence_similarities_scores(
    input_sentence_similarity_scores_1, input_sentence_similarity_scores_2
):
    return [
        SentenceSimilarityResult(**input_sentence_similarity_scores_1),
        SentenceSimilarityResult(**input_sentence_similarity_scores_2),
    ]


## Tests ########################################################################


@pytest.mark.parametrize(
    "data_object, inputs",
    [
        (RerankScore, "input_score"),
        (RerankScore, "input_random_score"),
        (RerankResult, "rerank_result_1"),
        (RerankResults, "rerank_results"),
        (SentenceSimilarityResult, "sentence_similarity_result"),
        (SentenceSimilarityResults, "sentence_similarity_results"),
    ],
)
def test_data_object(data_object, inputs, request):
    # Init data object
    fixture_values = request.getfixturevalue(inputs)
    new_do_from_init = data_object(**fixture_values)
    assert isinstance(new_do_from_init, data_object)
    assert_fields_match(new_do_from_init, fixture_values)

    # Test to/from proto
    proto_from_dm = new_do_from_init.to_proto()
    new_do_from_proto = data_object.from_proto(proto_from_dm)
    assert isinstance(new_do_from_proto, data_object)
    assert_fields_match(new_do_from_proto, fixture_values)
    assert new_do_from_init == new_do_from_proto

    # Test to/from json
    json_from_dm = new_do_from_init.to_json()
    new_do_from_json = data_object.from_json(json_from_dm)
    assert isinstance(new_do_from_json, data_object)
    assert_fields_match(new_do_from_json, fixture_values)
    assert new_do_from_init == new_do_from_json


def assert_fields_match(data_object, inputs):
    assert all(getattr(data_object, key) == value for key, value in inputs.items())
