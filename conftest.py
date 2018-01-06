import pytest
import spacy


@pytest.fixture
def nlp():
    return spacy.load('en')