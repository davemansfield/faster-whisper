from scipy.stats import entropy
from collections import Counter
import zlib
from typing import List
import spacy
from typing import Any, Tuple
from nltk import ngrams

REMOVE_STOPWORDS = False
WEIGHT_VALUES = False
nlp = spacy.load("en_core_web_sm")


def get_average_ngram_count(ngram_counts: Any) -> float:
    total_ngrams = sum(ngram_counts.values())
    if total_ngrams > 0:
        return total_ngrams / len(ngram_counts)
    else:
        return 0


# Currently unused, decided against weighting the values of ngrams
def weight_values(ngrams: Tuple[float, ...]) -> Tuple[float, ...]:
    return tuple(ngram * i for i, ngram in enumerate(ngrams))


def calculate_ngram_average(tokens: List[str], n: int) -> float:
    ng = ngrams(tokens, n)
    ng_counts = Counter(ng)

    # Remove ngrams that only appear once, as that definitely has not repeated
    ng_counts = {k: v for k, v in ng_counts.items() if v > 1}
    # pprint(ng_counts)
    return get_average_ngram_count(ng_counts)


def get_ngrams(txt: str):
    doc = nlp(txt)

    if REMOVE_STOPWORDS:
        tokens = [
            token.text
            for token in doc
            if not token.is_stop and not token.is_punct and token.text.strip() != ""
        ]
    else:
        tokens = txt.split()

    average_bigrams = calculate_ngram_average(tokens, 2)
    average_trigrams = calculate_ngram_average(tokens, 3)
    average_quadgrams = calculate_ngram_average(tokens, 4)
    average_pentagrams = calculate_ngram_average(tokens, 5)

    ng = tuple((average_bigrams, average_trigrams, average_quadgrams, average_pentagrams))

    if WEIGHT_VALUES:
        ng = weight_values(ng)

    return ng


def get_ngram_score(ngrams: Tuple[float, ...]) -> float:
    return sum(ngrams)


def calculate_text_entropy(text: str) -> float:
    counts = Counter(text)
    total_characters = len(text)
    probabilities = [i / total_characters for i in counts.values()]

    return entropy(probabilities)


def get_compression_rate(text: str) -> float:
    compression = len(zlib.compress(text.encode("utf-8")))
    # print(len(text))
    # print("compressed length", compression)
    ratio = len(text) / compression
    return ratio
