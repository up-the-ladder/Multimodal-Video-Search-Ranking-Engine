import pytest

from videosearch.bm25 import BM25Index, tokenize
from videosearch.segments import Segment


def make_segments(*texts):
    return [
        Segment(segment_id=f"v#{index:05d}", video_id="v", start=float(index), end=index + 1.0, text=text)
        for index, text in enumerate(texts)
    ]


def ranked_ids(hits):
    return [segment.segment_id for segment, _ in hits]


def test_tokenize_lowercases_and_splits_on_punctuation():
    assert tokenize("Edges & corners, pixel_values jump! café 3x3") == [
        "edges",
        "corners",
        "pixel_values",
        "jump",
        "café",
        "3x3",
    ]


def test_score_matches_hand_calculation():
    # Corpus: N = 3, lengths 3, 2, 1, so avgdl = 2.0.
    # "cat" is in two documents, so idf = ln(1 + (3 - 2 + 0.5) / (2 + 0.5)) = ln(1.6).
    # Document 0 has tf = 2 and length 3:
    #   ln(1.6) * (2 * 2.2) / (2 + 1.2 * (0.25 + 0.75 * 3 / 2)) = 0.4700036 * 4.4 / 3.65
    index = BM25Index(make_segments("cat cat dog", "cat bird", "fish"))
    hits = dict(index.search("cat"))

    assert hits[index.segments[0]] == pytest.approx(0.5665797174, abs=1e-9)
    assert hits[index.segments[1]] == pytest.approx(0.4700036292, abs=1e-9)


def test_repeating_a_term_helps_but_with_diminishing_returns():
    index = BM25Index(make_segments("kernel", "kernel kernel kernel kernel"))
    scores = dict(index.search("kernel"))

    once = scores[index.segments[0]]
    four_times = scores[index.segments[1]]

    # More mentions score higher, but four mentions are worth well under
    # four times one mention. That saturation is the k1 term doing its job.
    assert four_times > once
    assert four_times < 4 * once


def test_a_rare_term_outweighs_a_common_one():
    # "the" is in every document, so its idf is small. "kernel" is in one.
    index = BM25Index(make_segments("the kernel", "the image", "the frame", "the pixel"))
    assert ranked_ids(index.search("the kernel"))[0] == "v#00000"


def test_only_segments_containing_a_query_term_are_returned():
    index = BM25Index(make_segments("cat", "dog", "cat dog"))
    assert sorted(ranked_ids(index.search("cat"))) == ["v#00000", "v#00002"]


def test_ties_break_by_segment_id():
    # Identical text means identical scores, so the ID decides the order.
    index = BM25Index(make_segments("same words here", "same words here", "same words here"))
    assert ranked_ids(index.search("same words")) == ["v#00000", "v#00001", "v#00002"]


def test_top_k_truncates_the_ranking():
    index = BM25Index(make_segments("cat", "cat", "cat", "cat"))
    assert len(index.search("cat", top_k=2)) == 2
    assert index.search("cat", top_k=0) == []


@pytest.mark.parametrize("query", ["", "   ", "!!!", "unseen vocabulary"])
def test_empty_and_unmatched_queries_return_nothing(query):
    index = BM25Index(make_segments("cat cat dog", "cat bird"))
    assert index.search(query) == []


def test_empty_corpus_returns_nothing():
    assert BM25Index([]).search("cat") == []


def test_search_is_deterministic():
    index = BM25Index(make_segments("cat cat dog", "cat bird", "dog"))
    assert index.search("cat dog") == index.search("cat dog")
