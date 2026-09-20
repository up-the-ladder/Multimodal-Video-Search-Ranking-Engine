"""A small BM25 index over segment text: the sparse retrieval baseline."""

import math
import re
from collections import Counter, defaultdict

# Lowercase plus a plain word split. No stemming and no stopword list, so
# "convolution" will not match "convolutional". That is a real limitation,
# and seeing where it hurts is more useful than guessing at a fix now.
TOKEN_PATTERN = re.compile(r"\w+")

# The usual starting values from the BM25 literature, not tuned on anything.
DEFAULT_K1 = 1.2
DEFAULT_B = 0.75


def tokenize(text):
    return TOKEN_PATTERN.findall(text.lower())


class BM25Index:
    """Scores segments against a query with Okapi BM25.

    A class rather than a function because the index owns state that is built
    once and reused for every query: the postings, the document lengths, and
    the IDF of each term.
    """

    def __init__(self, segments, k1=DEFAULT_K1, b=DEFAULT_B):
        self.segments = list(segments)
        self.k1 = k1
        self.b = b

        # The inverted index: term -> [(document index, term frequency), ...].
        # This is what lets a query touch only the documents that contain its
        # terms instead of scanning the whole corpus every time.
        self.postings = defaultdict(list)
        self.doc_lengths = []

        for doc_index, segment in enumerate(self.segments):
            tokens = tokenize(segment.text)
            self.doc_lengths.append(len(tokens))
            for term, count in Counter(tokens).items():
                self.postings[term].append((doc_index, count))

        self.avg_doc_length = sum(self.doc_lengths) / len(self.doc_lengths) if self.doc_lengths else 0.0

        # IDF depends only on the corpus, so compute it once at build time.
        # This is the Lucene form, ln(1 + (N - df + 0.5) / (df + 0.5)), which
        # stays positive even for a term that appears in every document. The
        # textbook form goes negative there and can subtract from a score.
        total_docs = len(self.segments)
        self.idf = {
            term: math.log(1 + (total_docs - len(postings) + 0.5) / (len(postings) + 0.5))
            for term, postings in self.postings.items()
        }

    def search(self, query, top_k=10):
        """Return up to top_k (segment, score) pairs, best first."""
        if top_k <= 0:
            return []

        scores = defaultdict(float)
        # Query terms are counted as they appear, so repeating a word in the
        # query weighs it more heavily. Terms we have never seen simply
        # contribute nothing.
        for term in tokenize(query):
            if term not in self.postings:
                continue

            idf = self.idf[term]
            for doc_index, term_frequency in self.postings[term]:
                # Two ideas in one line. The k1 part makes repeated terms help
                # less and less, so a segment saying "kernel" ten times does not
                # beat everything else outright. The b part divides by how long
                # the segment is relative to the average, so a long segment does
                # not win just by containing more words.
                length_ratio = self.doc_lengths[doc_index] / self.avg_doc_length
                denominator = term_frequency + self.k1 * (1 - self.b + self.b * length_ratio)
                scores[doc_index] += idf * term_frequency * (self.k1 + 1) / denominator

        # Ties break by segment ID so the ranking is reproducible. Segment IDs
        # are zero padded, so tied segments come back in timeline order.
        ranked = sorted(scores.items(), key=lambda item: (-item[1], self.segments[item[0]].segment_id))
        return [(self.segments[doc_index], score) for doc_index, score in ranked[:top_k]]
