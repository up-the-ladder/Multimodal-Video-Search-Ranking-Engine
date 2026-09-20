# Multimodal Video Search and Ranking Engine

Search inside videos and get back the moment that answers your query, not just the video.

The long-term aim is multimodal: speech, on-screen text, and what is visible in the frames all
carry information, and a query should be able to find any of it. **None of that exists yet.**
This is V0, the first working layer: transcript-only retrieval with BM25 over timestamped
segments. It is the baseline that every later modality and retrieval method will have to beat.

## What works today

```text
WebVTT transcripts
  -> cues (start, end, text)
  -> fixed time windows
  -> segments (segment_id, video_id, start, end, text)
  -> segments.jsonl
  -> BM25 index
  -> ranked, timestamped results
```

Two commands. `build` is the offline step, `search` is the online one.

## Install

Python 3.12 or newer.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
python -m pytest
```

## Usage

```bash
videosearch build examples/transcripts --out data/segments.jsonl
videosearch search "how does a kernel slide across the image" -k 3
```

Output from the two example transcripts in this repository:

```text
6 segments from 2 transcripts -> data/segments.jsonl

1. 4.651  intro_to_convolutions  00:00-00:29
   Today we are going to look at how a convolution actually works on an image. A kernel is a small ...
2. 3.614  intro_to_convolutions  00:31-00:55
   Different kernels pick out different things. One might respond to horizontal edges, another to ...
3. 0.238  making_sourdough  01:03-01:11
   Take the lid off for the last fifteen minutes to let the outside turn a deep brown.
```

That third hit matched only on common words like "the" and "a". It scores far below the real
matches, but it is still returned, which is what no stopword list looks like in practice.

`build` takes `--window` and `--stride` in seconds. `search` takes `-k` and `--segments`.

## The retrieval unit

A search result is a **segment**: a stretch of one video, with a start and an end.

Whole videos are too coarse to answer "where does she explain padding", and a single subtitle cue
is usually too short to carry enough context for the ranker. A segment sits in between.

Segmentation works like this:

- Windows are laid out from time zero, so window `i` covers `[i * stride, i * stride + window)`.
- Windows are half-open, so a cue whose midpoint is exactly at 30.0s belongs to `[30, 60)`.
- A cue goes to every window containing its **midpoint**. With the default 30s window and 30s
  stride that means exactly one window. With a shorter stride, windows overlap and a cue can
  land in several.
- `stride <= window` is enforced, so no speech falls between windows and becomes unsearchable.
- Windows with no speech in them produce no segment.
- A segment's reported `start` and `end` come from its cues, not from the window bounds, so
  timestamps point at real speech. A side effect worth knowing: because a cue can begin before
  the window it was assigned to, reported spans can overlap slightly even when the windows do
  not. Nothing is duplicated when that happens.

Midpoint assignment is the V0 policy, not a claim about the best way to segment video. Assigning
by cue start, putting a cue in every window it touches, or splitting cues on word timestamps are
all reasonable alternatives. Choosing between them needs an evaluation set, which does not exist
yet.

Defaults of 30s and 30s are a starting point, not tuned values.

## Ranking

Okapi BM25, written out in `bm25.py` rather than pulled from a library, because understanding it
matters more here than saving fifty lines.

- Tokenization is lowercasing plus a `\w+` split. No stemming and no stopword list.
- IDF uses the Lucene form, `ln(1 + (N - df + 0.5) / (df + 0.5))`, which stays positive for terms
  that appear in every document. The textbook form goes negative there.
- `k1 = 1.2` and `b = 0.75`, the usual starting values, not tuned on anything.
- Rare query terms count for more, repeated occurrences within a segment give diminishing returns,
  and longer segments are penalised relative to the average length.
- Results are sorted by score, with ties broken by segment ID so a ranking is reproducible.
- The index is in memory and rebuilt on each search.

## Repository layout

```text
src/videosearch/
    transcripts.py   WebVTT -> Cue, text cleaning, timestamp parsing
    segments.py      cues -> Segment, JSONL read and write
    bm25.py          tokenizer and BM25 index
    cli.py           build and search commands
tests/               one module per source file, plus an end-to-end CLI test
examples/transcripts/  two short transcripts for the demo above
```

The example transcripts were written by hand for this repository. They are **synthetic** and do
not come from real videos.

## Tests

`pytest` covers timestamp parsing and text cleaning, segmentation boundaries and window overlap,
JSONL round-tripping, BM25 scoring against a hand-calculated value, ranking and tie-breaking
behaviour, and the CLI end to end.

## Limitations

Everything here is honest about V0 being early:

- No transcription. Transcripts have to exist already, as WebVTT files.
- No visual understanding, OCR, or generated captions.
- No dense or hybrid retrieval and no reranking.
- No evaluation. There is no labeled query set yet, so there are **no retrieval quality numbers,
  no latency numbers, and no claims about scale** anywhere in this repository.
- No stemming or stopwords, so "convolution" does not match "convolutional" and common words can
  still drag in weak matches.
- Overlapping windows produce near-duplicate results, which nothing currently suppresses.
- Segmentation defaults are untuned.
- Corpus size is tiny, and the index is rebuilt per query.

## What comes next

The next milestone is evaluation: a real transcript dataset, a written relevance protocol with
labels defined as time spans rather than segment IDs, and metrics such as Recall@K and MRR. Only
then is there any evidence about where BM25 actually fails.

After that, each addition has to earn its place against measurements: better transcript retrieval,
then Whisper for videos that have no transcript, then dense retrieval, hybrid fusion, reranking,
and visual signals. The order follows the experiments, not a fixed plan.

## License

MIT, see `LICENSE`. Note that datasets, models, and media added later come with their own terms,
which need checking separately.
