"""Command line entry point: build segments from transcripts, then search them."""

import argparse
import sys
import textwrap
from pathlib import Path

from videosearch.bm25 import BM25Index
from videosearch.segments import DEFAULT_STRIDE, DEFAULT_WINDOW, read_segments, segment_cues, write_segments
from videosearch.transcripts import load_vtt


def format_timestamp(seconds):
    """Seconds to mm:ss, or hh:mm:ss once the video is over an hour long."""
    whole = int(seconds)
    hours, remainder = divmod(whole, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def build(args):
    # Non-recursive on purpose: every transcript sits directly in the
    # directory, so video IDs (the filename stems) cannot collide.
    paths = sorted(Path(args.transcripts).glob("*.vtt"))
    if not paths:
        print(f"no .vtt files in {args.transcripts}", file=sys.stderr)
        return 1

    segments = []
    for path in paths:
        transcript = load_vtt(path)
        segments.extend(segment_cues(transcript.video_id, transcript.cues, args.window, args.stride))

    write_segments(args.out, segments)
    print(f"{len(segments)} segments from {len(paths)} transcripts -> {args.out}")
    return 0


def search(args):
    if not Path(args.segments).exists():
        print(f"no segments at {args.segments}, run 'videosearch build' first", file=sys.stderr)
        return 1

    # The index is rebuilt on every query. That is fine at this corpus size,
    # and we should only add persistence once load time is measurably a problem.
    hits = BM25Index(read_segments(args.segments)).search(args.query, args.top_k)
    if not hits:
        print("no matches")
        return 0

    for rank, (segment, score) in enumerate(hits, start=1):
        span = f"{format_timestamp(segment.start)}-{format_timestamp(segment.end)}"
        print(f"{rank}. {score:.3f}  {segment.video_id}  {span}")
        print(f"   {textwrap.shorten(segment.text, width=100, placeholder=' ...')}")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(prog="videosearch", description="Search video transcripts by segment.")
    commands = parser.add_subparsers(dest="command", required=True)

    build_parser = commands.add_parser("build", help="turn a directory of .vtt transcripts into segments")
    build_parser.add_argument("transcripts", help="directory holding .vtt files")
    build_parser.add_argument("--out", default="data/segments.jsonl", help="where to write the segments")
    build_parser.add_argument("--window", type=float, default=DEFAULT_WINDOW, help="window length in seconds")
    build_parser.add_argument("--stride", type=float, default=DEFAULT_STRIDE, help="window step in seconds")
    build_parser.set_defaults(run=build)

    search_parser = commands.add_parser("search", help="rank segments against a query")
    search_parser.add_argument("query")
    search_parser.add_argument("--segments", default="data/segments.jsonl", help="segments file to search")
    search_parser.add_argument("-k", "--top-k", type=int, default=5, help="how many results to show")
    search_parser.set_defaults(run=search)

    args = parser.parse_args(argv)
    try:
        return args.run(args)
    except (OSError, ValueError) as error:
        # Bad segmentation parameters and unreadable files are ordinary user
        # mistakes, so report them plainly instead of dumping a traceback.
        print(error, file=sys.stderr)
        return 1
