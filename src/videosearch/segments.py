"""Turn transcript cues into the retrieval unit: a timestamped segment."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

# A starting point, not a tuned value. Once we have relevance labels we can
# measure what window and stride actually do to retrieval quality.
DEFAULT_WINDOW = 30.0
DEFAULT_STRIDE = 30.0


@dataclass(frozen=True)
class Segment:
    segment_id: str
    video_id: str
    start: float
    end: float
    text: str

    def __post_init__(self):
        if self.end < self.start:
            raise ValueError(f"segment ends before it starts: {self.start} -> {self.end}")


def segment_cues(video_id, cues, window=DEFAULT_WINDOW, stride=DEFAULT_STRIDE):
    """Group cues into fixed time windows and return one segment per non-empty window.

    Windows are half-open [start, end) and laid out from time zero, so window i
    covers [i * stride, i * stride + window). A cue goes to every window that
    contains its midpoint, which means one window when stride equals window,
    and possibly several when the windows overlap.
    """
    if window <= 0 or stride <= 0:
        raise ValueError(f"window and stride must be positive: window={window}, stride={stride}")

    # A stride wider than the window would leave gaps in the grid, so some
    # speech would end up in no segment at all and become unsearchable.
    # That is a project rule about coverage, not a law of segmentation.
    if stride > window:
        raise ValueError(f"stride must be <= window for full coverage: stride={stride}, window={window}")

    if not cues:
        return []

    # Midpoint assignment is the V0 policy. It is deterministic and gives each
    # cue exactly one home when windows do not overlap, but it is only one
    # choice among several: we could assign by cue start, or put a cue in every
    # window it touches, or split cues if we had word-level timestamps.
    midpoints = [(cue.start + cue.end) / 2 for cue in cues]
    last_end = max(cue.end for cue in cues)

    segments = []
    index = 0
    while index * stride <= last_end:
        window_start = index * stride
        window_end = window_start + window

        # Compare against the window bounds directly instead of dividing the
        # midpoint by the stride. Division would make the half-open boundary
        # depend on float rounding, and the boundary is exactly what we care about.
        in_window = [cue for cue, midpoint in zip(cues, midpoints) if window_start <= midpoint < window_end]
        index += 1

        if not in_window:
            continue

        # The reported span comes from the cues themselves, not the window, so
        # timestamps point at real speech. A consequence worth remembering: a
        # cue can start before its window, so reported spans can overlap even
        # when the windows do not. Cues are sorted by start time, but a nested
        # cue can end earlier than one that starts before it, so take extremes.
        segments.append(
            Segment(
                # Zero padding keeps IDs sorting in time order, which is handy
                # later when ties in a ranking are broken by segment ID.
                segment_id=f"{video_id}#{index - 1:05d}",
                video_id=video_id,
                start=min(cue.start for cue in in_window),
                end=max(cue.end for cue in in_window),
                text=" ".join(cue.text for cue in in_window),
            )
        )

    return segments


def write_segments(path, segments):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # JSONL rather than one big JSON array: each line stands alone, so the file
    # stays readable, greppable, and easy to append to as the corpus grows.
    with path.open("w", encoding="utf-8") as out:
        for segment in segments:
            out.write(json.dumps(asdict(segment), ensure_ascii=False) + "\n")


def read_segments(path):
    segments = []
    with Path(path).open(encoding="utf-8") as lines:
        for line in lines:
            if line.strip():
                segments.append(Segment(**json.loads(line)))
    return segments
