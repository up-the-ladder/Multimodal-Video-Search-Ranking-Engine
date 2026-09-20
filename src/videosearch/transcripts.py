"""Load timestamped transcripts into timed cues."""

import html
import re
from dataclasses import dataclass
from pathlib import Path

import webvtt

# WebVTT timestamps are [hh:]mm:ss.ttt. Hours are optional (Whisper's VTT
# writer drops them for short videos) but need at least two digits when
# present. Minutes and seconds are always exactly two digits.
TIMESTAMP_PATTERN = re.compile(r"^(?:(\d{2,}):)?(\d{2}):(\d{2})\.(\d{3})$")

# In WebVTT a literal "<" must be written as &lt;, so any raw "<" in cue
# text starts a tag (voice, italics, karaoke timestamps, and so on).
TAG_PATTERN = re.compile(r"<[^>]*>")


@dataclass(frozen=True)
class Cue:
    start: float
    end: float
    text: str

    def __post_init__(self):
        # Validate here rather than in the VTT loader so cues coming from
        # any future source (Whisper, other subtitle formats) get the same check.
        if self.end < self.start:
            raise ValueError(f"cue ends before it starts: {self.start} -> {self.end}")


@dataclass(frozen=True)
class Transcript:
    video_id: str
    cues: list[Cue]


def parse_timestamp(value):
    match = TIMESTAMP_PATTERN.match(value.strip())
    if match is None:
        raise ValueError(f"not a WebVTT timestamp: {value!r}")

    hours, minutes, seconds, millis = match.groups()
    # The pattern only checks digit counts. Minutes and seconds also have to
    # be real clock values, with or without an hours component.
    if int(minutes) > 59 or int(seconds) > 59:
        raise ValueError(f"minutes and seconds must be 00-59: {value!r}")

    # Add everything up in whole milliseconds and divide once at the end,
    # so we don't pile up float error from summing fractional parts.
    total_ms = ((int(hours or 0) * 60 + int(minutes)) * 60 + int(seconds)) * 1000 + int(millis)
    return total_ms / 1000


def clean_text(lines):
    # Line breaks inside a cue are just on-screen layout, so join with spaces.
    text = " ".join(lines)

    # Strip tags before unescaping. The other order would turn an escaped
    # "&lt;b&gt;" into "<b>" and then wrongly delete it as if it were a tag.
    # Note that voice tags carry the speaker name, which we drop for now.
    # Nothing uses speakers yet, and the source file still has them.
    text = TAG_PATTERN.sub("", text)
    text = html.unescape(text)

    # Case and punctuation are left alone on purpose. That's the tokenizer's
    # job, and search results should show the text as it was written.
    return " ".join(text.split())


def load_vtt(path):
    path = Path(path)

    cues = []
    for caption in webvtt.read(str(path)):
        text = clean_text(caption.lines)
        # A cue that is only tags or whitespace has nothing to search.
        if not text:
            continue
        cues.append(Cue(parse_timestamp(caption.start), parse_timestamp(caption.end), text))

    # WebVTT already requires cues in start-time order. Sort here anyway so
    # segmentation can rely on chronological order whatever the source did.
    cues.sort(key=lambda cue: (cue.start, cue.end))

    # The video ID is the filename minus its final extension, so
    # "lecture.en.vtt" becomes "lecture.en". We don't guess at language
    # suffixes. Keeping one transcript per video is up to whoever builds
    # the transcript directory.
    return Transcript(video_id=path.stem, cues=cues)
