from pathlib import Path

import pytest

from videosearch.segments import Segment, read_segments, segment_cues, write_segments
from videosearch.transcripts import Cue, load_vtt

FIXTURE = Path(__file__).parent / "fixtures" / "sample.vtt"


def test_segments_from_fixture_use_cue_timestamps():
    transcript = load_vtt(FIXTURE)
    segments = segment_cues(transcript.video_id, transcript.cues)

    # Window 0 takes the cues at 0s and 5s. The 28s to 36s cue has its midpoint
    # at 32s, so it lands in window 1 together with the 41s cue. Spans come from
    # the cues, which is why window 1 reports 28.0 and not 30.0.
    assert segments == [
        Segment(
            segment_id="sample#00000",
            video_id="sample",
            start=0.0,
            end=12.5,
            text="Welcome to this short lesson on image filters. A convolution kernel slides across the image.",
        ),
        Segment(
            segment_id="sample#00001",
            video_id="sample",
            start=28.0,
            end=45.0,
            text="Edges & corners show up where pixel values jump. Pooling then shrinks the feature map.",
        ),
    ]


def test_midpoint_on_a_window_boundary_belongs_to_the_later_window():
    # Midpoint exactly 30.0 goes to [30, 60), not [0, 30), because the
    # windows are half-open. The second cue has its midpoint just below 30.
    cues = [Cue(28.0, 32.0, "on the boundary"), Cue(29.0, 29.9, "just before")]
    segments = segment_cues("v", cues)

    assert [(segment.segment_id, segment.text) for segment in segments] == [
        ("v#00000", "just before"),
        ("v#00001", "on the boundary"),
    ]


def test_reported_spans_can_overlap_when_windows_do_not():
    # Nothing is duplicated here: each cue is in exactly one segment. The spans
    # overlap only because a cue may start before the window it was assigned to.
    cues = [Cue(25.0, 29.0, "before"), Cue(28.0, 36.0, "across")]
    first, second = segment_cues("v", cues)

    assert (first.start, first.end) == (25.0, 29.0)
    assert (second.start, second.end) == (28.0, 36.0)
    assert second.start < first.end


def test_overlapping_windows_put_one_cue_in_several_segments():
    cues = [Cue(18.0, 22.0, "shared")]
    segments = segment_cues("v", cues, window=30.0, stride=15.0)

    # Midpoint 20.0 falls inside [0, 30) and [15, 45).
    assert [segment.segment_id for segment in segments] == ["v#00000", "v#00001"]
    assert all(segment.text == "shared" for segment in segments)


def test_silent_windows_produce_no_segments():
    cues = [Cue(1.0, 3.0, "start"), Cue(200.0, 202.0, "much later")]
    segments = segment_cues("v", cues)

    # Window 6 covers [180, 210). The five silent windows in between are gone,
    # and the IDs still say where each segment sits on the timeline.
    assert [segment.segment_id for segment in segments] == ["v#00000", "v#00006"]


def test_text_is_joined_in_cue_order():
    cues = [Cue(1.0, 2.0, "one"), Cue(3.0, 4.0, "two"), Cue(5.0, 6.0, "three")]
    assert segment_cues("v", cues)[0].text == "one two three"


def test_segmentation_is_deterministic():
    cues = [Cue(1.0, 2.0, "one"), Cue(40.0, 41.0, "two")]
    assert segment_cues("v", cues) == segment_cues("v", cues)


def test_no_cues_gives_no_segments():
    assert segment_cues("v", []) == []


@pytest.mark.parametrize(("window", "stride"), [(30.0, 60.0), (0.0, 30.0), (30.0, -5.0)])
def test_invalid_window_and_stride_are_rejected(window, stride):
    # stride > window would leave gaps, so some speech would land in no
    # segment and quietly become unsearchable.
    with pytest.raises(ValueError):
        segment_cues("v", [Cue(1.0, 2.0, "one")], window=window, stride=stride)


def test_segment_rejects_end_before_start():
    with pytest.raises(ValueError):
        Segment(segment_id="v#00000", video_id="v", start=5.0, end=4.0, text="backwards")


def test_jsonl_round_trip_preserves_segments(tmp_path):
    cues = [Cue(1.0, 2.5, 'quotes " and unicode: café'), Cue(40.0, 41.0, "two")]
    segments = segment_cues("v", cues)

    path = tmp_path / "nested" / "segments.jsonl"
    write_segments(path, segments)

    assert read_segments(path) == segments
    # One JSON object per line, so the file can be read back a line at a time.
    assert len(path.read_text(encoding="utf-8").strip().split("\n")) == 2
