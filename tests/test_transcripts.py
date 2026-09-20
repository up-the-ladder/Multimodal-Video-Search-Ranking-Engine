from pathlib import Path

import pytest

from videosearch.transcripts import Cue, load_vtt, parse_timestamp

FIXTURE = Path(__file__).parent / "fixtures" / "sample.vtt"


def write_vtt(directory, name, body):
    path = directory / name
    path.write_text("WEBVTT\n\n" + body, encoding="utf-8")
    return path


def test_parse_timestamp_with_and_without_hours():
    assert parse_timestamp("01:02:03.500") == 3723.5
    assert parse_timestamp("02:03.500") == 123.5
    assert parse_timestamp("00:00:30.000") == 30.0


@pytest.mark.parametrize(
    "value",
    ["00:30", "1:2:3.000", "1:02:03.000", "00:00:30,000", "00:99:99.000", "60:00.000", ""],
)
def test_parse_timestamp_rejects_malformed_values(value):
    # Bad shapes (missing fields, one-digit hours, SRT's comma) and
    # impossible clock values should all fail loudly instead of parsing.
    with pytest.raises(ValueError):
        parse_timestamp(value)


def test_cue_rejects_end_before_start():
    with pytest.raises(ValueError):
        Cue(start=5.0, end=4.0, text="backwards")


def test_load_vtt_reads_cues_as_seconds_and_clean_text():
    transcript = load_vtt(FIXTURE)

    # The tag-only cue at 13s is dropped. Everything else keeps its
    # timing, with multi-line text joined, the voice tag stripped,
    # and &amp; unescaped.
    assert transcript.cues == [
        Cue(0.0, 4.0, "Welcome to this short lesson on image filters."),
        Cue(5.0, 12.5, "A convolution kernel slides across the image."),
        Cue(28.0, 36.0, "Edges & corners show up where pixel values jump."),
        Cue(41.0, 45.0, "Pooling then shrinks the feature map."),
    ]


def test_escaped_angle_brackets_survive_as_text(tmp_path):
    path = write_vtt(tmp_path, "escapes.vtt", "00:00:01.000 --> 00:00:02.000\n<b>Keep</b> &lt;b&gt; as text\n")
    assert load_vtt(path).cues[0].text == "Keep <b> as text"


def test_load_vtt_returns_cues_in_time_order(tmp_path):
    body = "00:00:10.000 --> 00:00:12.000\nsecond\n\n00:00:01.000 --> 00:00:03.000\nfirst\n"
    path = write_vtt(tmp_path, "unordered.vtt", body)
    assert [cue.text for cue in load_vtt(path).cues] == ["first", "second"]


@pytest.mark.parametrize(
    ("filename", "video_id"),
    [("intro.vtt", "intro"), ("lecture.en.vtt", "lecture.en")],
)
def test_video_id_is_filename_without_final_extension(tmp_path, filename, video_id):
    path = write_vtt(tmp_path, filename, "00:00:01.000 --> 00:00:02.000\nhello\n")
    assert load_vtt(path).video_id == video_id
