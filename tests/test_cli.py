from pathlib import Path

import pytest

from videosearch.cli import format_timestamp, main

EXAMPLES = Path(__file__).parents[1] / "examples" / "transcripts"


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [(0.0, "00:00"), (83.6, "01:23"), (600.0, "10:00"), (3723.0, "01:02:03")],
)
def test_format_timestamp(seconds, expected):
    assert format_timestamp(seconds) == expected


def test_build_then_search_finds_the_right_segment(tmp_path, capsys):
    out = tmp_path / "segments.jsonl"

    assert main(["build", str(EXAMPLES), "--out", str(out)]) == 0
    assert out.exists()
    capsys.readouterr()

    assert main(["search", "how does a kernel slide across the image", "--segments", str(out), "-k", "3"]) == 0
    lines = capsys.readouterr().out.splitlines()

    # The convolution transcript talks about kernels sliding, the sourdough one
    # does not, so the top hit should be the first half minute of that video.
    assert lines[0].startswith("1.")
    assert "intro_to_convolutions" in lines[0]
    assert "00:00-00:29" in lines[0]


def test_search_ranks_the_other_video_for_a_different_query(tmp_path, capsys):
    out = tmp_path / "segments.jsonl"
    main(["build", str(EXAMPLES), "--out", str(out)])
    capsys.readouterr()

    main(["search", "sourdough starter flour and water", "--segments", str(out), "-k", "1"])
    assert "making_sourdough" in capsys.readouterr().out


def test_window_and_stride_reach_segmentation(tmp_path, capsys):
    default_out = tmp_path / "default.jsonl"
    overlapped_out = tmp_path / "overlapped.jsonl"

    main(["build", str(EXAMPLES), "--out", str(default_out)])
    main(["build", str(EXAMPLES), "--out", str(overlapped_out), "--window", "30", "--stride", "15"])
    capsys.readouterr()

    # Overlapping windows cover the same speech more than once, so there
    # have to be more segments than the non-overlapping default gives.
    assert len(overlapped_out.read_text(encoding="utf-8").splitlines()) > len(
        default_out.read_text(encoding="utf-8").splitlines()
    )


def test_a_query_matching_nothing_says_so(tmp_path, capsys):
    out = tmp_path / "segments.jsonl"
    main(["build", str(EXAMPLES), "--out", str(out)])
    capsys.readouterr()

    assert main(["search", "xylophone", "--segments", str(out)]) == 0
    assert capsys.readouterr().out.strip() == "no matches"


def test_missing_inputs_fail_without_a_traceback(tmp_path, capsys):
    assert main(["build", str(tmp_path), "--out", str(tmp_path / "out.jsonl")]) == 1
    assert "no .vtt files" in capsys.readouterr().err

    assert main(["search", "anything", "--segments", str(tmp_path / "missing.jsonl")]) == 1
    assert "videosearch build" in capsys.readouterr().err


def test_bad_segmentation_parameters_fail_cleanly(tmp_path, capsys):
    out = tmp_path / "segments.jsonl"
    # stride > window breaks the coverage invariant, and the CLI should report
    # that as a plain message rather than letting the ValueError escape.
    assert main(["build", str(EXAMPLES), "--out", str(out), "--window", "30", "--stride", "60"]) == 1
    assert "stride must be <= window" in capsys.readouterr().err
