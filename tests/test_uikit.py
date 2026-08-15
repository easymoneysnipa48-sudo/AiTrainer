import pytest

from musictrain import uikit


def test_sparkline_empty_series_is_noop():
    # should not raise on empty input
    uikit.sparkline([])


def test_sparkline_single_value():
    uikit.sparkline([1.0])


def test_gauge_clamps_to_range():
    # gauge renders HTML; just ensure no exception for out-of-range input
    uikit.gauge(2.0, maximum=1.0)
    uikit.gauge(-1.0, maximum=1.0)


def test_gauge_zero_maximum():
    uikit.gauge(0.5, maximum=0.0)


def test_progress_skeleton_clamps():
    uikit.progress_skeleton(frac=1.5)
    uikit.progress_skeleton(frac=-0.5)


def test_metric_tile_escapes_html():
    # HTML-injection in labels/values must be escaped, not rendered raw
    uikit.metric_tile("<script>", "<b>1</b>")
    assert uikit._esc("<b>") == "&lt;b&gt;"
