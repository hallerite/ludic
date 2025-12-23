from __future__ import annotations

from ludic.training.stats import Reducer, apply_reducers_to_records


def test_reducer_defaults() -> None:
    r = Reducer(kind="mean", source="x")
    assert r.kind == "mean"
    assert r.source == "x"
    assert r.transform is None
    assert r.normalize_by is None
    assert r.as_percent is False


def test_apply_reducers_to_records_mean_sum_count_true() -> None:
    records = [
        {"x": 1.0, "y": 2.0, "ok": True},
        {"x": 3.0, "y": 5.0, "ok": False},
    ]

    reducers = {
        "x_mean": Reducer(kind="mean", source="x"),
        "y_sum": Reducer(kind="sum", source="y"),
        "ok_count": Reducer(kind="count_true", source="ok"),
        "ok_rate": Reducer(kind="count_true", source="ok", normalize_by="samples"),
    }

    out = apply_reducers_to_records(records, reducers)
    assert out["x_mean"] == 2.0
    assert out["y_sum"] == 7.0
    assert out["ok_count"] == 1.0
    assert out["ok_rate"] == 0.5


def test_apply_reducers_to_records_dotted_path_and_transform() -> None:
    records = [
        {"nested": {"result": "win"}},
        {"nested": {"result": "loss"}},
        {"nested": {"result": "win"}},
    ]

    reducers = {
        "win_rate": Reducer(
            kind="count_true",
            source="nested.result",
            transform=lambda v: v == "win",
            normalize_by="samples",
            as_percent=True,
        )
    }

    out = apply_reducers_to_records(records, reducers)
    assert out["win_rate"] == 2.0 / 3.0


def test_apply_reducers_to_records_callable_source() -> None:
    records = [{"x": 1}, {"x": 3}]

    reducers = {"x_plus_one_mean": Reducer(kind="mean", source=lambda rec: rec["x"] + 1)}
    out = apply_reducers_to_records(records, reducers)
    assert out["x_plus_one_mean"] == 3.0


def test_apply_reducers_to_records_missing_values_are_skipped() -> None:
    records = [{"x": 1.0}, {"x": 3.0}, {}]
    reducers = {"x_mean": Reducer(kind="mean", source="x")}
    out = apply_reducers_to_records(records, reducers)
    assert out["x_mean"] == 2.0


def test_reducer_as_percent_is_display_only() -> None:
    records = [
        {"correct": True, "completion_length": 10},
        {"correct": False, "completion_length": 20},
    ]

    reducers = {
        "accuracy": Reducer(kind="count_true", source="correct", normalize_by="samples", as_percent=True),
        "avg_completion_tokens": Reducer(kind="mean", source="completion_length", as_percent=True),
    }

    out = apply_reducers_to_records(records, reducers)
    assert out["accuracy"] == 0.5
    assert out["avg_completion_tokens"] == 15.0

