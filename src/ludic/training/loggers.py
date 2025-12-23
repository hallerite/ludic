from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional, Protocol, Sequence

from rich.console import Console
from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
from rich.text import Text
from rich.table import Table

logger = logging.getLogger(__name__)


class TrainingLogger(Protocol):
    """
    Interface for logging training stats to arbitrary backends.

    Canonical stat prefixes:
      - train/*: training metrics (loss, rewards, reducers, counts)
      - eval/*: evaluation metrics
      - perf/*: performance counters (e.g., GPU memory)
    """

    def log(self, step: int, stats: Dict[str, float]) -> None:
        ...


class PrintLogger:
    """
    Lightweight console logger (plain stdout).
    """

    def __init__(
        self,
        *,
        prefix: str = "[trainer]",
        keys: Optional[Sequence[str]] = None,
        precision: int = 4,
        max_items_per_line: int = 6,
    ) -> None:
        self.prefix = prefix
        self.keys = list(keys) if keys is not None else None
        self.precision = precision
        self.max_items_per_line = max_items_per_line

    def _fmt_val(self, v: Any) -> str:
        if isinstance(v, float):
            return f"{v:.{self.precision}f}"
        if isinstance(v, int):
            return str(v)
        return str(v)

    def log(self, step: int, stats: Dict[str, float]) -> None:
        if self.keys is not None:
            keys = [k for k in self.keys if k in stats]
        else:
            keys = sorted(stats.keys())
        pairs = [f"{k}={self._fmt_val(stats[k])}" for k in keys]
        if not pairs:
            print(f"{self.prefix} step={step}")
            return

        header = f"{self.prefix} step={step}"
        lines = []
        for i in range(0, len(pairs), self.max_items_per_line):
            lines.append(" ".join(pairs[i : i + self.max_items_per_line]))

        print(f"{header} {lines[0]}")
        indent = " " * len(header)
        for line in lines[1:]:
            print(f"{indent} {line}")


class RichLiveLogger:
    """
    Live-updating console logger using rich.Live with a sparkline.
    """

    def __init__(
        self,
        *,
        keys: Optional[Sequence[str]] = None,
        spark_key: str = "avg_total_reward",
        history: int = 100,
        spark_window: int = 50,
        precision: int = 4,
        console: Optional[Console] = None,
        live: Optional[Live] = None,
    ) -> None:
        self.keys = list(keys) if keys is not None else None
        self.spark_key = spark_key
        self.history = history
        self.spark_window = spark_window
        self.precision = precision
        self.console = console or Console()
        self.live = live or Live(console=self.console, refresh_per_second=4, transient=False)
        self.history_vals: list[float] = []
        self._last_eval_step: Optional[int] = None
        self._own_live = live is None
        self._started = False

    def __enter__(self) -> "RichLiveLogger":
        if self._own_live:
            self.live.__enter__()
            self._started = True
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._own_live:
            self.live.__exit__(exc_type, exc, tb)

    def _fmt_val(self, v: Any) -> str:
        if isinstance(v, float):
            return f"{v:.{self.precision}f}"
        if isinstance(v, int):
            return str(v)
        return str(v)

    def _sparkline(self, max_len: Optional[int] = None) -> tuple[str, float, float]:
        if not self.history_vals:
            return "", 0.0, 0.0
        vals = self.history_vals[-self.history :]
        # Always keep a moving window of the most recent spark_window points
        if self.spark_window and self.spark_window > 0:
            vals = vals[-self.spark_window :]
        if max_len is not None and max_len > 0:
            vals = vals[-max_len:]
        lo, hi = min(vals), max(vals)
        if hi == lo:
            return "▁" * len(vals), lo, hi
        blocks = "▁▂▃▄▅▆▇█"
        scaled = []
        for v in vals:
            idx = int((v - lo) / (hi - lo) * (len(blocks) - 1))
            scaled.append(blocks[idx])
        return "".join(scaled), lo, hi

    def _render(self, step: int, stats: Dict[str, float]):

        def _make_table(items):
            table = Table(show_header=False, box=None, padding=(0, 1))
            table.add_column("metric", style="cyan", no_wrap=True)
            table.add_column("value", style="magenta", overflow="fold")
            for name, value in items:
                table.add_row(name, self._fmt_val(value))
            return table

        if self.keys is not None:
            allowed_keys = set(self.keys)
        else:
            allowed_keys = None

        train_items = []
        eval_items = []
        perf_items = []
        for k, v in sorted(stats.items()):
            if k in {"phase", "train_step", "eval_step"}:
                continue
            prefix = ""
            name = k
            if "/" in k:
                prefix, name = k.split("/", 1)
            if allowed_keys is not None and k not in allowed_keys and name not in allowed_keys:
                continue
            if prefix == "eval" or k.startswith("eval_"):
                eval_items.append((name.replace("eval_", "", 1), v))
            elif prefix == "perf":
                perf_items.append((name, v))
            else:
                train_items.append((name, v))

        train_table = _make_table(train_items) if train_items else Text("no train stats")
        eval_table = _make_table(eval_items) if eval_items else Text("no eval stats")
        perf_table = _make_table(perf_items) if perf_items else Text("no perf stats")

        train_title = "train"
        train_step = stats.get("train_step")
        if train_step is None:
            train_step = stats.get("train/step")
        if train_step is None:
            train_step = step
        try:
            train_title = f"train (@ step {int(train_step)})"
        except Exception:
            pass

        eval_title = "eval"
        if self._last_eval_step is not None:
            eval_title = f"eval (@ step {self._last_eval_step})"

        # Try to size the sparkline to the available column width to avoid ellipsis
        col_width = max(10, (self.console.width or 80) // 3 - 6)
        spark, lo, hi = self._sparkline(max_len=col_width)
        if spark:
            spark_panel = Panel(
                Text(spark, style="magenta", overflow="crop", no_wrap=True),
                title=f"{self.spark_key}",
                subtitle=f"step {step} | {lo:.{self.precision}f} – {hi:.{self.precision}f}",
                padding=(0, 1),
            )
        else:
            spark_panel = Panel(
                Text("-", style="magenta"),
                title=self.spark_key,
                subtitle=f"step {step}",
                padding=(0, 1),
            )

        root = Layout()
        top = Layout(name="top", ratio=3)
        top.split_row(
            Layout(
                Panel(train_table, title=train_title, padding=(0, 0), expand=True),
                name="train",
                ratio=1,
            ),
            Layout(
                Panel(eval_table, title=eval_title, padding=(0, 0), expand=True),
                name="eval",
                ratio=1,
            ),
            Layout(
                Panel(perf_table, title="perf", padding=(0, 0), expand=True),
                name="perf",
                ratio=1,
            ),
        )
        bottom = Layout(
            Panel(spark_panel, padding=(0, 0), expand=True),
            name="spark",
            ratio=1,
        )
        root.split_column(top, bottom)
        return root

    def log(self, step: int, stats: Dict[str, float]) -> None:
        if "eval/step" in stats:
            try:
                self._last_eval_step = int(stats["eval/step"])
            except Exception:
                self._last_eval_step = step
        elif "eval_step" in stats:
            try:
                self._last_eval_step = int(stats["eval_step"])
            except Exception:
                self._last_eval_step = step
        elif any(k.startswith("eval/") or k.startswith("eval_") for k in stats):
            self._last_eval_step = step

        spark_val = stats.get(self.spark_key)
        if spark_val is None and "/" not in self.spark_key:
            spark_val = stats.get(f"train/{self.spark_key}")
        if isinstance(spark_val, (int, float)):
            self.history_vals.append(float(spark_val))
            if len(self.history_vals) > self.history:
                self.history_vals = self.history_vals[-self.history :]

        if self._own_live and not self._started:
            self.live.start()
            self._started = True

        table = self._render(step, stats)
        self.live.update(table)


class TeeLogger:
    """
    Fan-out logger that forwards stats to multiple loggers.
    """

    def __init__(self, *loggers: TrainingLogger) -> None:
        self.loggers = [log for log in loggers if log is not None]

    def log(self, step: int, stats: Dict[str, float]) -> None:
        for log in self.loggers:
            try:
                log.log(step, stats)
            except Exception:
                logger.exception("Logger %s failed at step %s", log.__class__.__name__, step)


class WandbLogger:
    """
    Minimal Weights & Biases logger. Lazily imports wandb.
    """

    def __init__(
        self,
        *,
        init_kwargs: Optional[Dict[str, Any]] = None,
        project: Optional[str] = None,
        name: Optional[str] = None,
        group: Optional[str] = None,
        tags: Optional[Sequence[str]] = None,
        notes: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        dir: Optional[str] = None,
        mode: Optional[str] = None,
    ) -> None:
        try:
            import wandb
        except ImportError as exc:  # pragma: no cover - soft dependency
            raise ImportError("WandbLogger requires the 'wandb' package installed.") from exc

        self._wandb = wandb
        kwargs = dict(init_kwargs or {})
        kwargs.update(
            {
                k: v
                for k, v in {
                    "project": project,
                    "name": name,
                    "group": group,
                    "tags": None if tags is None else list(tags),
                    "notes": notes,
                    "config": None if config is None else dict(config),
                    "dir": dir,
                    "mode": mode,
                }.items()
                if v is not None
            }
        )
        if "project" not in kwargs and not os.environ.get("WANDB_PROJECT"):
            kwargs["project"] = "Ludic"

        self._run = wandb.init(**kwargs)

    def log(self, step: int, stats: Dict[str, float]) -> None:
        self._wandb.log(stats, step=step)

    def close(self) -> None:
        if self._run is not None:
            try:
                self._run.finish()
            except Exception:
                logger.exception("Failed to finish wandb run")
            self._run = None

    def __enter__(self) -> "WandbLogger":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
