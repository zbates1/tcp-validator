"""
Experiment tracker for bore/needle-tip fine-tuning runs.

Usage in notebook / script:
    from experiment_tracker import Experiment

    exp = Experiment({
        "model":      "facebook/dinov2-small",
        "datasets":   ["cam1_z3.75x"],
        "n_train":    85,
        "n_val":      15,
        "epochs":     30,
        "lr":         3e-4,
        "batch_size": 16,
        "note":       "baseline frozen backbone",
    })

    for epoch in range(1, EPOCHS + 1):
        # ... training loop ...
        exp.log(epoch, {"train_loss": t_loss, "val_loss": v_loss, "val_px": v_px})

    exp.save_model(head, "dinov2_head.pt")
    exp.finish()

    # Print a table of all past runs:
    Experiment.summary()
"""
import json, os, time
from datetime import datetime
from pathlib import Path


class Experiment:
    def __init__(self, config: dict, base_dir="experiments"):
        base_dir = Path(base_dir)
        base_dir.mkdir(exist_ok=True)

        # Auto-increment run id
        existing = sorted(
            int(p.name.split("_")[1]) for p in base_dir.iterdir()
            if p.is_dir() and p.name.startswith("run_")
        ) if any(base_dir.iterdir()) else []
        run_num = (existing[-1] + 1) if existing else 1

        self.run_id  = f"run_{run_num:03d}"
        self.run_dir = base_dir / self.run_id
        self.run_dir.mkdir()

        self.config   = dict(config)
        self.config["run_id"]    = self.run_id
        self.config["timestamp"] = datetime.now().isoformat(timespec="seconds")
        self._start   = time.time()
        self.results  = []

        with open(self.run_dir / "config.json", "w") as f:
            json.dump(self.config, f, indent=2)

        note = self.config.get("note", "")
        print(f"[{self.run_id}] started  {self.config['timestamp']}"
              + (f"  — {note}" if note else ""))

    def log(self, epoch: int, metrics: dict):
        """Record per-epoch metrics."""
        row = {"epoch": epoch, **metrics}
        self.results.append(row)
        # Pretty one-liner
        parts = "  ".join(f"{k}={v:.5f}" if isinstance(v, float) else f"{k}={v}"
                          for k, v in metrics.items())
        print(f"  ep {epoch:03d}  {parts}")
        self._save_results()

    def save_model(self, model, name="model.pt"):
        """Save model weights into the run folder."""
        import torch
        path = self.run_dir / name
        torch.save(model.state_dict(), path)
        print(f"  model saved → {path}")

    def finish(self, metrics: dict = None):
        """Mark the run done; optionally record final summary metrics."""
        elapsed = time.time() - self._start
        summary = {
            "elapsed_s": round(elapsed, 1),
            **(metrics or {}),
        }
        if self.results:
            last = self.results[-1]
            summary.setdefault("final_val_px",   last.get("val_px"))
            summary.setdefault("final_val_loss",  last.get("val_loss"))
        with open(self.run_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        print(f"[{self.run_id}] done  elapsed={elapsed:.0f}s"
              + (f"  val_px={summary['final_val_px']:.1f}" if summary.get("final_val_px") else ""))

    def _save_results(self):
        with open(self.run_dir / "results.json", "w") as f:
            json.dump(self.results, f, indent=2)

    # ── Class-level utilities ─────────────────────────────────────────────────

    @staticmethod
    def summary(base_dir="experiments"):
        """Print a table of all completed runs."""
        base_dir = Path(base_dir)
        if not base_dir.exists():
            print("No experiments yet.")
            return

        runs = sorted(p for p in base_dir.iterdir() if p.is_dir() and p.name.startswith("run_"))
        if not runs:
            print("No runs found.")
            return

        rows = []
        for run_dir in runs:
            cfg_path = run_dir / "config.json"
            sum_path = run_dir / "summary.json"
            res_path = run_dir / "results.json"
            if not cfg_path.exists():
                continue
            with open(cfg_path) as f:
                cfg = json.load(f)
            summ = json.load(open(sum_path)) if sum_path.exists() else {}
            # Best val_px from results
            best_px = None
            if res_path.exists():
                res = json.load(open(res_path))
                pxs = [r["val_px"] for r in res if "val_px" in r]
                best_px = min(pxs) if pxs else None

            rows.append({
                "run":       cfg.get("run_id", run_dir.name),
                "date":      cfg.get("timestamp", "")[:10],
                "model":     cfg.get("model", "")[-20:],
                "datasets":  ",".join(cfg.get("datasets", [])),
                "n_train":   cfg.get("n_train", "?"),
                "epochs":    cfg.get("epochs", "?"),
                "lr":        cfg.get("lr", "?"),
                "best_px":   f"{best_px:.2f}" if best_px else summ.get("final_val_px", "—"),
                "note":      cfg.get("note", ""),
            })

        # Column widths
        cols = ["run", "date", "model", "datasets", "n_train", "epochs", "lr", "best_px", "note"]
        widths = {c: max(len(c), max((len(str(r[c])) for r in rows), default=0)) for c in cols}
        header = "  ".join(c.ljust(widths[c]) for c in cols)
        sep    = "  ".join("-" * widths[c] for c in cols)
        print(header)
        print(sep)
        for r in rows:
            print("  ".join(str(r[c]).ljust(widths[c]) for c in cols))

    @staticmethod
    def load(run_id: str, base_dir="experiments"):
        """Load config + results for a past run."""
        run_dir = Path(base_dir) / run_id
        cfg  = json.load(open(run_dir / "config.json"))
        res  = json.load(open(run_dir / "results.json")) if (run_dir / "results.json").exists() else []
        return cfg, res
