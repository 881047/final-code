"""训练 CLI：python scripts/train_model.py [--quick]

--quick 用精简验证跑一遍（walk-forward 报告落盘），正式训练用默认。
产出: data/models/triad.pt（SeedEnsemble 工件）+ data/reports/train_*.txt
"""
import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fterm.config import load_config
from fterm.features.build import build_dataset
from fterm.model.dataset import stack_samples
from fterm.model.train import save_ensemble, train_ensemble
from fterm.model.validate import walk_forward_validate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="walk-forward 快速验证模式")
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[1]
    cfg = load_config(root / "config.yaml")

    bundle = build_dataset(cfg)
    meta = None
    if cfg.model.task == "meta_reversal":
        meta = np.sign(bundle.features[f"rev_{cfg.signal.rev_window}"])
    samples = stack_samples(bundle.features, bundle.labels, bundle.feature_cols,
                            cfg.features.lookback, bundle.universe,
                            weight_mode=cfg.model.sample_weight,
                            max_stocks_per_day=cfg.features.max_stocks_per_day,
                            meta_primary=meta)
    print(f"样本数: {len(samples)}（{len(bundle.feature_cols)} 特征，日采样密度 "
          f"{cfg.features.max_stocks_per_day}）")
    assert len(samples) >= 200, "样本不足，先跑 scripts/sync_data.py 并确认数据量"

    reports_dir = root / "data" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M")

    if args.quick:
        rep = walk_forward_validate(cfg, samples, n_folds=3)
        text = rep.summary()
        print(text)
        (reports_dir / f"walkforward_{stamp}.txt").write_text(text, encoding="utf-8")
        return

    ens, reports = train_ensemble(cfg, samples, bundle.feature_cols)
    model_path = root / "data" / "models" / "trihead.pt"
    save_ensemble(cfg, ens, model_path, bundle.fingerprints)
    lines = [f"seed {r.seed}: val_loss={r.best_val_loss:.4f} epochs={r.epochs_run}" for r in reports]
    text = "\n".join(["训练完成 → " + str(model_path)] + lines)
    print(text)
    (reports_dir / f"train_{stamp}.txt").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
