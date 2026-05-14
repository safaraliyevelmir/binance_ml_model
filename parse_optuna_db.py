"""Dump every trial from optuna.db (params + user_attrs + value/state) to CSV."""
import argparse
from pathlib import Path

import optuna
import pandas as pd


def dump(db_path: str, study_name: str | None, out_path: str) -> None:
    storage = f"sqlite:///{db_path}"
    names = [study_name] if study_name else optuna.get_all_study_names(storage)
    if not names:
        raise SystemExit(f"No studies found in {db_path}")

    frames = []
    for name in names:
        study = optuna.load_study(study_name=name, storage=storage)
        df = study.trials_dataframe(
            attrs=("number", "value", "state", "params", "user_attrs",
                   "datetime_start", "datetime_complete", "duration"),
        )
        df.insert(0, "study_name", name)
        frames.append(df)
        print(f"  {name}: {len(df):,} trials")

    out = pd.concat(frames, ignore_index=True)
    out.columns = [c.replace("params_", "").replace("user_attrs_", "") for c in out.columns]
    out.to_csv(out_path, index=False)
    print(f"Wrote {len(out):,} rows × {len(out.columns)} cols → {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--db", default="optuna.db")
    p.add_argument("--study-name", default="dollar_bar_15M_sobol_exp4", help="Default: dump every study in the db")
    p.add_argument("--out", default="optuna_trials.csv")
    args = p.parse_args()
    if not Path(args.db).exists():
        raise SystemExit(f"DB not found: {args.db}")
    dump(args.db, args.study_name, args.out)
