"""Manual Linux MRMR benchmark; not collected by pytest, no timing thresholds.

Run against the installed package (or PYTHONPATH=src for the checkout):
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
        python tests/mrmr_performance.py OUTPUT_DIR --adult-data adult.data
Download/extract adult.data from https://archive.ics.uci.edu/static/public/2/adult.zip.
Use a new OUTPUT_DIR for each run. --self-check needs no downloaded data.
Add --sampling-validation for a fixed Adult train/holdout sampling comparison.
"""

import argparse
import cProfile
import gc
import hashlib
import importlib.metadata
import io
import json
import os
import platform
import pstats
import resource
import subprocess
import sys
import time
from functools import partial
from pathlib import Path

SEED = 20261016
GIB = 1024**3
BUDGET = 4 * GIB
STOP_RSS = int(0.9 * BUDGET)
HEADROOM = 512 * 1024**2
THREADS = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
ADULT_COLUMNS = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education-num",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
    "native-country",
    "income",
]


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def proc_bytes(pid):
    status = Path(f"/proc/{pid}/status").read_text().splitlines()
    return {
        key: int(line.split()[1]) * 1024
        for line in status
        for key in ("VmRSS", "VmHWM")
        if line.startswith(key + ":")
    }


def available_bytes():
    """Honor both host availability and finite ancestor cgroup v2 limits."""
    available = next(
        int(line.split()[1]) * 1024
        for line in Path("/proc/meminfo").read_text().splitlines()
        if line.startswith("MemAvailable:")
    )
    relative = next(
        line.split(":", 2)[2]
        for line in Path("/proc/self/cgroup").read_text().splitlines()
        if line.startswith("0::")
    )
    root = Path("/sys/fs/cgroup")
    group = root / relative.lstrip("/")
    for parent in (group, *group.parents):
        if parent == root.parent:
            break
        limit = parent / "memory.max"
        current = parent / "memory.current"
        if limit.exists() and current.exists() and limit.read_text().strip() != "max":
            available = min(
                available, int(limit.read_text()) - int(current.read_text())
            )
    return available


def imports():
    # The supervisor stays small; only fresh workers import the scientific stack.
    global np, pd, a, MRMR
    import numpy as np
    import pandas as pd

    from arfs import association as a
    from arfs.feature_selection import MinRedundancyMaxRelevance as MRMR


def selector(count, metric="theil"):
    kwargs = {}
    if metric == "cramer":
        kwargs["redundancy_func"] = partial(
            a.association_series, nom_nom_assoc=a.cramer_v, n_jobs=1, normalize=False
        )
    return MRMR(count, task="classification", n_jobs=1, show_progress=False, **kwargs)


def synthetic(width, mixed, rows=2000):
    rng = np.random.default_rng(SEED)
    n_cat = width // 10 if mixed else 0
    X = pd.DataFrame(
        rng.normal(size=(rows, width - n_cat)),
        columns=[f"num_{i}" for i in range(width - n_cat)],
    )
    if n_cat:
        categories = pd.DataFrame(
            {
                f"cat_{i}": pd.Categorical(rng.integers(0, 20, size=rows))
                for i in range(n_cat)
            }
        )
        X = pd.concat([X, categories], axis=1)
    y = pd.Series(
        (X.iloc[:, 0] + rng.normal(size=rows) > 0).astype(int), dtype="category"
    )
    return X, y


def adult(path):
    frame = pd.read_csv(
        path, header=None, names=ADULT_COLUMNS, skipinitialspace=True, na_values="?"
    )
    original_rows = len(frame)
    frame = frame.dropna().reset_index(drop=True)
    for name in frame.select_dtypes("object"):
        frame[name] = frame[name].astype("category")
    return frame.drop(columns="income"), frame.income, original_rows


def fingerprint(frame):
    return hashlib.sha256(
        pd.util.hash_pandas_object(frame, index=True).values.tobytes()
    ).hexdigest()


def self_check():
    from pandas.testing import assert_frame_equal, assert_series_equal

    X, y = synthetic(30, True, rows=100)
    before, before_y = X.copy(deep=True), y.copy(deep=True)
    calls = []

    def counted(x, y, sample_weight=None, as_frame=False):
        calls.append((x.name, y.name))
        return a.cramer_v(x, y, sample_weight, as_frame)

    for metric in ("theil", "cramer"):
        models = [selector(5, metric).fit(X, y) for _ in range(2)]
        assert models[0].selected_features_ == models[1].selected_features_
        assert len(models[0].selected_features_) == 5
        assert models[0].get_support().sum() == 5
        assert_frame_equal(models[0].ranking_, models[1].ranking_)
        assert_frame_equal(X, before)
        assert_series_equal(y, before_y)
    # Force categorical comparisons, independent of the generated relevance order.
    cats = X.filter(like="cat_")
    callback = partial(
        a.association_series, nom_nom_assoc=counted, n_jobs=1, normalize=False
    )
    model = MRMR(
        2, task="classification", redundancy_func=callback, show_progress=False
    ).fit(cats, y)
    assert calls and len(model.selected_features_) == 2
    assert set(calls) == {(c, model.selected_features_[0]) for c in cats.columns}
    assert_frame_equal(cats, before[cats.columns])
    assert_series_equal(y, before_y)
    X2, y2 = synthetic(30, True, rows=100)
    assert_frame_equal(X2, before)
    assert_series_equal(y2, before_y)
    print("Self-check passed: callback, counts, determinism, input preservation.")


def worker(args):
    imports()
    from threadpoolctl import threadpool_info

    pools = threadpool_info()
    assert all(pool["num_threads"] == 1 for pool in pools), pools
    if args.worker == "self-check":
        self_check()
        return
    baseline = proc_bytes(os.getpid())["VmRSS"]
    if args.worker.startswith("adult"):
        X, y, original_rows = adult(args.adult_data)
        count = 10
    else:
        X, y = synthetic(args.width, args.worker == "mixed")
        original_rows, count = len(X), 20
    before, before_y = fingerprint(X), fingerprint(y)
    record = {
        "case": args.worker,
        "pid": os.getpid(),
        "shape": list(X.shape),
        "rows_before_dropna": original_rows,
        "categorical_features": len(X.select_dtypes("category").columns),
        "input_bytes": int(
            X.memory_usage(index=True, deep=True).sum()
            + y.memory_usage(index=True, deep=True)
        ),
        "rss_after_imports_bytes": baseline,
        "rss_before_fit_bytes": proc_bytes(os.getpid())["VmRSS"],
        "input_sha256": before,
        "target_sha256": before_y,
        "threadpools": pools,
    }
    write_json(
        args.result, record
    )  # Preserve input evidence if monitoring stops the fit.
    metric = "cramer" if args.worker.endswith("cramer") else "theil"
    if args.worker.startswith("adult-timing"):
        warm = selector(count, metric).fit(X, y)
        expected = warm.selected_features_.copy()
        del warm
        gc.collect()
        seconds = []
        for _ in range(5):
            start = time.perf_counter()
            model = selector(count, metric).fit(X, y)
            seconds.append(time.perf_counter() - start)
            assert model.selected_features_ == expected
            selected = model.selected_features_.copy()
            table_bytes = int(
                model.redundancy.memory_usage(index=True, deep=True).sum()
            )
            del model
            gc.collect()
    else:
        profiler = (
            cProfile.Profile() if args.worker.startswith("adult-profile") else None
        )
        if profiler:
            warm = selector(count, metric).fit(X, y)
            expected = warm.selected_features_.copy()
            del warm
            gc.collect()
        model = selector(count, metric)
        start = time.perf_counter()
        if profiler:
            profiler.enable()
        model.fit(X, y)
        if profiler:
            profiler.disable()
        seconds = [time.perf_counter() - start]
        selected = model.selected_features_.copy()
        table_bytes = int(model.redundancy.memory_usage(index=True, deep=True).sum())
        if profiler:
            assert selected == expected
            profiler.dump_stats(str(args.result) + ".prof")
            stream = io.StringIO()
            stats = (
                pstats.Stats(profiler, stream=stream)
                .strip_dirs()
                .sort_stats("cumulative")
            )
            stats.print_stats(40)
            stats.print_stats(
                "cramer_v|crosstab|pivot_table|chi2_contingency|groupby|factorize"
            )
            Path(str(args.result) + ".profile.txt").write_text(stream.getvalue())
            record["profile_functions"] = [
                {
                    "file": Path(file).name,
                    "line": line,
                    "function": fn,
                    "primitive_calls": cc,
                    "calls": nc,
                    "self_seconds": tt,
                    "cumulative_seconds": ct,
                }
                for (file, line, fn), (cc, nc, tt, ct, _) in stats.stats.items()
                if fn
                in {
                    "fit",
                    "cramer_v",
                    "crosstab",
                    "pivot_table",
                    "chi2_contingency",
                    "factorize",
                    "_conditional_entropies",
                }
            ]
    assert len(selected) == count
    assert fingerprint(X) == before and fingerprint(y) == before_y
    record.update(
        status="complete",
        seconds=seconds,
        selected_features=selected,
        redundancy_table_bytes=table_bytes,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
    )
    write_json(args.result, record)


def provenance():
    imports()
    import inspect

    import arfs.parallel
    import arfs.utils

    paths = [
        Path(inspect.getfile(module)).resolve()
        for module in (a, sys.modules[MRMR.__module__], arfs.parallel, arfs.utils)
    ]
    paths.append(Path(__file__).resolve())
    root = Path(__file__).resolve().parents[1]
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "cpu": next(
            line.split(":", 1)[1].strip()
            for line in Path("/proc/cpuinfo").read_text().splitlines()
            if line.startswith("model name")
        ),
        "versions": {
            name: importlib.metadata.version(name)
            for name in (
                "arfs",
                "numpy",
                "pandas",
                "scipy",
                "scikit-learn",
                "joblib",
                "threadpoolctl",
                "lightgbm",
                "shap",
                "matplotlib",
                "statsmodels",
                "tqdm",
            )
        },
        "sources": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths
        },
        "git_head": subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
        ).strip(),
        "git_status": subprocess.check_output(
            ["git", "-C", str(root), "status", "--short"], text=True
        ),
        "uv_lock_sha256": hashlib.sha256((root / "uv.lock").read_bytes()).hexdigest(),
    }


def run(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    # Capture provenance in its own process, keeping the monitoring parent small.
    meta_path = output / "provenance.json"
    subprocess.run(
        [
            sys.executable,
            __file__,
            "--worker",
            "provenance",
            "--result",
            str(meta_path),
        ],
        check=True,
    )
    record = {
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "provenance": json.loads(meta_path.read_text()),
        "settings": {
            "seed": SEED,
            "budget_bytes": BUDGET,
            "stop_rss_bytes": STOP_RSS,
            "minimum_available_bytes": HEADROOM,
            "monitor_interval_seconds": 0.05,
            "n_jobs": 1,
            "native_threads": {name: os.environ[name] for name in THREADS},
            "sample_weight": None,
            "task": "classification",
            "denominator": "numpy.mean",
            "normalize": False,
            "synthetic_rows": 2000,
            "synthetic_categories": 20,
            "synthetic_mixed_fraction": 0.1,
            "adult_select": 10,
            "synthetic_select": 20,
            "timing_repetitions": 5,
            "timing_warmups": 1,
            "profile_warmups": 1,
        },
        "adult": {
            "source": "https://archive.ics.uci.edu/dataset/2/adult",
            "download": "https://archive.ics.uci.edu/static/public/2/adult.zip",
            "file": "adult.data",
            "sha256": hashlib.sha256(args.adult_data.read_bytes()).hexdigest(),
        },
        "results": [],
    }
    cases = [(f"adult-timing-{metric}", 0) for metric in ("theil", "cramer")]
    cases += [(f"adult-profile-{metric}", 0) for metric in ("theil", "cramer")]
    cases += [
        (kind, width)
        for width in (1000, 5000, 10000, 20000)
        for kind in ("numeric", "mixed")
    ]
    for kind, width in cases:
        name = f"{kind}-{width}" if width else kind
        available = available_bytes()
        # ponytail: heuristic input/copy allowance, refine only if a skipped case is needed.
        forecast = 8 * width**2 + 3 * 2000 * width * 8 + HEADROOM
        result = {
            "name": name,
            "available_before_bytes": available,
            "forecast_bytes": forecast,
            "dense_values_lower_bound_bytes": 8 * width**2,
        }
        if forecast >= min(STOP_RSS, available - HEADROOM):
            result.update(
                status="skipped",
                reason="Heuristic preflight approaches RSS budget or machine headroom.",
            )
        else:
            path = output / f"{name}.json"
            cmd = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker",
                kind,
                "--width",
                str(width),
                "--adult-data",
                str(args.adult_data.resolve()),
                "--result",
                str(path),
            ]
            stop_reason, observed_peak = None, 0
            with (output / f"{name}.log").open("w") as log:
                process = subprocess.Popen(cmd, stdout=log, stderr=log)
                while process.poll() is None:
                    try:
                        rss = proc_bytes(process.pid)
                    except FileNotFoundError:
                        continue
                    observed_peak = max(observed_peak, rss.get("VmHWM", 0))
                    if rss.get("VmRSS", 0) >= STOP_RSS or available_bytes() < HEADROOM:
                        stop_reason = "RSS approached budget or available machine headroom fell below reserve."
                        process.terminate()
                        try:
                            process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait()
                        break
                    time.sleep(0.05)
            if path.exists():
                result.update(json.loads(path.read_text()))
            result.update(
                returncode=process.returncode, monitored_peak_rss_bytes=observed_peak
            )
            if stop_reason or process.returncode:
                result.update(
                    status="stopped" if stop_reason else "failed",
                    reason=stop_reason or "Worker failed; see case log.",
                )
        record["results"].append(result)
        write_json(output / "results.json", record)
        print(
            name,
            result["status"],
            result.get("seconds", result.get("reason")),
            flush=True,
        )


def sampling_validation(args):
    """Compare opt-in sampling against full-training-row fits; never tune on holdout."""
    imports()
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import log_loss
    from sklearn.model_selection import train_test_split
    from threadpoolctl import threadpool_info

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    pools = threadpool_info()
    assert all(pool["num_threads"] == 1 for pool in pools)
    X, y, original_rows = adult(args.adult_data)
    before, before_y = fingerprint(X), fingerprint(y)
    train, holdout = train_test_split(
        np.arange(len(X)), test_size=0.2, stratify=y, random_state=SEED
    )
    X_train, y_train = X.iloc[train], y.iloc[train]
    X_test, y_test = X.iloc[holdout], y.iloc[holdout]
    categories = X_train.select_dtypes("category").columns
    learner_settings = {
        "categorical_features": "from_dtype",
        "max_iter": 100,
        "max_leaf_nodes": 15,
        "l2_regularization": 1,
        "early_stopping": False,
        "random_state": 0,
    }

    def loss(selected):
        # Keep input column order so tied learner splits do not depend on MRMR order.
        columns = [c for c in X_train if c in selected]
        learner = HistGradientBoostingClassifier(**learner_settings).fit(
            X_train[columns], y_train
        )
        return float(
            log_loss(
                y_test, learner.predict_proba(X_test[columns]), labels=learner.classes_
            )
        )

    def associations(frame, metric):
        kwargs = {"nom_nom_assoc": a.cramer_v} if metric == "cramer" else {}
        values = (
            a.association_matrix(frame, n_jobs=1, **kwargs)
            .set_index(["row", "col"])
            .val
        )
        assert values.index.is_unique
        return values

    # ponytail: one fixed holdout and learner, use repeated outer splits before generalizing.
    record = {
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "provenance": provenance(),
        "threadpools": pools,
        "dataset_sha256": hashlib.sha256(args.adult_data.read_bytes()).hexdigest(),
        "input_sha256": before,
        "target_sha256": before_y,
        "rows_before_dropna": original_rows,
        "complete_rows": len(X),
        "train_rows": len(train),
        "holdout_rows": len(holdout),
        "train_positions_sha256": hashlib.sha256(train.tobytes()).hexdigest(),
        "holdout_positions_sha256": hashlib.sha256(holdout.tobytes()).hexdigest(),
        "settings": {
            "split_seed": SEED,
            "sample_sizes": [5000, 10000],
            "seeds": list(range(5)),
            "n_features_to_select": 10,
            "sample_weight": None,
            "n_jobs": 1,
            "threads": {name: os.environ[name] for name in THREADS},
            "learner": learner_settings,
        },
        "all_features_log_loss": loss(X_train.columns),
        "full": {},
        "sampled": [],
    }
    for metric in ("theil", "cramer"):
        warm = selector(10, metric).fit(X_train, y_train)
        full = selector(10, metric)
        start = time.perf_counter()
        full.fit(X_train, y_train)
        seconds = time.perf_counter() - start
        assert warm.selected_features_ == full.selected_features_
        full_assoc = associations(X_train, metric)
        record["full"][metric] = {
            "seconds": seconds,
            "selected": full.selected_features_,
            "log_loss": loss(full.selected_features_),
            "relevance": full.relevance_.to_dict(),
        }
        for cap in record["settings"]["sample_sizes"]:
            for seed in record["settings"]["seeds"]:
                model = selector(10, metric).set_params(
                    sample_size=cap, random_state=seed
                )
                start = time.perf_counter()
                model.fit(X_train, y_train)
                seconds = time.perf_counter() - start
                rows = model.sample_indices_
                sampled = X_train.iloc[rows]
                sample_assoc = associations(sampled, metric).reindex(full_assoc.index)
                assert sample_assoc.notna().all()
                error = (sample_assoc - full_assoc).abs()
                selected, reference = (
                    set(model.selected_features_),
                    set(full.selected_features_),
                )
                result = {
                    "metric": metric,
                    "sample_size": cap,
                    "seed": seed,
                    "seconds": seconds,
                    "selected": model.selected_features_,
                    "log_loss": loss(selected),
                    "overlap_count": len(selected & reference),
                    "jaccard": len(selected & reference) / len(selected | reference),
                    "relevance_rank_correlation": float(
                        model.relevance_.corr(full.relevance_, method="spearman")
                    ),
                    "association_mean_abs_error": float(error.mean()),
                    "association_max_abs_error": float(error.max()),
                    "class_proportions": y_train.iloc[rows]
                    .value_counts(normalize=True)
                    .to_dict(),
                    "missing_predictor_levels": {
                        c: sorted(set(X_train[c].unique()) - set(sampled[c].unique()))
                        for c in categories
                    },
                    "sample_positions_sha256": hashlib.sha256(
                        rows.tobytes()
                    ).hexdigest(),
                }
                assert len(selected) == 10 and len(rows) == cap
                assert len(np.unique(rows)) == cap
                assert y_train.iloc[rows].nunique() == y_train.nunique()
                record["sampled"].append(result)
                write_json(output / "sampling.json", record)
                print(
                    metric,
                    cap,
                    seed,
                    "overlap",
                    result["overlap_count"],
                    "log_loss",
                    result["log_loss"],
                    flush=True,
                )
    assert fingerprint(X) == before and fingerprint(y) == before_y
    record["input_preserved"] = True
    record["train_class_proportions"] = y_train.value_counts(normalize=True).to_dict()
    write_json(output / "sampling.json", record)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, nargs="?")
    parser.add_argument("--adult-data", type=Path)
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--sampling-validation", action="store_true")
    parser.add_argument(
        "--worker",
        choices=(
            "self-check",
            "provenance",
            "adult-timing-theil",
            "adult-timing-cramer",
            "adult-profile-theil",
            "adult-profile-cramer",
            "numeric",
            "mixed",
        ),
    )
    parser.add_argument("--width", type=int, default=0)
    parser.add_argument("--result", type=Path)
    args = parser.parse_args()
    for name in THREADS:
        if os.environ.get(name) != "1":
            parser.error(f"Set {name}=1 before starting Python")
    if sys.platform != "linux":
        parser.error("This manual runner measures Linux /proc RSS in bytes.")
    if args.self_check:
        imports()
        self_check()
    elif args.worker == "provenance":
        write_json(args.result, provenance())
    elif args.worker:
        worker(args)
    elif args.output is None or args.adult_data is None:
        parser.error("Supply OUTPUT_DIR and --adult-data, or --self-check")
    elif args.sampling_validation:
        sampling_validation(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
