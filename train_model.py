"""Train the optional, local custom gesture model from independent sessions."""

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys
from tempfile import NamedTemporaryFile

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from utils.custom_gestures import (
    ACTION_LABELS,
    DESTRUCTIVE,
    LABELS,
    MODEL_LIMIT,
    MODEL_NAME,
    PREPROCESSING,
    SCHEMA,
    CustomGestureRecognizer,
    Sample,
    custom_eligible,
)
from utils.dataset import DatasetError, load_training_data
from utils.model_assets import data_dir

_REJECT = "REJECT"


def _fit_model(training: list[tuple[str, np.ndarray]]) -> Pipeline:
    labels, features = zip(*training)
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(class_weight="balanced", max_iter=2000, random_state=42)),
    ])
    return pipeline.fit(np.stack(features), labels)


def _parameters(pipeline: Pipeline) -> dict:
    scaler = pipeline.named_steps["scaler"]
    classifier = pipeline.named_steps["classifier"]
    return {
        "schema": SCHEMA,
        "preprocessing": PREPROCESSING,
        "classes": classifier.classes_.tolist(),
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "coef": classifier.coef_.tolist(),
        "intercept": classifier.intercept_.tolist(),
    }


def evaluate_holdout(
    recognizer: CustomGestureRecognizer, rows: list[tuple[str, Sample]]
) -> Counter[tuple[str, str]]:
    confusion = Counter()
    for label, sample in rows:
        prediction = recognizer.predict(sample)
        decided = prediction.label if prediction is not None and custom_eligible(prediction, sample) else _REJECT
        confusion[label, decided] += 1
    return confusion


def _report_holdout(confusion: Counter[tuple[str, str]]) -> None:
    decisions = (*ACTION_LABELS, _REJECT)
    print("Held-out decisions after runtime custom gates (OTHER is correctly rejected):")
    print(f"{'Actual':>12} {'Support':>8}" + "".join(f"{label:>12}" for label in decisions))
    for label in LABELS:
        counts = [confusion[label, decision] for decision in decisions]
        print(f"{label:>12} {sum(counts):>8}" + "".join(f"{count:>12}" for count in counts))


def _install_model(path: Path, parameters: dict) -> None:
    content = json.dumps(parameters, separators=(",", ":"), allow_nan=False).encode("utf-8")
    if len(content) > MODEL_LIMIT:
        raise DatasetError("custom model exceeds the 256 KiB runtime limit")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as file:
            temporary = Path(file.name)
            file.write(content)
            file.flush()
            os.fsync(file.fileno())
        temporary.replace(path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def train_pipeline(dataset_root: Path, model_path: Path, legacy_csv: Path | None = None) -> Pipeline:
    """Fit and install only after a held-out session passes the custom safety gates."""
    data = load_training_data(dataset_root, legacy_csv)
    print(
        f"Held-out session: {data.holdout}; {len(data.validation)} validation rows, "
        f"{len(data.training)} training rows ({data.legacy_count} legacy), "
        f"{data.duplicates} duplicate rows skipped."
    )
    pipeline = _fit_model(data.training)
    parameters = _parameters(pipeline)
    portable = CustomGestureRecognizer.from_parameters(parameters)
    confusion = evaluate_holdout(portable, data.validation)
    _report_holdout(confusion)
    zero_correct = [label for label in ACTION_LABELS if not confusion[label, label]]
    destructive_other = sum(confusion["OTHER", label] for label in DESTRUCTIVE)
    failures = []
    if zero_correct:
        failures.append("no correct held-out decisions for " + ", ".join(zero_correct))
    if destructive_other:
        failures.append(f"{destructive_other} held-out OTHER samples produced destructive decisions")
    if failures:
        raise DatasetError("; ".join(failures))
    _install_model(model_path, parameters)
    print(f"Installed custom gesture model: {model_path}")
    return pipeline


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-csv", type=Path, help="optional legacy CSV with a matching .parity.json receipt")
    args = parser.parse_args(argv)
    try:
        root = data_dir()
        train_pipeline(root / "custom-data", root / "models" / MODEL_NAME, args.legacy_csv)
    except (OSError, ValueError) as exc:
        print(f"Custom training failed: {exc}. Installed model not changed.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())