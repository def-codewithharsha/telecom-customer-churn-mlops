"""Persist sklearn models without pickle-based deserialization."""

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import skops.io as sio

from telecom_churn.exception import TelecomChurnError

ARTIFACT_SCHEMA_VERSION = 1
TRUSTED_SKOPS_TYPES = (
    "numpy.dtype",
    "sklearn.compose._column_transformer.make_column_selector",
    # These sklearn calibration internals and CV splitter are part of the
    # fitted, version-pinned CalibratedClassifierCV artifact.
    "sklearn.calibration._CalibratedClassifier",
    "sklearn.calibration._SigmoidCalibration",
    "sklearn.model_selection._split.StratifiedKFold",
    "telecom_churn.features.add_service_count_features",
)
_HASH_CHUNK_SIZE = 1024 * 1024


def metadata_path_for(model_path: Path) -> Path:
    """Return the JSON metadata path paired with a serialized model."""
    return model_path.with_suffix(".json")


def save_model_artifact(
    model: Any,
    model_path: Path,
    metadata: dict[str, Any],
) -> str:
    """Save a model and metadata with an integrity digest and skops type check."""
    model_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path = metadata_path_for(model_path)
    model_temp = model_path.with_name(f"{model_path.name}.tmp")
    metadata_temp = metadata_path.with_name(f"{metadata_path.name}.tmp")

    complete_metadata = {
        **metadata,
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
    }
    _validate_metadata(complete_metadata)

    try:
        sio.dump(model, model_temp)
        untrusted_types = set(sio.get_untrusted_types(file=model_temp))
        unapproved_types = sorted(untrusted_types - set(TRUSTED_SKOPS_TYPES))
        if unapproved_types:
            raise TelecomChurnError(
                f"Model contains unreviewed serialized types: {unapproved_types}"
            )

        complete_metadata["model_sha256"] = _sha256_file(model_temp)
        complete_metadata["artifact_sha256"] = _sha256_artifact(
            model_temp, complete_metadata
        )
        _validate_metadata(complete_metadata, require_digests=True)

        with metadata_temp.open("w", encoding="utf-8") as file:
            json.dump(complete_metadata, file, indent=2, allow_nan=False)
            file.write("\n")

        os.replace(model_temp, model_path)
        os.replace(metadata_temp, metadata_path)
        return complete_metadata["artifact_sha256"]
    except Exception as exc:
        model_temp.unlink(missing_ok=True)
        metadata_temp.unlink(missing_ok=True)
        if isinstance(exc, TelecomChurnError):
            raise
        raise TelecomChurnError(
            f"Could not save model artifact to {model_path}: {exc}"
        ) from exc


def load_model_artifact(model_path: Path) -> tuple[Any, dict[str, Any]]:
    """Verify artifact digests and allowed types before deserializing the model."""
    metadata_path = metadata_path_for(model_path)
    if not model_path.is_file():
        raise TelecomChurnError(f"Model artifact not found at {model_path}.")
    if not metadata_path.is_file():
        raise TelecomChurnError(f"Model metadata not found at {metadata_path}.")

    try:
        with metadata_path.open("r", encoding="utf-8") as file:
            metadata = json.load(file)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise TelecomChurnError(
            f"Could not read model metadata from {metadata_path}: {exc}"
        ) from exc

    if not isinstance(metadata, dict):
        raise TelecomChurnError("Model metadata must be a JSON object.")
    _validate_metadata(metadata, require_digests=True)

    actual_model_sha256 = _sha256_file(model_path)
    if actual_model_sha256 != metadata["model_sha256"]:
        raise TelecomChurnError("Model artifact checksum does not match its metadata.")

    actual_artifact_sha256 = _sha256_artifact(model_path, metadata)
    if actual_artifact_sha256 != metadata["artifact_sha256"]:
        raise TelecomChurnError("Model artifact digest does not match its metadata.")

    expected_digest = os.getenv("MODEL_ARTIFACT_SHA256")
    if expected_digest is not None:
        expected_digest = expected_digest.strip().lower()
        if expected_digest != actual_artifact_sha256:
            raise TelecomChurnError(
                "Model artifact digest does not match MODEL_ARTIFACT_SHA256."
            )

    try:
        untrusted_types = set(sio.get_untrusted_types(file=model_path))
        unapproved_types = sorted(untrusted_types - set(TRUSTED_SKOPS_TYPES))
        if unapproved_types:
            raise TelecomChurnError(
                f"Model contains unreviewed serialized types: {unapproved_types}"
            )
        model = sio.load(model_path, trusted=list(TRUSTED_SKOPS_TYPES))
    except Exception as exc:
        if isinstance(exc, TelecomChurnError):
            raise
        raise TelecomChurnError(
            f"Could not load model artifact from {model_path}: {exc}"
        ) from exc

    return model, metadata


def _sha256_file(path: Path) -> str:
    """Compute a SHA-256 digest without reading the entire file into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(_HASH_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_artifact(model_path: Path, metadata: dict[str, Any]) -> str:
    """Hash model bytes and canonical metadata, excluding the digest itself."""
    digest = hashlib.sha256()
    with model_path.open("rb") as file:
        for chunk in iter(lambda: file.read(_HASH_CHUNK_SIZE), b""):
            digest.update(chunk)

    metadata_for_digest = {
        key: value for key, value in metadata.items() if key != "artifact_sha256"
    }
    canonical_metadata = json.dumps(
        metadata_for_digest,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    digest.update(b"\0")
    digest.update(canonical_metadata)
    return digest.hexdigest()


def _validate_metadata(
    metadata: dict[str, Any],
    *,
    require_digests: bool = False,
) -> None:
    """Reject unsupported or incomplete model artifact metadata."""
    if metadata.get("artifact_schema_version") != ARTIFACT_SCHEMA_VERSION:
        raise TelecomChurnError(
            "Unsupported model artifact schema version: "
            f"{metadata.get('artifact_schema_version')!r}"
        )

    for field in ("model_name", "model_version", "target_column"):
        value = metadata.get(field)
        if not isinstance(value, str) or not value.strip():
            raise TelecomChurnError(f"Model metadata has an invalid {field}.")

    threshold = metadata.get("threshold")
    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, (int, float))
        or not 0.0 <= threshold <= 1.0
    ):
        raise TelecomChurnError("Model metadata has an invalid prediction threshold.")

    if require_digests:
        for field in ("model_sha256", "artifact_sha256"):
            digest = metadata.get(field)
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise TelecomChurnError(f"Model metadata has an invalid {field}.")
