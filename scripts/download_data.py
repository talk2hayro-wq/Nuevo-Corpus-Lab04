from __future__ import annotations

import io
import os
import sys
from pathlib import Path

import pandas as pd
import requests

from inf8239_u02.config import ROOT, settings
from inf8239_u02.data import sha256

# Splits opcionales (validation/test) definidos en .env; el principal es DATASET_URL.
EXTRA_SPLITS = {
    "validation": "DATASET_URL_VALIDATION",
    "test": "DATASET_URL_TEST",
}


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def fetch(url: str) -> bytes:
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    return response.content


def save(content: bytes, url: str, output: Path) -> None:
    """Guarda la copia original y, si la fuente es Parquet, una versión CSV
    en `output` para que load_dataset() la lea sin cambios."""
    output.parent.mkdir(parents=True, exist_ok=True)
    if ".parquet" in url.lower():
        raw = output.with_suffix(".parquet")
        raw.write_bytes(content)
        print(f"Original: {raw}  SHA-256: {sha256(raw)}")
        df = pd.read_parquet(io.BytesIO(content))
        df.to_csv(output, index=False, encoding="utf-8")
    else:
        output.write_bytes(content)
    print(f"Guardado: {output}  filas={len(pd.read_csv(output))}  SHA-256: {sha256(output)}")


def main() -> int:
    if settings.data_source == "local":
        print(f"Modo local. Archivo configurado: {settings.dataset_path}")
        return 0
    if settings.data_source != "url" or not settings.dataset_url:
        print("Configure DATA_SOURCE=url y DATASET_URL en .env", file=sys.stderr)
        return 2

    main_output = resolve(settings.dataset_path)
    save(fetch(settings.dataset_url), settings.dataset_url, main_output)

    for split, env_var in EXTRA_SPLITS.items():
        url = os.getenv(env_var)
        if not url:
            continue
        output = main_output.with_name(f"{main_output.stem}_{split}{main_output.suffix}")
        save(fetch(url), url, output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
