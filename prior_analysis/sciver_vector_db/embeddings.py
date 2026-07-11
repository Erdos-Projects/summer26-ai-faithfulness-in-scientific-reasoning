from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from tqdm import tqdm


def model_cache_name(model_name: str) -> str:
    return model_name.replace("/", "__").replace(":", "_")


def normalize_array(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    if array.ndim == 1:
        denom = np.linalg.norm(array)
        return array if denom == 0 else array / denom
    denom = np.linalg.norm(array, axis=1, keepdims=True)
    denom[denom == 0] = 1.0
    return array / denom


def text_cache_key(model_name: str, text: str) -> str:
    payload = json.dumps({"model": model_name, "text": text}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def image_cache_key(model_name: str, path: Path) -> str:
    stat = path.stat()
    payload = json.dumps(
        {
            "model": model_name,
            "path": str(path.resolve()),
            "mtime_ns": stat.st_mtime_ns,
            "size": stat.st_size,
        },
        sort_keys=True,
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


@dataclass
class TextEmbedder:
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    cache_root: Path = Path("data/processed/embedding_cache")
    batch_size: int = 64

    def __post_init__(self) -> None:
        self.cache_dir = self.cache_root / "text" / model_cache_name(self.model_name)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    def encode(self, texts: Iterable[str]) -> dict[str, np.ndarray]:
        unique_texts = list(dict.fromkeys(texts))
        result: dict[str, np.ndarray] = {}
        missing: list[str] = []
        for text in unique_texts:
            cache_path = self.cache_dir / f"{text_cache_key(self.model_name, text)}.npy"
            if cache_path.exists():
                result[text] = np.load(cache_path)
            else:
                missing.append(text)
        if missing:
            for start in tqdm(range(0, len(missing), self.batch_size), desc="Embedding text"):
                batch = missing[start : start + self.batch_size]
                vectors = self.model.encode(
                    batch,
                    batch_size=self.batch_size,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                ).astype(np.float32)
                for text, vector in zip(batch, vectors):
                    vector = normalize_array(vector)
                    np.save(self.cache_dir / f"{text_cache_key(self.model_name, text)}.npy", vector)
                    result[text] = vector
        return result


@dataclass
class ImageEmbedder:
    model_name: str = "openai/clip-vit-base-patch32"
    cache_root: Path = Path("data/processed/embedding_cache")
    batch_size: int = 16
    device: str | None = None

    def __post_init__(self) -> None:
        self.cache_dir = self.cache_root / "image" / model_cache_name(self.model_name)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._processor = None
        self._model = None
        self._device = self.device

    @property
    def processor(self):
        if self._processor is None:
            try:
                from transformers import AutoImageProcessor

                self._processor = AutoImageProcessor.from_pretrained(self.model_name)
            except Exception:
                try:
                    from transformers import CLIPImageProcessor

                    self._processor = CLIPImageProcessor.from_pretrained(self.model_name)
                except Exception:
                    from transformers import CLIPImageProcessor

                    self._processor = CLIPImageProcessor()
        return self._processor

    @property
    def model(self):
        if self._model is None:
            import torch
            from transformers import CLIPModel

            if self._device is None:
                self._device = "cuda" if torch.cuda.is_available() else "cpu"
            self._model = CLIPModel.from_pretrained(self.model_name).to(self._device)
            self._model.eval()
        return self._model

    def encode(self, paths: Iterable[Path]) -> tuple[dict[Path, np.ndarray], dict[Path, str]]:
        unique_paths = list(dict.fromkeys(Path(path) for path in paths))
        result: dict[Path, np.ndarray] = {}
        failures: dict[Path, str] = {}
        missing: list[Path] = []
        for path in unique_paths:
            cache_path = self.cache_dir / f"{image_cache_key(self.model_name, path)}.npy"
            if cache_path.exists():
                result[path] = np.load(cache_path)
            else:
                missing.append(path)
        if missing:
            from PIL import Image
            import torch

            for start in tqdm(range(0, len(missing), self.batch_size), desc="Embedding images"):
                batch_paths = missing[start : start + self.batch_size]
                images = []
                valid_paths: list[Path] = []
                for path in batch_paths:
                    try:
                        images.append(Image.open(path).convert("RGB"))
                        valid_paths.append(path)
                    except Exception as exc:  # noqa: BLE001 - report per-file image failures.
                        failures[path] = str(exc)
                if not images:
                    continue
                try:
                    inputs = self.processor(images=images, return_tensors="pt")
                    inputs = {key: value.to(self.model.device) for key, value in inputs.items()}
                    with torch.no_grad():
                        vectors = self.model.get_image_features(**inputs)
                    if hasattr(vectors, "image_embeds"):
                        vectors = vectors.image_embeds
                    elif hasattr(vectors, "pooler_output"):
                        vectors = vectors.pooler_output
                    elif isinstance(vectors, (tuple, list)):
                        vectors = vectors[0]
                    vectors = vectors.detach().cpu().numpy().astype(np.float32)
                    vectors = normalize_array(vectors)
                    for path, vector in zip(valid_paths, vectors):
                        np.save(self.cache_dir / f"{image_cache_key(self.model_name, path)}.npy", vector)
                        result[path] = vector
                except Exception as exc:  # noqa: BLE001 - batch failure is actionable in report.
                    for path in valid_paths:
                        failures[path] = str(exc)
        return result, failures
