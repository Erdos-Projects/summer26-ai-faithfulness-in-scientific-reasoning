from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

import numpy as np


def point_uuid(human_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, human_id))


def vector_list(vector: np.ndarray | list[float]) -> list[float]:
    return np.asarray(vector, dtype=np.float32).tolist()


def make_client(qdrant_path: str | Path | None = None, qdrant_url: str | None = None):
    from qdrant_client import QdrantClient

    if qdrant_url:
        return QdrantClient(url=qdrant_url)
    if qdrant_path is None:
        raise ValueError("Either qdrant_path or qdrant_url is required")
    return QdrantClient(path=str(qdrant_path))


def collection_exists(client: Any, collection_name: str) -> bool:
    try:
        return bool(client.collection_exists(collection_name))
    except Exception:
        try:
            client.get_collection(collection_name)
            return True
        except Exception:
            return False


def recreate_collection(
    client: Any,
    collection_name: str,
    vectors_config: dict[str, Any],
    *,
    reset: bool,
) -> None:
    from qdrant_client import models

    if reset and collection_exists(client, collection_name):
        client.delete_collection(collection_name)
    if not collection_exists(client, collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config={
                name: models.VectorParams(size=size, distance=models.Distance.COSINE)
                for name, size in vectors_config.items()
            },
        )


def create_payload_indexes(client: Any, collection_name: str, fields: dict[str, str]) -> None:
    from qdrant_client import models

    schema_map = {
        "keyword": models.PayloadSchemaType.KEYWORD,
        "integer": models.PayloadSchemaType.INTEGER,
        "bool": models.PayloadSchemaType.BOOL,
    }
    for field_name, schema_name in fields.items():
        try:
            client.create_payload_index(
                collection_name=collection_name,
                field_name=field_name,
                field_schema=schema_map[schema_name],
            )
        except Exception:
            # Index creation is idempotent across qdrant-client versions; existing indexes may raise.
            pass


def scroll_all(client: Any, collection_name: str, *, with_vectors: bool = False, batch_size: int = 256):
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=collection_name,
            limit=batch_size,
            offset=offset,
            with_payload=True,
            with_vectors=with_vectors,
        )
        for point in points:
            yield point
        if offset is None:
            break


def query_points(
    client: Any,
    collection_name: str,
    *,
    vector_name: str,
    vector: list[float],
    limit: int,
    query_filter: Any | None = None,
):
    try:
        response = client.query_points(
            collection_name=collection_name,
            query=vector,
            using=vector_name,
            limit=limit,
            query_filter=query_filter,
            with_payload=True,
        )
        return response.points
    except Exception:
        return client.search(
            collection_name=collection_name,
            query_vector=(vector_name, vector),
            limit=limit,
            query_filter=query_filter,
            with_payload=True,
        )
