"""Minimal client for the openFDA drug label API (https://open.fda.gov/apis/drug/label/)."""

from __future__ import annotations

from typing import Any

import httpx

from ..chunking import parse_openfda_label
from ..models import DrugLabel

BASE_URL = "https://api.fda.gov/drug/label.json"


class OpenFDAClient:
    def __init__(self, api_key: str | None = None, timeout: float = 15.0,
                 client: httpx.Client | None = None) -> None:
        self.api_key = api_key
        self._client = client or httpx.Client(timeout=timeout)

    def _get(self, search: str, limit: int) -> list[dict[str, Any]]:
        params: dict[str, Any] = {"search": search, "limit": limit}
        if self.api_key:
            params["api_key"] = self.api_key
        resp = self._client.get(BASE_URL, params=params)
        if resp.status_code == 404:  # openFDA returns 404 when nothing matches
            return []
        resp.raise_for_status()
        return resp.json().get("results", [])

    def otc_labels_for_ingredient(self, ingredient: str, limit: int = 10) -> list[dict[str, Any]]:
        search = (
            'openfda.product_type:"HUMAN OTC DRUG" AND '
            f'openfda.generic_name:"{ingredient}"'
        )
        return self._get(search, limit)

    def label_by_set_id(self, set_id: str) -> DrugLabel | None:
        results = self._get(f'set_id:"{set_id}"', 1)
        return parse_openfda_label(results[0]) if results else None
