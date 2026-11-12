"""RxNorm name normalization through NLM's RxNav REST API (https://lhncbc.nlm.nih.gov/RxNav/).

Note: RxNav's drug-drug interaction API was discontinued in January 2024, so interaction
checks in this project are done against label text instead (see safety.py).
"""

from __future__ import annotations

from functools import lru_cache

import httpx

from ..models import RxNormConcept

BASE_URL = "https://rxnav.nlm.nih.gov/REST"
INGREDIENT_TTYS = {"IN", "PIN", "MIN"}


class RxNavClient:
    def __init__(self, timeout: float = 15.0, client: httpx.Client | None = None) -> None:
        self._client = client or httpx.Client(timeout=timeout)
        self.normalize = lru_cache(maxsize=512)(self._normalize)

    def _json(self, path: str, **params: str) -> dict:
        resp = self._client.get(f"{BASE_URL}{path}", params=params)
        resp.raise_for_status()
        return resp.json()

    def _find_rxcui(self, name: str) -> str | None:
        ids = self._json("/rxcui.json", name=name, search="2").get("idGroup", {})
        if ids.get("rxnormId"):
            return ids["rxnormId"][0]
        approx = self._json("/approximateTerm.json", term=name, maxEntries="1")
        candidates = approx.get("approximateGroup", {}).get("candidate") or []
        return candidates[0].get("rxcui") if candidates else None

    def _normalize(self, name: str) -> RxNormConcept:
        name = name.strip()
        if not name:
            return RxNormConcept(input_name=name, found=False, message="Empty drug name")
        try:
            rxcui = self._find_rxcui(name)
            if not rxcui:
                return RxNormConcept(input_name=name, found=False,
                                     message="No RxNorm match for this name")
            props = self._json(f"/rxcui/{rxcui}/properties.json").get("properties") or {}
            tty, matched = props.get("tty"), props.get("name")
            if tty in INGREDIENT_TTYS and matched:
                ingredients = [matched.lower()]
            else:
                related = self._json(f"/rxcui/{rxcui}/related.json", tty="IN")
                groups = related.get("relatedGroup", {}).get("conceptGroup") or []
                ingredients = sorted({
                    p["name"].lower()
                    for g in groups
                    for p in (g.get("conceptProperties") or [])
                    if p.get("name")
                })
            return RxNormConcept(input_name=name, found=True, rxcui=rxcui, matched_name=matched,
                                 term_type=tty, ingredients=ingredients)
        except httpx.HTTPError as exc:
            return RxNormConcept(input_name=name, found=False,
                                 message=f"RxNav request failed: {exc.__class__.__name__}")
