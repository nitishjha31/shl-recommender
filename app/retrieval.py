"""
Loads the catalog once at process start and exposes a simple, dependency-light
retrieval function over it.

Design choice: TF-IDF + cosine similarity, not embeddings.
- The catalog is small (tens to a few hundred rows), so a heavier vector DB buys
  nothing and adds a network dependency (embedding API) that can fail mid-eval.
- TF-IDF is deterministic, has zero external calls, and is fast enough to run
  per-request inside the 30s timeout with room to spare.
- We bias the corpus text towards name + keywords (repeated) over the free-text
  description, since job-title / skill queries tend to hit keyword vocabulary
  more directly than prose descriptions do.
"""
import json
from pathlib import Path
from typing import Optional

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

CATALOG_PATH = Path(__file__).parent.parent / "data" / "catalog.json"


def _doc_text(item: dict) -> str:
    parts = [
        item.get("name", ""),
        item.get("name", ""),  # weight name x2
        " ".join(item.get("keywords", [])),
        " ".join(item.get("keywords", [])),  # weight keywords x2
        item.get("description", ""),
        " ".join(item.get("job_levels", [])),
    ]
    return " ".join(p for p in parts if p)


class CatalogIndex:
    def __init__(self, catalog_path: Path = CATALOG_PATH):
        self.items = json.loads(catalog_path.read_text())
        self._by_name_lower = {i["name"].lower(): i for i in self.items}
        self._vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self._matrix = self._vectorizer.fit_transform(_doc_text(i) for i in self.items)

    def search(
        self,
        query: str,
        top_k: int = 10,
        test_types: Optional[list[str]] = None,
        job_level: Optional[str] = None,
    ) -> list[dict]:
        if not query.strip():
            return []
        q_vec = self._vectorizer.transform([query])
        sims = cosine_similarity(q_vec, self._matrix)[0]

        scored = []
        for item, score in zip(self.items, sims):
            if score <= 0:
                continue
            if test_types and item.get("test_type") not in test_types:
                # soft filter: don't zero it out entirely, just deprioritize,
                # so a mismatched-but-relevant item can still surface if nothing else fits
                score *= 0.3
            if job_level and job_level not in item.get("job_levels", []):
                score *= 0.85
            scored.append((score, item))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [item for score, item in scored[:top_k]]

    def get_by_name(self, name: str) -> Optional[dict]:
        return self._by_name_lower.get(name.lower())

    def fuzzy_get(self, text: str) -> Optional[dict]:
        """Best-effort match of a short mention (e.g. 'OPQ', 'GSA') to a catalog item."""
        text_l = text.lower().strip()
        # exact / substring match on name first
        for item in self.items:
            if text_l == item["name"].lower():
                return item
        for item in self.items:
            if text_l in item["name"].lower() or item["name"].lower() in text_l:
                return item
        # then keyword match
        for item in self.items:
            if text_l in [k.lower() for k in item.get("keywords", [])]:
                return item
        return None


_index: Optional[CatalogIndex] = None


def get_index() -> CatalogIndex:
    global _index
    if _index is None:
        _index = CatalogIndex()
    return _index
