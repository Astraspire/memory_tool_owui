"""
title: Memory Write Tool
author: Danny
version: 3.2.1
description: Mutation-only memory tool. Saves, updates, deletes, and prunes stored memories. Updated for Open Web UI v0.9.
"""

import re
import asyncio
import logging
from pydantic import BaseModel, Field

log = logging.getLogger(__name__)

try:
    from open_webui.models.memories import Memories as _Memories
    from open_webui.retrieval.vector.factory import VECTOR_DB_CLIENT as _VDB

    _DIRECT_IMPORT_OK = True
except Exception as _e:
    _DIRECT_IMPORT_OK = False
    log.warning(
        f"Memory Write Tool v3.2.0: direct import unavailable ({_e}). Falling back to HTTP."
    )

try:
    import requests

    _HTTP_AVAILABLE = True
except ImportError:
    _HTTP_AVAILABLE = False


class Tools:
    class Valves(BaseModel):
        api_key: str = Field(
            default="",
            description="HTTP fallback only. Leave blank when running inside Open WebUI.",
        )
        base_url: str = Field(
            default="http://127.0.0.1:8080",
            description="HTTP fallback base URL. Ignored when direct imports are available.",
        )

    class UserValves(BaseModel):
        duplicate_threshold: float = Field(
            default=0.75,
            ge=0.0,
            le=1.0,
            description="Jaccard similarity above which a new fact is blocked as near-duplicate.",
        )
        duplicate_check_k: int = Field(
            default=10,
            ge=1,
            le=50,
            description="How many similar memories to compare before saving.",
        )
        recall_k: int = Field(
            default=8,
            ge=1,
            le=64,
            description="How many semantically relevant memories to pull for scoring.",
        )

    def __init__(self):
        self.valves = self.Valves()
        self.user_valves = self.UserValves()
        self._current_request = None

    # -- helpers --------------------------------------------------------------

    def _user_id(self, __user__: dict) -> str:
        return __user__.get("id", "")

    def _resolve_val(self, __user__: dict, name: str, default):
        try:
            v = getattr(__user__.get("valves"), name, None)
            if v is not None:
                return type(default)(v)
        except Exception:
            pass
        return getattr(self.user_valves, name, default)

    def _tokenize(self, text: str) -> set:
        return set(re.sub(r"[^a-z0-9\s]", " ", (text or "").lower()).split())

    def _jaccard_score(self, a: set, b: set) -> float:
        if not a or not b:
            return 0.0
        return len(a & b) / len(a | b)

    def _http_base_url(self) -> str:
        url = getattr(self.valves, "base_url", "").strip()
        return url if url else "http://127.0.0.1:8080"

    def _http_json_hdr(self) -> dict:
        return {
            "Authorization": f"Bearer {self.valves.api_key}",
            "Content-Type": "application/json",
        }

    def _http_auth_hdr(self) -> dict:
        return {"Authorization": f"Bearer {self.valves.api_key}"}

    # -- direct-import backend -------------------------------------------------

    async def _di_get_memories(self, user_id: str) -> list:
        rows = await _Memories.get_memories_by_user_id(user_id)
        return [
            {
                "id": getattr(m, "id", ""),
                "content": getattr(m, "content", ""),
                "created_at": getattr(m, "created_at", None),
                "updated_at": getattr(m, "updated_at", None),
            }
            for m in (rows or [])
        ]

    async def _di_keyword_search(self, user_id: str, content: str, k: int) -> list:
        all_mems = await self._di_get_memories(user_id)
        q_tok = self._tokenize(content)
        scored = sorted(
            all_mems,
            key=lambda m: -self._jaccard_score(
                q_tok, self._tokenize(m.get("content", ""))
            ),
        )
        return scored[:k]

    async def _di_query_memories(self, user_id: str, content: str, k: int) -> list:
        try:
            req = self._current_request
            if req is None:
                return await self._di_keyword_search(user_id, content, k)
            embedding_fn = req.app.state.EMBEDDING_FUNCTION
            loop = asyncio.new_event_loop()
            try:
                vector = await loop.run_in_executor(None, embedding_fn, content)
            finally:
                loop.close()
            results = _VDB.search(
                collection_name=f"user-memory-{user_id}", vectors=[vector], limit=k
            )
            ids = []
            if results and getattr(results, "ids", None):
                ids = (
                    results.ids[0] if isinstance(results.ids[0], list) else results.ids
                )
            mems = [await _Memories.get_memory_by_id(mid) for mid in ids]
            return [
                {
                    "id": m.id,
                    "content": m.content,
                    "created_at": getattr(m, "created_at", None),
                    "updated_at": getattr(m, "updated_at", None),
                }
                for m in mems
                if m
            ]
        except Exception as e:
            log.warning(
                f"Memory Write Tool: vector search failed ({e}), using keyword fallback"
            )
            return await self._di_keyword_search(user_id, content, k)

    async def _di_save_memory(self, user_id: str, content: str):
        m = await _Memories.insert_new_memory(user_id, content)
        if m:
            try:
                req = self._current_request
                if req:
                    loop = asyncio.new_event_loop()
                    try:
                        vector = await loop.run_in_executor(
                            None, req.app.state.EMBEDDING_FUNCTION, content
                        )
                    finally:
                        loop.close()
                    _VDB.upsert(
                        collection_name=f"user-memory-{user_id}",
                        items=[
                            {
                                "id": m.id,
                                "text": content,
                                "vector": vector,
                                "metadata": {
                                    "created_at": getattr(m, "created_at", None)
                                },
                            }
                        ],
                    )
            except Exception as e:
                log.warning(f"Memory Write Tool: vector upsert failed ({e})")
        return m

    async def _di_update_memory(self, user_id: str, memory_id: str, new_content: str) -> bool:
        m = await _Memories.update_memory_by_id_and_user_id(memory_id, user_id, new_content)
        if m:
            try:
                req = self._current_request
                if req:
                    loop = asyncio.new_event_loop()
                    try:
                        vector = await loop.run_in_executor(
                            None, req.app.state.EMBEDDING_FUNCTION, new_content
                        )
                    finally:
                        loop.close()
                    _VDB.upsert(
                        collection_name=f"user-memory-{user_id}",
                        items=[
                            {
                                "id": m.id,
                                "text": new_content,
                                "vector": vector,
                                "metadata": {
                                    "updated_at": getattr(m, "updated_at", None)
                                },
                            }
                        ],
                    )
            except Exception as e:
                log.warning(f"Memory Write Tool: vector upsert on update failed ({e})")
            return True
        return False

    async def _di_delete_memory(self, user_id: str, memory_id: str) -> tuple:
        result = await _Memories.delete_memory_by_id_and_user_id(memory_id, user_id)
        if result is None:
            return (False, "not found or does not belong to this user")
        if result is False:
            return (False, "DB error during deletion")
        try:
            _VDB.delete(collection_name=f"user-memory-{user_id}", ids=[memory_id])
        except Exception as e:
            log.warning(
                f"Memory Write Tool: vector delete failed for {memory_id} ({e})"
            )
        return (True, "")

    # -- HTTP fallback backend -------------------------------------------------

    def _http_get_memories(self) -> list:
        if not _HTTP_AVAILABLE:
            return []
        try:
            r = requests.get(
                f"{self._http_base_url()}/api/v1/memories/",
                headers=self._http_json_hdr(),
                timeout=(5, 20),
            )
            r.raise_for_status()
            d = r.json()
            return d if isinstance(d, list) else d.get("items", d.get("memories", []))
        except Exception:
            return []

    def _http_query_memories(self, content: str, k: int) -> list:
        if not _HTTP_AVAILABLE:
            return []
        try:
            r = requests.post(
                f"{self._http_base_url()}/api/v1/memories/query",
                headers=self._http_json_hdr(),
                json={"content": content, "k": k},
                timeout=(5, 20),
            )
            r.raise_for_status()
            d = r.json()
            return d if isinstance(d, list) else d.get("items", d.get("memories", []))
        except Exception:
            return []

    def _http_save_memory(self, content: str):
        return requests.post(
            f"{self._http_base_url()}/api/v1/memories/add",
            headers=self._http_json_hdr(),
            json={"content": content},
            timeout=(5, 20),
        )

    def _http_update_memory(self, memory_id: str, content: str):
        return requests.post(
            f"{self._http_base_url()}/api/v1/memories/{memory_id}/update",
            headers=self._http_json_hdr(),
            json={"content": content},
            timeout=(5, 20),
        )

    def _http_delete_memory(self, memory_id: str) -> tuple:
        try:
            r = requests.delete(
                f"{self._http_base_url()}/api/v1/memories/{memory_id}",
                headers=self._http_auth_hdr(),
                timeout=(5, 20),
            )
            if not r.ok:
                return (False, f"HTTP {r.status_code}: {r.text[:200]}")
            body = r.json()
            if body is True:
                return (True, "")
            if body is None:
                return (False, "not found or wrong user (null response)")
            return (False, f"server returned: {body}")
        except Exception as e:
            return (False, str(e))

    # -- unified dispatch ------------------------------------------------------

    async def _get_memories(self, uid: str) -> list:
        if _DIRECT_IMPORT_OK:
            return await self._di_get_memories(uid)
        return await asyncio.to_thread(self._http_get_memories)

    async def _query_memories(self, uid: str, content: str, k: int) -> list:
        if _DIRECT_IMPORT_OK:
            return await self._di_query_memories(uid, content, k)
        return await asyncio.to_thread(self._http_query_memories, content, k)

    async def _save_memory(self, uid: str, content: str):
        if _DIRECT_IMPORT_OK:
            return await self._di_save_memory(uid, content)
        return await asyncio.to_thread(self._http_save_memory, content)

    async def _update_memory(self, uid: str, mid: str, content: str):
        if _DIRECT_IMPORT_OK:
            return await self._di_update_memory(uid, mid, content)
        return await asyncio.to_thread(self._http_update_memory, mid, content)

    async def _delete_memory(self, uid: str, mid: str) -> tuple:
        if _DIRECT_IMPORT_OK:
            return await self._di_delete_memory(uid, mid)
        return await asyncio.to_thread(self._http_delete_memory, mid)

    # -- public tools ----------------------------------------------------------

    async def save_key_fact(
        self,
        fact: str,
        force_save: bool = False,
        __user__: dict = None,
        __request__=None,
        __event_emitter__=None,
    ) -> str:
        """
        Save a new fact or preference about the user to persistent memory.
        Blocks near-duplicates automatically using Jaccard similarity.
        Use force_save=True to store alongside an existing similar entry.

        CORRECT: save_key_fact(fact="User prefers step-by-step explanations")
        WRONG:   save_key_fact(content="...") or save_key_fact(name="...")
        """
        self._current_request = __request__
        __user__ = __user__ or {}
        uid = self._user_id(__user__)
        threshold = self._resolve_val(__user__, "duplicate_threshold", 0.75)
        check_k = self._resolve_val(__user__, "duplicate_check_k", 10)

        if __event_emitter__:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "description": "Checking for duplicates\u2026",
                        "done": False,
                    },
                }
            )

        if not force_save:
            similar = await self._query_memories(uid, fact, check_k)
            fact_tok = self._tokenize(fact)
            for m in similar:
                existing = m.get("content", "")
                score = self._jaccard_score(fact_tok, self._tokenize(existing))
                if score >= threshold:
                    return (
                        f"\u26a0\ufe0f DUPLICATE BLOCKED (similarity: {score:.2f}, threshold: {threshold})\n"
                        f"Existing ID: {m.get('id', '?')}\n"
                        f'Existing: "{existing}"\n'
                        f'Not saved: "{fact}"\n\n'
                        "Use update_fact(old_keyword=..., new_fact=...) to replace it, "
                        "or save_key_fact(fact=..., force_save=True) to store alongside it."
                    )

        try:
            result = await self._save_memory(uid, fact)
        except Exception as e:
            return f"\u274c save_key_fact failed: {e}"

        if __event_emitter__:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {"description": "Memory saved.", "done": True},
                }
            )

        if _DIRECT_IMPORT_OK:
            return (
                f"\u2705 Saved: '{fact}'"
                if result
                else f"\u274c Failed to save: '{fact}'"
            )
        return (
            f"\u2705 Saved: '{fact}'"
            if (result and result.ok)
            else f"\u274c HTTP error saving: '{fact}'"
        )

    async def update_fact(
        self,
        old_keyword: str,
        new_fact: str,
        __user__: dict = None,
        __request__=None,
        __event_emitter__=None,
    ) -> str:
        """
        Replace an existing memory containing old_keyword with new_fact.

        :param old_keyword: Word or phrase that identifies the memory to replace.
        :param new_fact: Complete replacement text.
        """
        self._current_request = __request__
        __user__ = __user__ or {}
        uid = self._user_id(__user__)

        if __event_emitter__:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "description": f"Finding '{old_keyword}'\u2026",
                        "done": False,
                    },
                }
            )

        memories = await self._get_memories(uid)
        for m in memories:
            if old_keyword.lower() in m.get("content", "").lower():
                mid = m["id"]
                success = await self._update_memory(uid, mid, new_fact)
                if __event_emitter__:
                    await __event_emitter__(
                        {
                            "type": "status",
                            "data": {"description": "Updated.", "done": True},
                        }
                    )
                if _DIRECT_IMPORT_OK:
                    return (
                        f"\u2705 Updated [id: {mid}] to: '{new_fact}'"
                        if success
                        else f"\u274c Failed to update [id: {mid}]"
                    )
                return (
                    f"\u2705 Updated [id: {mid}] to: '{new_fact}'"
                    if (success and success.ok)
                    else f"\u274c Failed to update [id: {mid}]"
                )

        return f"No memory found containing '{old_keyword}'."

    async def delete_memory_by_keyword(
        self,
        keyword: str,
        __user__: dict = None,
        __request__=None,
        __event_emitter__=None,
    ) -> str:
        """
        Delete every stored memory whose content contains the given keyword.
        For ID-based deletion use batch_delete_memories().

        THIS FUNCTION ONLY DELETES. It does not audit or reorganize.

        :param keyword: Case-insensitive substring to match.
        """
        self._current_request = __request__
        __user__ = __user__ or {}
        uid = self._user_id(__user__)

        memories = await self._get_memories(uid)
        matches = [
            m for m in memories if keyword.lower() in m.get("content", "").lower()
        ]

        if not matches:
            return f"No memories found containing '{keyword}'."

        deleted, failed = [], []
        for m in matches:
            ok, detail = await self._delete_memory(uid, m["id"])
            if ok:
                deleted.append(f"[id: {m['id']}] \"{m.get('content', '')}\"")
            else:
                failed.append(
                    f"[id: {m['id']}] \"{m.get('content', '')}\" \u2014 {detail}"
                )

        if __event_emitter__:
            await __event_emitter__(
                {"type": "status", "data": {"description": "Done.", "done": True}}
            )

        lines = []
        if deleted:
            lines.append("\U0001f5d1\ufe0f Deleted:\n  " + "\n  ".join(deleted))
        if failed:
            lines.append("\u274c Failed:\n  " + "\n  ".join(failed))
        return "\n".join(lines) if lines else "Nothing was deleted."

    async def batch_delete_memories(
        self,
        memory_ids: list,
        __user__: dict = None,
        __request__=None,
        __event_emitter__=None,
    ) -> str:
        """
        Delete multiple memories by their IDs in one call.
        Get IDs from audit_memory_store() or analyze_memory_audit().

        THIS FUNCTION ONLY DELETES. It does not audit or reorganize.

        CORRECT: batch_delete_memories(memory_ids=["abc123", "def456"])
        WRONG:   batch_delete_memories("abc123")  - must be a list
        """
        self._current_request = __request__
        __user__ = __user__ or {}
        uid = self._user_id(__user__)

        if not memory_ids:
            return "\u26a0\ufe0f No IDs provided."
        if isinstance(memory_ids, str):
            return "\u26a0\ufe0f memory_ids must be a list, not a string."

        if __event_emitter__:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "description": f"Deleting {len(memory_ids)} memories\u2026",
                        "done": False,
                    },
                }
            )

        deleted, failed = [], []
        for mid in memory_ids:
            ok, detail = await self._delete_memory(uid, str(mid))
            if ok:
                deleted.append(str(mid))
            else:
                failed.append(f"{mid} ({detail})")

        if __event_emitter__:
            await __event_emitter__(
                {"type": "status", "data": {"description": "Done.", "done": True}}
            )

        lines = [
            f"\U0001f5d1\ufe0f Batch Delete: {len(deleted)} deleted, {len(failed)} failed (of {len(memory_ids)})"
        ]
        if deleted:
            lines.append("\u2705 Deleted IDs: " + ", ".join(deleted))
        if failed:
            lines.append("\u274c Failed: " + ", ".join(failed))
        return "\n".join(lines)

    async def semantic_prune_memories(
        self,
        topic: str,
        keep_top_k: int = 1,
        dry_run: bool = False,
        __user__: dict = None,
        __request__=None,
        __event_emitter__=None,
    ) -> str:
        """
        Retrieve memories about a topic, rank by relevance, keep the top-k, delete the rest.
        ALWAYS run with dry_run=True first to preview results.

        CORRECT workflow:
          semantic_prune_memories(topic="docker", keep_top_k=1, dry_run=True)
          semantic_prune_memories(topic="docker", keep_top_k=1, dry_run=False)

        :param topic: Short phrase describing the subject to prune.
        :param keep_top_k: How many top-scoring entries to keep (default 1).
        :param dry_run: Preview without deleting (default False).
        """
        self._current_request = __request__
        __user__ = __user__ or {}
        uid = self._user_id(__user__)
        recall_k = self._resolve_val(__user__, "recall_k", 8)

        if keep_top_k < 1:
            return "\u26a0\ufe0f keep_top_k must be \u2265 1."

        if __event_emitter__:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "description": f"Querying memories about '{topic}'\u2026",
                        "done": False,
                    },
                }
            )

        results = await self._query_memories(uid, topic, recall_k)
        if not results:
            return f"No memories found for '{topic}'."

        topic_tok = self._tokenize(topic)
        scored = sorted(
            [
                {
                    "id": m.get("id", "?"),
                    "content": m.get("content", ""),
                    "score": self._jaccard_score(
                        topic_tok, self._tokenize(m.get("content", ""))
                    ),
                }
                for m in results
            ],
            key=lambda x: -x["score"],
        )
        to_keep = scored[:keep_top_k]
        to_delete = scored[keep_top_k:]

        if not to_delete:
            return (
                f"\u2705 No pruning needed \u2014 only {len(scored)} "
                f"entries for '{topic}' (\u2264 keep_top_k={keep_top_k})."
            )

        mode = (
            "\U0001f4dd DRY RUN (no changes)" if dry_run else "\U0001f5d1\ufe0f PRUNING"
        )
        lines = [
            f"**Semantic Prune: '{topic}'** [{mode}]",
            f"Found: {len(scored)} | Keeping: {len(to_keep)} | To delete: {len(to_delete)}",
            "---",
            "\u2705 KEEPING:",
        ]
        for m in to_keep:
            lines.append(
                f"  \u2022 [id: {m['id']}] (score: {m['score']:.3f}) \"{m['content']}\""
            )
        delete_label = "[PREVIEW] WOULD DELETE" if dry_run else "[DELETE] DELETING"
        lines.append(f"\n{delete_label}:")
        for m in to_delete:
            lines.append(
                f"  \u2022 [id: {m['id']}] (score: {m['score']:.3f}) \"{m['content']}\""
            )

        if dry_run:
            lines.append(
                "\n\u26a0\ufe0f DRY RUN \u2014 nothing deleted.\n"
                f'To commit: semantic_prune_memories(topic="{topic}", keep_top_k={keep_top_k}, dry_run=False)'
            )
            return "\n".join(lines)

        deleted_ids, failed_ids = [], []
        for m in to_delete:
            ok, detail = await self._delete_memory(uid, m["id"])
            if ok:
                deleted_ids.append(m["id"])
            else:
                failed_ids.append(f"{m['id']} ({detail})")

        lines.append(f"\n\u2705 Deleted: {len(deleted_ids)}/{len(to_delete)}")
        if failed_ids:
            lines.append("\u274c Failed: " + ", ".join(failed_ids))

        if __event_emitter__:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {"description": "Prune complete.", "done": True},
                }
            )
        return "\n".join(lines)
