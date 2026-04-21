"""
title: Memory Recall Tool
author: Danny
version: 3.2.1
description: Read-only memory recall tool for response-time personalization. Updated for Open Web UI v0.9.
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
        f"Memory Recall Tool v3.2.0: direct import unavailable ({_e}). Falling back to HTTP."
    )

try:
    import requests

    _HTTP_AVAILABLE = True
except ImportError:
    _HTTP_AVAILABLE = False


class Tools:
    class Valves(BaseModel):
        api_key: str = Field(default="", description="HTTP fallback only.")
        base_url: str = Field(
            default="http://127.0.0.1:8080", description="HTTP fallback base URL."
        )

    class UserValves(BaseModel):
        recall_k: int = Field(
            default=8,
            ge=1,
            le=64,
            description="How many relevant memories to pull before local scoring.",
        )

    def __init__(self):
        self.valves = self.Valves()
        self.user_valves = self.UserValves()
        self._current_request = None
        self.category_order = [
            "identity",
            "preference",
            "learning_style",
            "work_style",
            "technical_preference",
            "project_goal",
            "personal_fact",
            "communication_style",
            "temporary_state",
            "unknown",
        ]
        # Categories that always apply regardless of query topic
        self.always_apply = {"identity", "communication_style"}
        # Categories that apply only during teaching/explanation requests
        self.teaching_apply = {"learning_style"}
        self.teaching_signals = {
            "explain",
            "how",
            "why",
            "what is",
            "teach",
            "learn",
            "understand",
            "example",
            "step",
            "walk",
            "show",
            "define",
            "difference",
            "compare",
            "help",
            "tutorial",
            "concept",
            "meaning",
            "clarify",
        }

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

    def _detect_category(self, content: str) -> str:
        """
        Categorize a memory by its linguistic content alone.
        Heuristic-first, conservative, no hardcoded user-specific values.
        """
        t = (content or "").lower()

        # Temporary/current-state signals - check early so stale entries are surfaced correctly
        if any(
            x in t
            for x in [
                "currently",
                "right now",
                "at the moment",
                "this week",
                "this month",
                "temporary",
                "for now",
                "testing",
                "trial",
                "debug",
                "trying",
                "experiment",
            ]
        ):
            return "temporary_state"

        # Identity - who the person fundamentally is as a person
        if any(
            x in t
            for x in [
                "personality",
                "identity",
                "who the user is",
                "as a person",
                "character",
                "nature",
                "values",
                "beliefs",
                "temperament",
                "analytical",
                "creative",
                "curious",
                "honest",
                "empathetic",
                "introverted",
                "extroverted",
            ]
        ):
            return "identity"

        # Communication style - how the person likes to be spoken to
        if any(
            x in t
            for x in [
                "communication",
                "tone",
                "direct",
                "blunt",
                "formal",
                "informal",
                "no sugarcoat",
                "unsugar",
                "plain language",
                "push back",
                "concise",
                "verbose",
                "brutal honesty",
                "be upfront",
            ]
        ):
            return "communication_style"

        # Learning style - how the person learns or wants to be taught
        if any(
            x in t
            for x in [
                "learn",
                "explain",
                "teach",
                "step-by-step",
                "example",
                "mnemonic",
                "socratic",
                "outline",
                "structured notes",
                "structured",
                "markdown notes",
                "repeat until",
                "tiny steps",
                "name the rule",
            ]
        ):
            return "learning_style"

        # Work style - how the person approaches tasks and work
        if any(
            x in t
            for x in [
                "workflow",
                "work style",
                "methodical",
                "systematic",
                "organized",
                "check output",
                "verify",
                "iterate",
                "test before",
                "process",
            ]
        ):
            return "work_style"

        # Technical preferences - tools, stacks, platforms (domain-agnostic keywords)
        if any(
            x in t
            for x in [
                "software",
                "hardware",
                "programming",
                "language",
                "framework",
                "library",
                "tool",
                "platform",
                "stack",
                "code",
                "coding",
                "development",
                "database",
                "server",
                "cloud",
                "api",
                "model",
                "ai",
                "machine learning",
                "local only",
                "self-hosted",
                "open source",
            ]
        ):
            return "technical_preference"

        # Project goals - active goals, things being built or worked toward
        if any(
            x in t
            for x in [
                "goal",
                "project",
                "working on",
                "building",
                "trying to",
                "wants to",
                "planning to",
                "objective",
                "target",
                "roadmap",
            ]
        ):
            return "project_goal"

        # Personal facts - biographical/demographic/situational facts about the person
        if any(
            x in t
            for x in [
                "name is",
                "called",
                "located",
                "lives in",
                "based in",
                "from ",
                "age",
                "born",
                "occupation",
                "job",
                "works as",
                "student",
                "studies",
                "profession",
                "hobby",
                "hobbies",
                "native language",
                "speaks",
            ]
        ):
            return "personal_fact"

        # Preferences - general likes, dislikes, wants that didn't match above
        if any(
            x in t
            for x in [
                "prefer",
                "like",
                "dislike",
                "hate",
                "love",
                "want",
                "need",
                "enjoy",
                "avoid",
                "favorite",
                "interest",
                "passionate about",
            ]
        ):
            return "preference"

        return "unknown"

    def _category_rank(self, category: str) -> int:
        try:
            return self.category_order.index(category)
        except ValueError:
            return len(self.category_order)

    def _is_teaching_query(self, query: str) -> bool:
        q = query.lower()
        return any(s in q for s in self.teaching_signals)

    def _relevance_verdict(
        self, score: float, cat: str, is_teaching: bool = False
    ) -> str:
        if cat in self.always_apply:
            return "\u26a1 ALWAYS APPLY \u2014 shapes tone and framing unconditionally."
        if cat in self.teaching_apply:
            if is_teaching:
                return "\U0001f4d8 TEACHING MODE \u2014 learning style active."
            return "\U0001f537 STANDBY \u2014 apply only when explaining a concept."
        if score >= 0.15:
            return "\u2705 DIRECTLY RELEVANT \u2014 actively shape response from this."
        if score >= 0.06:
            return "\U0001f7e0 CONTEXTUALLY RELEVANT \u2014 apply if it improves the response."
        return "\u2b07 LOW RELEVANCE \u2014 skip unless no other memories apply."

    def _build_recall_report(self, scored: list, header: str) -> str:
        lines = [f"\U0001f9e0 {header}\n", "---"]
        for i, m in enumerate(scored, 1):
            lines.append(
                f"\n{i}. [{m['category'].upper()}] [id: {m['id']}] \"{m['content']}\"\n"
                f"   Score: {m['score']:.3f} | {m['verdict']}"
            )
        lines.append("\n---")
        lines.append(
            "\u26a0\ufe0f APPLY ORDER:\n"
            "  1. Apply all \u26a1 ALWAYS APPLY entries unconditionally.\n"
            "  2. If explaining a concept, apply \U0001f4d8 TEACHING MODE entries.\n"
            "  3. Apply \u2705 DIRECTLY RELEVANT and \U0001f7e0 CONTEXTUALLY RELEVANT.\n"
            "  4. Skip \u2b07 LOW RELEVANCE and \U0001f537 STANDBY unless nothing else applies.\n\n"
            "READ-ONLY: no memory was modified. No cleanup was performed. Fewer memories is not the goal."
        )
        return "\n".join(lines)

    # -- direct-import backend -------------------------------------------------

    async def _di_get_memories(self, user_id: str) -> list:
        rows = await _Memories.get_memories_by_user_id(user_id)
        return [
            {"id": getattr(m, "id", ""), "content": getattr(m, "content", "")}
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
            return [{"id": m.id, "content": m.content} for m in mems if m]
        except Exception as e:
            log.warning(
                f"Memory Recall Tool: vector search failed ({e}), using keyword fallback"
            )
            return await self._di_keyword_search(user_id, content, k)

    # -- HTTP fallback backend -------------------------------------------------

    def _http_base_url(self) -> str:
        url = getattr(self.valves, "base_url", "").strip()
        return url if url else "http://127.0.0.1:8080"

    def _http_json_hdr(self) -> dict:
        return {
            "Authorization": f"Bearer {self.valves.api_key}",
            "Content-Type": "application/json",
        }

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

    # -- unified dispatch ------------------------------------------------------

    async def _get_memories(self, uid: str) -> list:
        if _DIRECT_IMPORT_OK:
            return await self._di_get_memories(uid)
        return await asyncio.to_thread(self._http_get_memories)

    async def _query_memories(self, uid: str, content: str, k: int) -> list:
        if _DIRECT_IMPORT_OK:
            return await self._di_query_memories(uid, content, k)
        return await asyncio.to_thread(self._http_query_memories, content, k)

    # -- public tools ----------------------------------------------------------

    async def recall_relevant_memories(
        self,
        query: str,
        __user__: dict = None,
        __request__=None,
        __event_emitter__=None,
    ) -> str:
        """
        Read-only retrieval for response generation.
        Returns stored memories ranked for relevance to a specific query topic.
        Call BEFORE generating a personalized response when the topic is known.

        READ-ONLY: no memory was modified, no cleanup was performed,
        fewer memories is not the goal.

        CORRECT: recall_relevant_memories(query="python data structures")
        WRONG:   recall_relevant_memories(content="...") or recall_relevant_memories(text="...")
        """
        self._current_request = __request__
        __user__ = __user__ or {}
        uid = self._user_id(__user__)
        recall_k = self._resolve_val(__user__, "recall_k", 8)
        is_teaching = self._is_teaching_query(query)

        if __event_emitter__:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "action": "memory_recall",
                        "description": "Recalling relevant memories\u2026",
                        "done": False,
                    },
                }
            )

        query_results, all_memories = await asyncio.gather(
            self._query_memories(uid, query, recall_k),
            self._get_memories(uid),
        )

        seen_ids: set = set()
        merged = []

        for m in query_results:
            mid = m.get("id")
            if mid and mid not in seen_ids:
                seen_ids.add(mid)
                merged.append(m)

        for m in all_memories:
            mid = m.get("id")
            if not mid or mid in seen_ids:
                continue
            cat = self._detect_category(m.get("content", ""))
            if cat in self.always_apply:
                seen_ids.add(mid)
                merged.append(m)
            elif cat in self.teaching_apply and is_teaching:
                seen_ids.add(mid)
                merged.append(m)

        if not merged:
            return (
                "READ-ONLY RECALL: no memories stored yet.\n"
                "Proceed without personalization context.\n"
                "No memory was modified. No cleanup was performed. Fewer memories is not the goal."
            )

        qtok = self._tokenize(query)
        scored = []
        for m in merged:
            content = m.get("content", "")
            cat = self._detect_category(content)
            score = self._jaccard_score(qtok, self._tokenize(content))
            scored.append(
                {
                    "id": m.get("id", "?"),
                    "content": content,
                    "category": cat,
                    "score": score,
                    "verdict": self._relevance_verdict(score, cat, is_teaching),
                }
            )

        scored.sort(key=lambda x: (self._category_rank(x["category"]), -x["score"]))

        if __event_emitter__:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "action": "memory_recall",
                        "description": "Memory recall complete.",
                        "done": True,
                    },
                }
            )

        return self._build_recall_report(
            scored, f"Memory Recall Report for: {query[:80]}"
        )

    async def recall_all_memories(
        self,
        __user__: dict = None,
        __request__=None,
        __event_emitter__=None,
    ) -> str:
        """
        Read-only full recall for conversation start or broad personalization.
        Returns and prioritizes all stored memories by category.
        Use audit_memory_store() for a clean browsable inventory instead.

        READ-ONLY: no memory was modified, no cleanup was performed,
        fewer memories is not the goal.

        CORRECT: recall_all_memories()
        WRONG:   recall_all_memories(query="...") - takes no parameters
        """
        self._current_request = __request__
        __user__ = __user__ or {}
        uid = self._user_id(__user__)

        if __event_emitter__:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "action": "recall_all",
                        "description": "Loading all memories\u2026",
                        "done": False,
                    },
                }
            )

        all_memories = await self._get_memories(uid)
        if not all_memories:
            return (
                "READ-ONLY FULL RECALL: memory store is empty.\n"
                "No memory was modified. No cleanup was performed. Fewer memories is not the goal."
            )

        scored = []
        for m in all_memories:
            content = m.get("content", "")
            cat = self._detect_category(content)
            if cat in self.always_apply:
                score = 1.0
            elif cat in self.teaching_apply:
                score = 0.85
            else:
                score = 0.5
            scored.append(
                {
                    "id": m.get("id", "?"),
                    "content": content,
                    "category": cat,
                    "score": score,
                    "verdict": self._relevance_verdict(score, cat, is_teaching=True),
                }
            )

        scored.sort(key=lambda x: (self._category_rank(x["category"]), -x["score"]))

        if __event_emitter__:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "action": "recall_all",
                        "description": "Full recall complete.",
                        "done": True,
                    },
                }
            )

        return self._build_recall_report(
            scored, "Full Memory Recall Report \u2014 ALL stored memories"
        )
