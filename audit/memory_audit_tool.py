"""
title: Memory Audit Tool
author: Danny, Spectra
version: 3.2.4
description: Read-only memory inventory and structural quality analysis tool. Updated for Open Web UI v0.9.
"""

import re
import asyncio
import logging
from pydantic import BaseModel, Field

log = logging.getLogger(__name__)

try:
    from open_webui.models.memories import Memories as _Memories

    _DIRECT_IMPORT_OK = True
except Exception as _e:
    _DIRECT_IMPORT_OK = False
    log.warning(
        f"Memory Audit Tool v3.2.0: direct import unavailable ({_e}). Falling back to HTTP."
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
        cluster_threshold: float = Field(
            default=0.25,
            ge=0.0,
            le=1.0,
            description="Jaccard threshold for redundancy cluster grouping in inventory view.",
        )
        audit_duplicate_threshold: float = Field(
            default=0.45,
            ge=0.0,
            le=1.0,
            description="Jaccard threshold above which two memories are flagged as likely duplicates.",
        )
        audit_stale_keywords: str = Field(
            default=("nna"),
            description="Comma-separated keywords that signal a memory may be temporary or stale.",
        )
        multi_idea_clause_threshold: int = Field(
            default=3,
            ge=1,
            le=10,
            description="Clause-count score above which an entry is flagged as multi-idea.",
        )
        max_memories_to_analyze: int = Field(
            default=200,
            ge=1,
            le=500,
            description="Maximum number of memories to analyze before capping for performance.",
        )
        underspecified_min_chars: int = Field(
            default=12,
            ge=1,
            le=200,
            description="Minimum character count below which a memory is flagged as underspecified.",
        )
        min_token_length: int = Field(
            default=2,
            ge=1,
            le=10,
            description="Minimum token length for meaningful token filtering.",
        )
        max_related_ids_per_entry: int = Field(
            default=5,
            ge=0,
            le=20,
            description="Maximum number of related IDs recorded per memory entry.",
        )
        enable_redundancy_clusters: bool = Field(
            default=True,
            description="Show light redundancy clusters in the inventory view.",
        )
        user_name: str = Field(
            default="",
            description="User's name for personalized memory audit and category detection.",
        )

    def __init__(self):
        self.valves = self.Valves()
        self.user_valves = self.UserValves()
        self._current_request = None
        self.category_order = [
            "Path-SoulTone",
            "Path-PhilosophicalPrinciples",
            "Path-OperationalProtocols",
            "Path-ExtractedKnowledge",
            "Path-UserIdentityCore",
            "Path-UserProfessionalData",
            "Path-UserPersonalityTraits",
            "Path-UserInterestsDomains",
            "Path-LearningMethodology",
            "Path-HealthMedicalContext",
            "Path-SocialRelationships",
            "Path-SystemVisionGoals",
            "Path-ConversationalLogTimeline",
            "Path-DevelopmentAuditRecords",
            "Path-SystemEnvironmentState",
            "Path-GeneralContextCatchAll",
        ]

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

    def _http_base_url(self) -> str:
        url = getattr(self.valves, "base_url", "").strip()
        return url if url else "http://127.0.0.1:8080"

    def _http_json_hdr(self) -> dict:
        return {
            "Authorization": f"Bearer {self.valves.api_key}",
            "Content-Type": "application/json",
        }

    # -- backend --------------------------------------------------------------

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

    async def _get_memories(self, uid: str) -> list:
        if _DIRECT_IMPORT_OK:
            return await self._di_get_memories(uid)
        return await asyncio.to_thread(self._http_get_memories)

    # -- tokenization & scoring ------------------------------------------------

    def _tokenize(self, text: str) -> list:
        return re.sub(r"[^a-z0-9\s]", " ", (text or "").lower()).split()

    def _meaningful_tokens(self, tokens: list) -> list:
        return [t for t in tokens if len(t) >= self.user_valves.min_token_length]

    def _jaccard_score(self, a: list, b: list) -> float:
        sa, sb = set(a), set(b)
        if not sa or not sb:
            return 0.0
        return len(sa & sb) / len(sa | sb)

    # -- normalization ---------------------------------------------------------

    def _normalize_memory_record(self, raw: dict) -> dict:
        content = "" if raw.get("content") is None else str(raw.get("content"))
        # Remove all text within brackets [] to ignore category tags during duplicate detection
        import re

        content_cleaned = re.sub(r"\[.*?\]", "", content)
        return {
            "id": str(raw.get("id", "")),
            "content": content_cleaned,
            "created_at": raw.get("created_at") or None,
            "updated_at": raw.get("updated_at") or None,
            "category": "unknown",
            "tokens": self._meaningful_tokens(self._tokenize(content_cleaned)),
        }

    # -- category detection ----------------------------------------------------

    def _detect_category(self, content: str) -> str:
        """
        Categorize a memory by its linguistic content.
        1. Check for bracketed category tags first
        2. Fallback to keyword detection
        3. Return the first matching category
        """
        content_lower = (content or "").lower()
        user_name = self._resolve_val(None, "user_name", "")

        # Check for bracketed category tags
        for category in self.category_order:
            if f"[{category}]" in content_lower:
                return category

        # Keyword-based detection for each category
        # Path-SoulTone - Spectra's tone and interaction style
        # More specific to avoid over-matching
        if any(
            x in content_lower
            for x in [
                "spectra",
                "tone",
                "ai persona",
                "who spectra is",
                "identity pillar",
                "machine",
                "vibe",
                "interaction",
                "conversation",
                "agent",
                "assistant",
                "ai",
            ]
        ):
            return "Path-SoulTone"

        # Path-PhilosophicalPrinciples - Foundational why statements
        if any(
            x in content_lower
            for x in [
                "philosophy",
                "principles",
                "why",
                "logic",
                "worldview",
                "foundational",
                "deep system",
            ]
        ):
            return "Path-PhilosophicalPrinciples"

        # Path-OperationalProtocols - Rigid rules and constraints
        if any(
            x in content_lower
            for x in [
                "protocol",
                "rules",
                "constraints",
                "guidelines",
                "mandatory",
                "requirement",
                "must",
                "always",
                "never",
                "do not ever",
            ]
        ):
            return "Path-OperationalProtocols"

        # Path-ExtractedKnowledge - Spectra's learned insights
        if any(
            x in content_lower
            for x in [
                "learned",
                "insight",
                "discovery",
                "emergent",
                "personal diary",
                "knowledge",
                "understood",
            ]
        ):
            return "Path-ExtractedKnowledge"

        # Path-UserIdentityCore - User's fundamental facts
        if any(
            x in content_lower
            for x in [
                "name",
                "human",
                "user",
                "person",
                "who the user is",
                "basic trait",
                "personality",
            ]
        ):
            return "Path-UserIdentityCore"

        # Path-UserProfessionalData - Career and work info
        if any(
            x in content_lower
            for x in [
                "professional",
                "career",
                "work",
                "job",
                "occupation",
                "education",
                "skills",
                "created",
                "developed",
            ]
        ):
            return "Path-UserProfessionalData"

        # Path-UserPersonalityTraits - User's behavioral patterns
        if any(
            x in content_lower
            for x in [
                "user personality",
                "user traits",
                "user's character",
                "user temperament",
                "user behavior",
                "user pattern",
                "values",
                "prefers",
            ]
        ):
            return "Path-UserPersonalityTraits"

        # Path-UserInterestsDomains - User's hobbies and interests
        if any(
            x in content_lower
            for x in [
                "interests",
                "hobbies",
                "likes",
                "enjoys",
                "passionate",
                "favorite",
                "love",
                "dislike",
            ]
        ):
            return "Path-UserInterestsDomains"

        # Path-LearningMethodology - User's learning preferences
        if any(
            x in content_lower
            for x in [
                "learn",
                "teach",
                "instruction",
                "method",
                "approach",
                "style",
                "preference",
                "tutorial",
            ]
        ):
            return "Path-LearningMethodology"

        # Path-HealthMedicalContext - Health and medical info
        if any(
            x in content_lower
            for x in [
                "health",
                "medical",
                "doctor",
                "illness",
                "treatment",
                "wellness",
                "exercise",
                "diet",
            ]
        ):
            return "Path-HealthMedicalContext"

        # Path-SocialRelationships - Social connections
        if any(
            x in content_lower
            for x in [
                "friends",
                "family",
                "relationship",
                "partner",
                "colleague",
                "social",
                "worked with",
                "collaborated",
            ]
        ):
            return "Path-SocialRelationships"

        # Path-SystemVisionGoals - Long-term objectives
        if any(
            x in content_lower
            for x in [
                "goal",
                "vision",
                "project",
                "target",
                "objective",
                "long-term",
                "future",
                "plan",
                "want to",
            ]
        ):
            return "Path-SystemVisionGoals"

        # Path-ConversationalLogTimeline - Conversation history
        if any(
            x in content_lower
            for x in [
                "conversation",
                "chat",
                "dialogue",
                "interaction",
                "meeting",
                "session",
                "started",
                "on",
            ]
        ):
            return "Path-ConversationalLogTimeline"

        # Path-DevelopmentAuditRecords - Development tracking
        if any(
            x in content_lower
            for x in [
                "audit",
                "development",
                "code",
                "project",
                "progress",
                "tracking",
                "records",
                "log",
            ]
        ):
            return "Path-DevelopmentAuditRecords"

        # Path-SystemEnvironmentState - System configuration
        if any(
            x in content_lower
            for x in [
                "environment",
                "system",
                "state",
                "configuration",
                "setup",
                "status",
                "running",
                "using",
            ]
        ):
            return "Path-SystemEnvironmentState"

        # Path-GeneralContextCatchAll - Other information
        if any(
            x in content_lower
            for x in [
                "context",
                "information",
                "note",
                "general",
                "other",
                "additional",
            ]
        ):
            return "Path-GeneralContextCatchAll"

        return "unknown"

    # -- sorting ---------------------------------------------------------------

    def _category_rank(self, category: str) -> int:
        try:
            return self.category_order.index(category)
        except ValueError:
            return len(self.category_order)

    def _stable_sort(self, memories: list) -> list:
        return sorted(
            memories,
            key=lambda m: (
                self._category_rank(m.get("category", "unknown")),
                str(m.get("updated_at") or ""),
                str(m.get("created_at") or ""),
                str(m.get("id") or ""),
            ),
        )

    # -- audit heuristics ------------------------------------------------------

    def _audit_stale_terms(self, __user__: dict) -> list:
        raw = self._resolve_val(
            __user__, "audit_stale_keywords", self.user_valves.audit_stale_keywords
        )
        return [x.strip().lower() for x in str(raw).split(",") if x.strip()]

    def _is_probably_temporary(self, text: str, stale_terms: list) -> str:
        norm = (text or "").lower()
        hits = [t for t in stale_terms if t in norm]
        if not hits:
            return "none"
        if any(
            t in norm
            for t in ["right now", "this week", "for now", "temporary", "at the moment"]
        ):
            return "obvious_temporary"
        if any(
            t in norm for t in ["testing", "trial", "debug", "trying", "experiment"]
        ):
            return "likely_temporary"
        return "ambiguous_temporal"

    def _sentenceish_split_count(self, text: str) -> int:
        parts = re.split(r"[.;:!?]|\band\b|\bbut\b|\bor\b", text or "", flags=re.I)
        return len([p for p in parts if p.strip()])

    def _multi_idea_score(self, text: str) -> float:
        clauses = self._sentenceish_split_count(text)
        conj = len(re.findall(r"\b(and|but|or)\b", (text or "").lower()))
        punctuation = (text or "").count(";") + (text or "").count(":")
        return max(
            0.0, (clauses - 1) * 0.6 + max(0, conj - 1) * 0.3 + punctuation * 0.2
        )

    def _has_multi_idea_signals(self, text: str) -> bool:
        return self._multi_idea_score(text) >= float(
            self.user_valves.multi_idea_clause_threshold
        )

    def _find_duplicate_candidates(
        self, memory: dict, all_memories: list, threshold: float, top_k: int
    ):
        scored = []
        for other in all_memories:
            if other["id"] == memory["id"]:
                continue
            score = self._jaccard_score(memory["tokens"], other["tokens"])
            if score >= threshold:
                scored.append((other, score))
        scored.sort(key=lambda x: -x[1])
        return scored[:top_k]

    def _detect_underspecified(self, memory: dict) -> bool:
        txt = (memory.get("content") or "").strip()
        if len(txt) < self.user_valves.underspecified_min_chars:
            return True
        if len(memory.get("tokens", [])) <= 1:
            return True
        return False

    def _choose_primary_action(self, issues: list) -> str:
        """
        Precedence: DELETE > SPLIT > MERGE > REVIEW > KEEP.
        Prevents a single memory from receiving conflicting recommendations.
        """
        if "obvious_temporary" in issues:
            return "DELETE"
        if "multi-idea" in issues:
            return "SPLIT"
        if "duplicate" in issues:
            return "MERGE"
        if any(
            i in issues
            for i in ["likely_temporary", "ambiguous_temporal", "underspecified"]
        ):
            return "REVIEW"
        return "KEEP"

    def _suggest_next_step(self, action: str) -> str:
        if action == "DELETE":
            return "Use delete_memory_by_keyword or batch_delete_memories to remove this entry."
        if action == "SPLIT":
            return (
                "Rewrite as separate single-idea memories, then replace or delete "
                "the packed original using update_fact or batch_delete_memories."
            )
        if action == "MERGE":
            return (
                "Combine the overlapping memories into one clear canonical fact "
                "using update_fact, then delete the duplicate with batch_delete_memories."
            )
        if action == "REVIEW":
            return "Review manually and clarify with update_fact or save_key_fact as needed."
        return (
            "This memory already stands as a distinct durable idea. No action needed."
        )

    def _audit_single_memory(
        self, memory: dict, all_memories: list, __user__: dict
    ) -> dict:
        issues, notes = [], []
        stale_terms = self._audit_stale_terms(__user__)

        temp = self._is_probably_temporary(memory["content"], stale_terms)
        if temp != "none":
            issues.append(temp)
            notes.append(
                f"Temporal language detected ({temp.replace('_', ' ')}). May not be durable long-term."
            )

        if self._has_multi_idea_signals(memory["content"]):
            issues.append("multi-idea")
            notes.append(
                "Contains multiple separable ideas. Each idea should be its own memory."
            )

        dups = self._find_duplicate_candidates(
            memory,
            all_memories,
            self.user_valves.audit_duplicate_threshold,
            self.user_valves.max_related_ids_per_entry,
        )
        if dups:
            issues.append("duplicate")
            dup_ids = [m["id"] for m, _ in dups]
            notes.append(f"Overlaps with {len(dups)} other memory/memories: {dup_ids}.")

        if self._detect_underspecified(memory):
            issues.append("underspecified")
            notes.append(
                "Very short or thin. May not be reusable as a standalone searchable fact."
            )

        # Confidence based on evidence quality
        confidence = "high"
        if not memory.get("created_at") or not memory.get("updated_at"):
            confidence = "medium"
        if "underspecified" in issues or "ambiguous_temporal" in issues:
            confidence = "low" if confidence == "medium" else "medium"

        action = self._choose_primary_action(issues)
        why_not_keep = (
            ""
            if action == "KEEP"
            else (
                "; ".join(notes) or "Better stored as a more focused, durable memory."
            )
        )

        return {
            "id": memory["id"],
            "content": memory["content"],
            "category": memory["category"],
            "issues": issues,
            "notes": notes,
            "why_not_keep": why_not_keep,
            "recommended_action": action,
            "suggested_next_step": self._suggest_next_step(action),
            "confidence": confidence,
            "related_ids": [
                m["id"] for m, _ in dups[: self.user_valves.max_related_ids_per_entry]
            ],
        }

    # -- clustering ------------------------------------------------------------

    def _cluster_memories(self, memories: list, threshold: float) -> list:
        n = len(memories)
        if n == 0:
            return []
        adj = [set() for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                if (
                    self._jaccard_score(memories[i]["tokens"], memories[j]["tokens"])
                    >= threshold
                ):
                    adj[i].add(j)
                    adj[j].add(i)
        visited = [False] * n
        clusters = []
        for start in range(n):
            if visited[start]:
                continue
            cluster, stack = [], [start]
            visited[start] = True
            while stack:
                node = stack.pop()
                cluster.append(memories[node])
                for nb in adj[node]:
                    if not visited[nb]:
                        visited[nb] = True
                        stack.append(nb)
            if len(cluster) > 1:
                clusters.append(cluster)
        return clusters

    # -- formatting ------------------------------------------------------------

    def _format_audit_inventory(self, memories: list, clusters: list) -> str:
        lines = [
            "READ-ONLY INVENTORY",
            f"User: {self._resolve_val(None, 'user_name', '')}",
            "No memory was modified.",
            "No cleanup was performed.",
            "Fewer memories is not the goal.",
            f"Total stored memories: {len(memories)}",
            "---",
        ]
        grouped: dict = {}
        for m in memories:
            grouped.setdefault(m["category"], []).append(m)
        for cat in self.category_order:
            entries = grouped.get(cat, [])
            if not entries:
                continue
            lines.append(
                f"\n[{cat.upper()}] ({len(entries)} {'entry' if len(entries) == 1 else 'entries'})"
            )
            for m in entries:
                lines.append(f"  - [id: {m['id']}] {m['content']}")
        lines.append("")
        if clusters:
            lines.append(
                "Light redundancy clusters (read-only signal, not a deletion list):"
            )
            for i, cluster in enumerate(clusters, 1):
                lines.append(f"  Group {i} ({len(cluster)} entries)")
                for m in cluster:
                    lines.append(f"    • [id: {m['id']}] \"{m['content']}\"")
            lines.append("")
        lines.append("---")
        lines.append("NOTHING WAS CHANGED.")
        lines.append(
            "No memory was modified. No cleanup was performed. Fewer memories is not the goal.\n"
            "Use analyze_memory_audit() for structural quality judgment.\n"
            "Use the Memory Write Tool separately to make any changes."
        )
        return "\n".join(lines)

    def _format_analysis_report(
        self, results: list, totals: dict, partial: bool = False
    ) -> str:
        lines = [
            "READ-ONLY QUALITY ANALYSIS",
            "This is read-only. This is quality analysis.",
            "The goal is distinct, durable, reusable memories.",
            "Quantity is irrelevant, only quality matters. No redundancy or duplicates, only useful, concrete memories..",
            "No memory was modified. No cleanup was performed. Automatic audit will only display recommendations.",
            "---",
        ]
        if partial:
            lines.append(
                f"PARTIAL REPORT: store exceeds max_memories_to_analyze ({self.user_valves.max_memories_to_analyze}). "
                "Analysis is capped. Increase the valve limit for a full report."
            )
            lines.append("---")
        lines.extend(
            [
                f"KEEP:   {totals.get('KEEP', 0)}",
                f"REVIEW: {totals.get('REVIEW', 0)}",
                f"MERGE:  {totals.get('MERGE', 0)}",
                f"SPLIT:  {totals.get('SPLIT', 0)}",
                f"DELETE: {totals.get('DELETE', 0)}",
                "",
                f"Total analyzed:                {totals.get('total_analyzed', 0)}",
                f"Total skipped (cap):           {totals.get('total_skipped', 0)}",
                f"Duplicate-risk links found:    {totals.get('duplicate_links', 0)}",
                f"Contradiction-risk links found:{totals.get('contradiction_links', 0)}",
                "---",
            ]
        )
        for r in results:
            lines.append(
                f"\n[{r['recommended_action']}] [id: {r['id']}] ({r['category']}) — confidence: {r['confidence']}"
            )
            lines.append(f"  content:          {r['content']}")
            if r["issues"]:
                lines.append(f"  issues:           {', '.join(r['issues'])}")
            for note in r["notes"]:
                lines.append(f"  ✓ {note}")
            if r["why_not_keep"]:
                lines.append(f"  why not KEEP:     {r['why_not_keep']}")
            lines.append(f"  suggested action: {r['suggested_next_step']}")
            if r["related_ids"]:
                lines.append(f"  related ids:      {', '.join(r['related_ids'])}")
        lines.append("\n---")
        lines.append("NOTHING WAS CHANGED.")
        lines.append(
            "No memory was modified. No cleanup was performed. Fewer memories is not the goal.\n"
            "Apply changes only via the Memory Write Tool."
        )
        return "\n".join(lines)

    # -- public tools ----------------------------------------------------------

    async def audit_memory_store(
        self,
        __user__: dict = None,
        __request__=None,
        __event_emitter__=None,
    ) -> str:
        """
        READ-ONLY inventory browser for the memory store.

        - Lists every stored memory grouped by category.
        - Preserves exact stored content text - nothing is rewritten.
        - Shows light redundancy clusters as a signal, not a to-do list.
        - Does not judge long-term quality or recommend deletions in depth.
        - Does not optimize for fewer entries or any target count.
        - Does not modify, delete, update, or re-embed anything.
        - Use analyze_memory_audit() for structural quality judgment.
        - Use the Memory Write Tool separately for any changes.

        A successful memory audit does not necessarily minimize or mazimize the number of memories;
        it improves how well each memory stands alone as a durable,
        distinct, reusable long-term idea, eliminatjng redundancy along the way.

        """
        self._current_request = __request__
        __user__ = __user__ or {}
        uid = self._user_id(__user__)

        if __event_emitter__:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "description": "Starting inventory audit…",
                        "done": False,
                    },
                }
            )

        try:
            raw = await self._get_memories(uid)
        except Exception as e:
            return (
                "READ-ONLY INVENTORY ERROR: unable to load memories safely.\n"
                f"Diagnostic detail: {e}\n"
                "No memory was modified. No cleanup was performed. Fewer memories is not the goal."
            )

        if not raw:
            return (
                "READ-ONLY INVENTORY: memory store is empty.\n"
                "No memory was modified. No cleanup was performed. Fewer memories is not the goal."
            )

        normalized = []
        for r in raw:
            rec = self._normalize_memory_record(r)
            rec["category"] = self._detect_category(rec["content"])
            normalized.append(rec)

        normalized = self._stable_sort(normalized)

        clusters = []
        if self.user_valves.enable_redundancy_clusters:
            clusters = self._cluster_memories(
                normalized, self.user_valves.cluster_threshold
            )

        if __event_emitter__:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {"description": "Inventory audit complete.", "done": True},
                }
            )

        return self._format_audit_inventory(normalized, clusters)

    async def analyze_memory_audit(
        self,
        __user__: dict = None,
        __request__=None,
        __event_emitter__=None,
    ) -> str:
        """
        READ-ONLY structural quality analyzer for stored memories.

        - Evaluates distinctness, durability, duplication risk, vagueness, and packing.
        - Recommends exactly one primary action per memory: KEEP, REVIEW, MERGE, SPLIT, DELETE.
        - Does not modify any memory state.
        - Does not auto-delete, auto-merge, or rewrite anything.
        - Does not minimize toward a target count.
        - Follow-up changes must be made using the Memory Write Tool.

        A successful memory audit does not minimize the number of memories;
        it improves how well each memory stands alone as a durable,
        distinct, reusable long-term idea.
        """
        self._current_request = __request__
        __user__ = __user__ or {}
        uid = self._user_id(__user__)
        max_analyze = self.user_valves.max_memories_to_analyze

        if __event_emitter__:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {"description": "Analyzing memories…", "done": False},
                }
            )

        try:
            raw = await self._get_memories(uid)
        except Exception as e:
            return (
                "READ-ONLY QUALITY ANALYSIS ERROR: unable to load memories safely.\n"
                f"Diagnostic detail: {e}\n"
                "No memory was modified. No cleanup was performed. Fewer memories is not the goal."
            )

        if not raw:
            return (
                "READ-ONLY QUALITY ANALYSIS: memory store is empty.\n"
                "No memory was modified. No cleanup was performed. Fewer memories is not the goal."
            )

        normalized = [self._normalize_memory_record(r) for r in raw]
        for n in normalized:
            n["category"] = self._detect_category(n["content"])
        normalized = self._stable_sort(normalized)

        to_analyze = normalized[:max_analyze]
        skipped = max(0, len(normalized) - len(to_analyze))

        results = [
            self._audit_single_memory(mem, to_analyze, __user__) for mem in to_analyze
        ]

        totals: dict = {"KEEP": 0, "REVIEW": 0, "MERGE": 0, "SPLIT": 0, "DELETE": 0}
        duplicate_links = 0
        for r in results:
            totals[r["recommended_action"]] = totals.get(r["recommended_action"], 0) + 1
            duplicate_links += len(r.get("related_ids", []))

        totals["total_analyzed"] = len(results)
        totals["total_skipped"] = skipped
        totals["duplicate_links"] = duplicate_links
        totals["contradiction_links"] = 0

        if __event_emitter__:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {"description": "Memory analysis complete.", "done": True},
                }
            )

        return self._format_analysis_report(results, totals, partial=skipped > 0)
