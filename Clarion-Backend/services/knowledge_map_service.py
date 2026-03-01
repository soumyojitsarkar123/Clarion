"""
Knowledge Map Service - extracts concepts and relationships from document chunks.
"""

import uuid
import sqlite3
import json
import re
import ast
from collections import Counter, defaultdict
from typing import List, Optional, Dict, Any
import logging

from models.chunk import Chunk
from models.knowledge_map import (
    KnowledgeMap,
    Concept,
    Relation,
    RelationType,
    MainTopic,
    Subtopic,
)
from utils.config import settings
from utils.sqlite import connect as sqlite_connect

logger = logging.getLogger(__name__)

_llm_instance = None


def get_llm_interface() -> "LLMInterface":
    """Get singleton instance of LLMInterface."""
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = LLMInterface()
    return _llm_instance


class LLMInterface:
    """
    Interface for LLM calls.
    Supports multiple providers: Ollama, OpenAI, DeepSeek, Gemini.
    """

    def __init__(self):
        self.provider = settings.llm_provider
        self.temperature = settings.llm_temperature
        self.max_tokens = settings.llm_max_tokens
        self._client = None
        self._available = None

        # Configure based on provider
        if self.provider == "ollama":
            self.model = settings.ollama_model
            self.api_base = settings.ollama_api_base
            self.api_key = "ollama"
        elif self.provider == "openai":
            self.model = settings.openai_model
            self.api_base = "https://api.openai.com/v1"
            self.api_key = settings.openai_api_key or ""
        elif self.provider == "deepseek":
            self.model = settings.deepseek_model
            self.api_base = settings.deepseek_api_base
            self.api_key = settings.deepseek_api_key or ""
        elif self.provider == "gemini":
            self.model = settings.gemini_model
            self.api_base = settings.gemini_api_base
            self.api_key = settings.gemini_api_key or ""
        else:
            self.model = settings.ollama_model
            self.api_base = settings.ollama_api_base
            self.api_key = "ollama"

    def _get_client(self):
        """Get or create OpenAI-compatible client."""
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(api_key=self.api_key, base_url=self.api_base)
        return self._client

    def _is_service_available(self) -> bool:
        """Check if the configured LLM service is available."""
        if self._available is not None:
            return bool(self._available)

        try:
            import requests

            if self.provider == "ollama":
                response = requests.get(
                    f"{self.api_base.replace('/v1', '')}/api/tags", timeout=2
                )
                self._available = response.status_code == 200
                if self._available:
                    logger.info(f"Ollama service is available ({self.model})")

            elif self.provider == "openai":
                if not self.api_key:
                    self._available = False
                else:
                    client = self._get_client()
                    client.models.list()
                    self._available = True
                    logger.info(f"OpenAI API is available ({self.model})")

            elif self.provider == "deepseek":
                if not self.api_key:
                    self._available = False
                else:
                    client = self._get_client()
                    client.models.list()
                    self._available = True
                    logger.info(f"DeepSeek API is available ({self.model})")

            elif self.provider == "gemini":
                if not self.api_key:
                    self._available = False
                else:
                    url = f"{self.api_base}/models?key={self.api_key}"
                    response = requests.get(url, timeout=2)
                    self._available = response.status_code == 200
                    if self._available:
                        logger.info(f"Gemini API is available ({self.model})")

            if not self._available:
                logger.warning(
                    f"{self.provider} service not available, using demo mode"
                )

            return bool(self._available)

        except Exception as e:
            self._available = False
            logger.warning(f"{self.provider} service error: {e}, using demo mode")
            return False

    def _call_api(self, prompt: str) -> str:
        """Call the configured LLM API."""
        try:
            client = self._get_client()

            if self.provider == "gemini":
                # Gemini uses different format
                url = f"{self.api_base}/models/{self.model}:generateContent?key={self.api_key}"
                import requests

                response = requests.post(
                    url,
                    json={
                        "contents": [{"parts": [{"text": prompt}]}],
                        "generationConfig": {
                            "temperature": self.temperature,
                            "maxOutputTokens": self.max_tokens,
                        },
                    },
                    timeout=30,
                )
                if response.status_code == 200:
                    result = response.json()
                    return (
                        result.get("candidates", [{}])[0]
                        .get("content", {})
                        .get("parts", [{}])[0]
                        .get("text", "")
                    )
                else:
                    raise Exception(f"Gemini API error: {response.text}")
            else:
                response = client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    timeout=180,
                )
                return response.choices[0].message.content or ""

        except Exception as e:
            logger.warning(f"LLM call failed: {str(e)}, falling back to demo mode")
            self._available = False
            return self._fallback_generate(prompt)

    def _fallback_generate(self, prompt: str) -> str:
        """Generate demo responses when Ollama is unavailable."""
        if "Extract key concepts" in prompt:
            # Return empty so deterministic heuristic extraction can run on real text.
            return "[]"
        elif "relationship between" in prompt:
            return '{"relation_type": null}'
        elif "Analyze relationships between these concepts" in prompt:
            return "[]"
        elif "main topics and their hierarchy" in prompt:
            return '{"main_topics": [], "subtopics": []}'
        else:
            return "LLM unavailable. Running deterministic fallback analysis from extracted document text."

    def generate(self, prompt: str) -> str:
        """
        Generate text using configured LLM or fallback to demo mode.

        Args:
            prompt: Input prompt

        Returns:
            Generated text
        """
        if not self._is_service_available():
            return self._fallback_generate(prompt)
        return self._call_api(prompt)

    def _extract_json_fragment(self, response: str) -> Optional[Any]:
        """Best-effort JSON extraction from noisy model output."""
        text = (response or "").strip()
        if not text:
            return None

        # Remove markdown code fences if present.
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
            text = text.strip()

        candidates = [text]
        first_arr = text.find("[")
        last_arr = text.rfind("]")
        if first_arr != -1 and last_arr != -1 and first_arr < last_arr:
            candidates.append(text[first_arr:last_arr + 1])
        first_obj = text.find("{")
        last_obj = text.rfind("}")
        if first_obj != -1 and last_obj != -1 and first_obj < last_obj:
            candidates.append(text[first_obj:last_obj + 1])

        for candidate in candidates:
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                continue

        # Last resort: parse python-style literals.
        for candidate in candidates:
            try:
                parsed = ast.literal_eval(candidate)
                return parsed
            except Exception:
                continue

        return None

    @staticmethod
    def _clean_concept_name(value: str) -> str:
        cleaned = re.sub(r"^[\-\*\d\.\)\s]+", "", (value or "").strip())
        cleaned = re.sub(r"\s+", " ", cleaned)
        return cleaned[:120]

    def _extract_concepts_from_text(self, response: str) -> List[Dict[str, str]]:
        """Fallback parser for non-JSON concept responses."""
        concepts: List[Dict[str, str]] = []
        for raw_line in (response or "").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            line = re.sub(r"^[\-\*\d\.\)\s]+", "", line)
            if ":" in line:
                name, definition = line.split(":", 1)
            elif " - " in line:
                name, definition = line.split(" - ", 1)
            else:
                name, definition = line, ""

            name = self._clean_concept_name(name)
            definition = definition.strip()
            if len(name) < 3:
                continue
            concepts.append({"name": name, "definition": definition})

        return concepts

    @staticmethod
    def _normalize_relation_type(value: str) -> str:
        normalized = (value or "").strip().lower()
        allowed = {
            "prerequisite",
            "definition",
            "explanation",
            "cause-effect",
            "example-of",
            "similar-to",
            "part-of",
            "derives-from",
        }
        aliases = {
            "cause_effect": "cause-effect",
            "cause effect": "cause-effect",
            "example": "example-of",
            "example_of": "example-of",
            "part_of": "part-of",
            "derive-from": "derives-from",
            "derived-from": "derives-from",
            "similar": "similar-to",
        }
        normalized = aliases.get(normalized, normalized)
        if normalized not in allowed:
            return "explanation"
        return normalized

    def extract_concepts(self, chunk_text: str) -> List[Dict[str, str]]:
        """
        Extract key concepts from chunk text.

        Args:
            chunk_text: Text to analyze

        Returns:
            List of concept dictionaries
        """
        prompt = f"""Identify the most significant knowledge concepts in this text for a mind-map visualization.
Focus on entities and ideas that represent the core subject matter.
Return a JSON array of concepts, each with 'name' (concise) and 'definition' (1 clear sentence).
Prioritize concepts that form the structural backbone of the information.

Text:
{chunk_text[:3000]}

Response (JSON only):"""

        response = self.generate(prompt)

        parsed = self._extract_json_fragment(response)
        if isinstance(parsed, list):
            concepts = []
            for item in parsed:
                if not isinstance(item, dict):
                    continue
                name = self._clean_concept_name(str(item.get("name", "")))
                definition = str(item.get("definition", "")).strip()
                if len(name) < 3:
                    continue
                concepts.append({"name": name, "definition": definition})
            if concepts:
                return concepts

        text_concepts = self._extract_concepts_from_text(response)
        if text_concepts:
            return text_concepts

        logger.warning("Failed to parse concepts JSON")
        return []

    def detect_relations(
        self, concept_a: str, concept_b: str, context: str
    ) -> Optional[Dict[str, Any]]:
        """
        Detect relationship between two concepts.

        Args:
            concept_a: First concept
            concept_b: Second concept
            context: Context text

        Returns:
            Relation dictionary or None
        """
        prompt = f"""Analyze the relationship between these two concepts:
- Concept A: {concept_a}
- Concept B: {concept_b}

Context:
{context[:1000]}

Determine if there is a meaningful relationship. If yes, return JSON with:
- "relation_type": one of: prerequisite, definition, explanation, cause-effect, example-of, similar-to, part-of, derives-from
- "description": brief description of the relationship
- "confidence": score from 0 to 1

If no meaningful relationship, return: {{"relation_type": null}}

Response (JSON only):"""

        response = self.generate(prompt)

        parsed = self._extract_json_fragment(response)
        if isinstance(parsed, dict) and parsed.get("relation_type"):
            parsed["relation_type"] = self._normalize_relation_type(
                str(parsed.get("relation_type", ""))
            )
            return parsed

        # Handle plain text fallback
        lines = (response or "").strip().split("\n")
        text_result: Dict[str, Any] = {}
        for line in lines:
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            key = key.strip().lower()
            value = value.strip()
            if key in {"relation type", "relation_type"}:
                text_result["relation_type"] = self._normalize_relation_type(value)
            elif key == "description":
                text_result["description"] = value
            elif key == "confidence":
                try:
                    text_result["confidence"] = float(value)
                except ValueError:
                    pass

        if text_result.get("relation_type"):
            return text_result

        logger.warning("Failed to parse relation JSON")
        return None

    def detect_all_relations(
        self, concepts: List[str], context: str
    ) -> List[Dict[str, Any]]:
        """
        Detect relationships between ALL concepts in a single call - optimized for free tier.

        Args:
            concepts: List of concept names
            context: Context text

        Returns:
            List of relation dictionaries
        """
        if len(concepts) < 2:
            return []

        concepts_str = ", ".join([f'"{c}"' for c in concepts])

        prompt = f"""Map out the hierarchical and semantic connections between these concepts: [{concepts_str}]

Context:
{context[:2500]}

Identify how these concepts connect to each other, prioritizing parent-child (part-of) or prerequisite links to build a clear and logical mind-map structure.
Return a JSON array of relationships with this format:
[
  {{
    "from_concept": "Parent/Topic",
    "to_concept": "Child/Sub-concept", 
    "relation_type": "part-of|prerequisite|cause-effect|example-of|similar-to|explanation",
    "description": "brief description of connection",
    "confidence": 0.85
  }}
]

Only include pairs with meaningful structural relationships. Response (JSON only):"""

        response = self.generate(prompt)

        parsed = self._extract_json_fragment(response)
        if isinstance(parsed, list):
            normalized_items: List[Dict[str, Any]] = []
            for item in parsed:
                if not isinstance(item, dict):
                    continue
                from_concept = str(item.get("from_concept", "")).strip()
                to_concept = str(item.get("to_concept", "")).strip()
                if not from_concept or not to_concept:
                    continue
                normalized_items.append(
                    {
                        "from_concept": from_concept,
                        "to_concept": to_concept,
                        "relation_type": self._normalize_relation_type(
                            str(item.get("relation_type", ""))
                        ),
                        "description": str(item.get("description", "")).strip(),
                        "confidence": float(item.get("confidence", 0.6) or 0.6),
                    }
                )
            if normalized_items:
                return normalized_items

        logger.warning("Failed to parse relations JSON")
        return []

    def identify_topics(self, chunks: List[Chunk]) -> Dict[str, Any]:
        """
        Identify main topics and subtopics from chunks.

        Args:
            chunks: Document chunks

        Returns:
            Topics dictionary
        """
        chunk_summaries = []
        step = max(1, len(chunks) // 20)
        for chunk in chunks[::step][:20]:
            summary = (
                f"Section: {chunk.section_title or 'Untitled'}\n{chunk.content[:300]}"
            )
            chunk_summaries.append(summary)

        prompt = f"""Analyze these document sections and identify the main topics and their hierarchy.
Return a JSON object with:
- "main_topics": array of main topic objects with "id", "title", "description", "subtopic_ids"
- "subtopics": array of subtopic objects with "id", "title", "parent_topic_id"

Document Sections:
{chr(10).join(chunk_summaries)}

Response (JSON only):"""

        response = self.generate(prompt)

        parsed = self._extract_json_fragment(response)
        if isinstance(parsed, dict):
            return parsed

        logger.warning("Failed to parse topics JSON")
        return {"main_topics": [], "subtopics": []}

    def generate_summary(self, title: str, chunks: List[Chunk]) -> str:
        """
        Generate structured summary.

        Args:
            title: Document title
            chunks: Document chunks

        Returns:
            Summary text
        """
        step = max(1, len(chunks) // 10)
        content = "\n\n".join(
            [
                f"--- {c.section_title or 'Section'} ---\n{c.content[:500]}"
                for c in chunks[::step][:10]
            ]
        )

        prompt = f"""Generate a comprehensive structured summary for this document titled "{title}".
Focus on the main ideas and conceptual organization, not just paraphrasing.

Document Content:
{content}

Provide a well-organized summary that reflects the conceptual hierarchy:
"""

        return self.generate(prompt)


class KnowledgeMapService:
    """
    Service for building knowledge maps from documents.
    Extracts concepts, relationships, and builds hierarchical structure.
    """

    def __init__(self):
        self.llm = LLMInterface()
        self._init_database()

    def _init_database(self) -> None:
        """Initialize SQLite database for knowledge maps."""
        db_path = settings.data_dir / "clarion.db"
        self.db_path = str(db_path)

        conn = sqlite_connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS knowledge_maps (
                document_id TEXT PRIMARY KEY,
                data TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)

        conn.commit()
        conn.close()

    def build_knowledge_map(
        self, document_id: str, chunks: List[Chunk]
    ) -> KnowledgeMap:
        """
        Build knowledge map from document chunks.

        Args:
            document_id: Document identifier
            chunks: Document chunks

        Returns:
            KnowledgeMap object
        """
        logger.info(f"Building knowledge map for document {document_id}")

        concepts = self._extract_concepts(chunks)

        logger.info(f"Extracted {len(concepts)} concepts")

        relations = self._detect_relations(concepts, chunks)

        logger.info(f"Detected {len(relations)} relations")

        topic_structure = self.llm.identify_topics(chunks)

        main_topics = self._build_main_topics(topic_structure.get("main_topics", []))
        subtopics = self._build_subtopics(topic_structure.get("subtopics", []))

        knowledge_map = KnowledgeMap(
            document_id=document_id,
            main_topics=main_topics,
            subtopics=subtopics,
            concepts=concepts,
            relations=relations,
            metadata={
                "total_concepts": len(concepts),
                "total_relations": len(relations),
                "total_topics": len(main_topics),
            },
        )

        self._save_knowledge_map(knowledge_map)

        logger.info(f"Knowledge map built successfully for {document_id}")
        return knowledge_map

    def _extract_concepts(self, chunks: List[Chunk]) -> List[Concept]:
        """Extract concepts from chunks using LLM with deterministic fallback."""
        concepts = []
        seen_names = set()

        # OPTIMIZED: Process up to 10 evenly-spaced chunks for better coverage (was 3)
        step = max(1, len(chunks) // 10)
        chunks_to_process = chunks[::step][:10]

        for chunk in chunks_to_process:
            try:
                extracted = self.llm.extract_concepts(chunk.content)

                for item in extracted:
                    name = self.llm._clean_concept_name(item.get("name", ""))
                    if not name or name.lower() in seen_names:
                        continue

                    seen_names.add(name.lower())

                    concept = Concept(
                        id=str(uuid.uuid4()),
                        name=name,
                        definition=item.get("definition"),
                        context=chunk.content[:200],
                        chunk_ids=[chunk.chunk_id],
                    )
                    concepts.append(concept)

            except Exception as e:
                logger.warning(f"Error extracting concepts from chunk: {str(e)}")
                continue

        if not concepts:
            concepts = self._extract_concepts_heuristic(chunks, seen_names=seen_names)
            if concepts:
                logger.info(
                    "Fell back to heuristic concept extraction for %d concepts",
                    len(concepts),
                )

        return concepts

    def _detect_relations(
        self, concepts: List[Concept], chunks: List[Chunk]
    ) -> List[Relation]:
        """Detect relationships between concepts with robust fallback."""
        relations = []

        # OPTIMIZED: Use context from evenly sampled chunks
        step = max(1, len(chunks) // 10)
        chunk_text = "\n\n".join([c.content for c in chunks[::step][:10]])

        # Get concept names for batch processing
        concept_names = [c.name for c in concepts[:12]]  # Limit to 12 concepts

        if len(concept_names) < 2:
            return relations

        try:
            # Single LLM call to detect all relations
            result = self.llm.detect_all_relations(concept_names, chunk_text)

            if result and isinstance(result, list):
                for item in result:
                    try:
                        from_name = str(item.get("from_concept", "")).strip()
                        to_name = str(item.get("to_concept", "")).strip()

                        if from_name and to_name:
                            relation_type = self._safe_relation_type(
                                str(item.get("relation_type", "explanation"))
                            )
                            try:
                                confidence = float(item.get("confidence", 0.8) or 0.8)
                            except Exception:
                                confidence = 0.8
                            relation = Relation(
                                id=str(uuid.uuid4()),
                                from_concept=from_name,
                                to_concept=to_name,
                                relation_type=relation_type,
                                description=item.get("description"),
                                confidence=confidence,
                            )
                            relations.append(relation)
                    except Exception:
                        continue

        except Exception as e:
            logger.warning(f"Error detecting relations: {str(e)}")

        if not relations:
            relations = self._detect_relations_by_cooccurrence(concepts, chunks)
            if relations:
                logger.info(
                    "Fell back to co-occurrence relations for %d relations",
                    len(relations),
                )

        return relations

    def _safe_relation_type(self, value: str) -> RelationType:
        normalized = (value or "").strip().lower()
        mapping = {
            "prerequisite": RelationType.PREREQUISITE,
            "definition": RelationType.DEFINITION,
            "explanation": RelationType.EXPLANATION,
            "cause-effect": RelationType.CAUSE_EFFECT,
            "cause_effect": RelationType.CAUSE_EFFECT,
            "cause effect": RelationType.CAUSE_EFFECT,
            "example-of": RelationType.EXAMPLE_OF,
            "example_of": RelationType.EXAMPLE_OF,
            "example": RelationType.EXAMPLE_OF,
            "similar-to": RelationType.SIMILAR_TO,
            "similar_to": RelationType.SIMILAR_TO,
            "similar": RelationType.SIMILAR_TO,
            "part-of": RelationType.PART_OF,
            "part_of": RelationType.PART_OF,
            "derives-from": RelationType.DERIVES_FROM,
            "derive-from": RelationType.DERIVES_FROM,
            "derived-from": RelationType.DERIVES_FROM,
        }
        return mapping.get(normalized, RelationType.EXPLANATION)

    def _extract_concepts_heuristic(
        self, chunks: List[Chunk], seen_names: Optional[set] = None
    ) -> List[Concept]:
        """Deterministic fallback concept extractor using frequent terms."""
        seen_names = seen_names or set()
        stopwords = {
            "the", "and", "for", "with", "that", "this", "from", "into", "your",
            "have", "has", "were", "was", "are", "our", "their", "which", "when",
            "where", "than", "also", "can", "will", "should", "could", "would",
            "about", "after", "before", "between", "because", "using", "used",
            "document", "section", "analysis", "system", "data", "model",
        }

        token_counter: Counter = Counter()
        token_sources: Dict[str, List[str]] = defaultdict(list)

        for chunk in chunks[:10]:
            words = re.findall(r"\b[A-Za-z][A-Za-z0-9\-]{3,}\b", chunk.content)
            for raw_word in words:
                word = raw_word.lower()
                if word in stopwords:
                    continue
                token_counter[word] += 1
                if len(token_sources[word]) < 3:
                    token_sources[word].append(chunk.chunk_id)

            if chunk.section_title:
                title = chunk.section_title.strip()
                if len(title) >= 3:
                    key = title.lower()
                    token_counter[key] += 2
                    if len(token_sources[key]) < 3:
                        token_sources[key].append(chunk.chunk_id)

        concepts: List[Concept] = []
        for token, freq in token_counter.most_common(14):
            if freq < 2:
                continue
            name = token.title()
            if name.lower() in seen_names:
                continue
            seen_names.add(name.lower())
            concepts.append(
                Concept(
                    id=str(uuid.uuid4()),
                    name=name,
                    definition=f"Frequent concept inferred from document context (frequency={freq}).",
                    context=None,
                    chunk_ids=token_sources.get(token, [])[:3],
                )
            )

        return concepts

    def _detect_relations_by_cooccurrence(
        self, concepts: List[Concept], chunks: List[Chunk]
    ) -> List[Relation]:
        """Fallback relation extraction from concept co-occurrence within chunks."""
        if len(concepts) < 2:
            return []

        chunk_text = {chunk.chunk_id: chunk.content.lower() for chunk in chunks[:20]}
        concept_to_chunks: Dict[str, set] = {}

        for concept in concepts:
            lower = concept.name.lower()
            matched = set()
            for chunk_id, content in chunk_text.items():
                if lower in content:
                    matched.add(chunk_id)
            if concept.chunk_ids:
                matched.update(concept.chunk_ids)
            concept_to_chunks[concept.name] = matched

        relations: List[Relation] = []
        seen_pairs = set()
        concept_names = [concept.name for concept in concepts[:12]]

        for i in range(len(concept_names)):
            for j in range(i + 1, len(concept_names)):
                a = concept_names[i]
                b = concept_names[j]
                overlap = concept_to_chunks[a].intersection(concept_to_chunks[b])
                if not overlap:
                    continue
                pair_key = tuple(sorted((a.lower(), b.lower())))
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)
                confidence = min(0.45 + (0.08 * len(overlap)), 0.78)
                relations.append(
                    Relation(
                        id=str(uuid.uuid4()),
                        from_concept=a,
                        to_concept=b,
                        relation_type=RelationType.EXPLANATION,
                        description="Inferred from repeated co-occurrence within the same chunks.",
                        confidence=round(confidence, 3),
                    )
                )

        return relations[:40]

    def _build_main_topics(self, topic_data: List[Dict]) -> List[MainTopic]:
        """Build main topics from LLM output."""
        topics = []

        for item in topic_data:
            topic = MainTopic(
                id=item.get("id", str(uuid.uuid4())),
                title=item.get("title", "Untitled"),
                description=item.get("description"),
                concept_ids=[],
                subtopic_ids=item.get("subtopic_ids", []),
            )
            topics.append(topic)

        return topics

    def _build_subtopics(self, subtopic_data: List[Dict]) -> List[Subtopic]:
        """Build subtopics from LLM output."""
        subtopics = []

        for item in subtopic_data:
            subtopic = Subtopic(
                id=item.get("id", str(uuid.uuid4())),
                title=item.get("title", "Untitled"),
                description=item.get("description"),
                parent_topic_id=item.get("parent_topic_id", ""),
                concept_ids=[],
            )
            subtopics.append(subtopic)

        return subtopics

    def _save_knowledge_map(self, knowledge_map: KnowledgeMap) -> None:
        """Save knowledge map to database."""
        from datetime import datetime

        conn = sqlite_connect(self.db_path)
        cursor = conn.cursor()

        data = knowledge_map.model_dump_json()

        cursor.execute(
            """
            INSERT OR REPLACE INTO knowledge_maps (document_id, data, created_at)
            VALUES (?, ?, ?)
        """,
            (knowledge_map.document_id, data, datetime.now().isoformat()),
        )

        conn.commit()
        conn.close()

    def get_knowledge_map(self, document_id: str) -> Optional[KnowledgeMap]:
        """
        Retrieve knowledge map for a document.

        Args:
            document_id: Document identifier

        Returns:
            KnowledgeMap object if found
        """
        conn = sqlite_connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            "SELECT data FROM knowledge_maps WHERE document_id = ?", (document_id,)
        )

        row = cursor.fetchone()
        conn.close()

        if not row:
            return None

        return KnowledgeMap.model_validate_json(row[0])

    def delete_knowledge_map(self, document_id: str) -> None:
        """Delete knowledge map for a document."""
        conn = sqlite_connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            "DELETE FROM knowledge_maps WHERE document_id = ?", (document_id,)
        )

        conn.commit()
        conn.close()

        logger.info(f"Deleted knowledge map for document {document_id}")
