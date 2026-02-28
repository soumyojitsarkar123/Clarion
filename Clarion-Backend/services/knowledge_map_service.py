"""
Knowledge Map Service - extracts concepts and relationships from document chunks.
"""

import uuid
import sqlite3
import json
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


class LLMInterface:
    """
    Interface for LLM calls.
    Uses Ollama by default (local models), with fallback to demo mode.
    """

    def __init__(self):
        self.model = settings.ollama_model
        self.api_base = settings.ollama_api_base
        self.temperature = settings.llm_temperature
        self.max_tokens = settings.llm_max_tokens
        self._client = None
        self._ollama_available = None  # Cache availability status

    def _get_client(self):
        """Get or create Ollama client."""
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(api_key="ollama", base_url=self.api_base)
        return self._client

    def _is_ollama_available(self) -> bool:
        """Check if Ollama service is available."""
        if self._ollama_available is not None:
            return self._ollama_available

        try:
            import requests

            response = requests.get(f"{self.api_base.replace('/v1', '')}/tags", timeout=2)
            self._ollama_available = response.status_code == 200
            if self._ollama_available:
                logger.info("Ollama service is available")
            return self._ollama_available
        except Exception:
            self._ollama_available = False
            logger.warning("Ollama service not available, using demo mode")
            return False

    def _call_ollama(self, prompt: str) -> str:
        """Call Ollama API."""
        try:
            client = self._get_client()

            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                timeout=30,
            )

            return response.choices[0].message.content or ""

        except Exception as e:
            logger.warning(f"Ollama call failed: {str(e)}, falling back to demo mode")
            return self._fallback_generate(prompt)

    def _fallback_generate(self, prompt: str) -> str:
        """Generate demo responses when Ollama is unavailable."""
        import random

        if "Extract key concepts" in prompt:
            concepts = [
                {"name": "Information Theory", "definition": "Mathematical study of data and communication"},
                {"name": "Entropy", "definition": "Measure of uncertainty in information"},
                {"name": "Algorithm", "definition": "Step-by-step procedure for solving a problem"},
                {"name": "Complexity Analysis", "definition": "Study of computational resource requirements"},
            ]
            return json.dumps(random.sample(concepts, min(2, len(concepts))))
        elif "relationship between" in prompt:
            relations = [
                {"relation_type": "definition", "description": "Concept A defines or explains Concept B", "confidence": 0.85},
                {"relation_type": "prerequisite", "description": "Concept A is prerequisite for Concept B", "confidence": 0.75},
                {"relation_type": "part-of", "description": "Concept A is part of Concept B", "confidence": 0.80},
            ]
            return json.dumps(random.choice(relations))
        elif "main topics and their hierarchy" in prompt:
            return json.dumps({
                "main_topics": [
                    {"id": "t1", "title": "Foundations", "description": "Core concepts", "subtopic_ids": ["s1", "s2"]},
                    {"id": "t2", "title": "Applications", "description": "Practical uses", "subtopic_ids": ["s3"]},
                ],
                "subtopics": [
                    {"id": "s1", "title": "Basics", "parent_topic_id": "t1"},
                    {"id": "s2", "title": "Theory", "parent_topic_id": "t1"},
                    {"id": "s3", "title": "Real-world Examples", "parent_topic_id": "t2"},
                ]
            })
        else:
            return "Generated summary for demonstration purposes. The system would provide full analysis with a running LLM service."

    def generate(self, prompt: str) -> str:
        """
        Generate text using Ollama or fallback to demo mode.

        Args:
            prompt: Input prompt

        Returns:
            Generated text
        """
        if not self._is_ollama_available():
            return self._fallback_generate(prompt)
        return self._call_ollama(prompt)

    def extract_concepts(self, chunk_text: str) -> List[Dict[str, str]]:
        """
        Extract key concepts from chunk text.

        Args:
            chunk_text: Text to analyze

        Returns:
            List of concept dictionaries
        """
        prompt = f"""Extract key concepts from the following text. 
Return a JSON array of concepts, each with 'name' and 'definition' fields.
If no clear concepts, return an empty array.

Text:
{chunk_text[:2000]}

Response (JSON only):"""

        response = self.generate(prompt)

        try:
            concepts = json.loads(response)
            return concepts if isinstance(concepts, list) else []
        except json.JSONDecodeError:
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

        try:
            result = json.loads(response)
            if result.get("relation_type"):
                return result
            return None
        except json.JSONDecodeError:
            logger.warning("Failed to parse relation JSON")
            return None

    def identify_topics(self, chunks: List[Chunk]) -> Dict[str, Any]:
        """
        Identify main topics and subtopics from chunks.

        Args:
            chunks: Document chunks

        Returns:
            Topics dictionary
        """
        chunk_summaries = []
        for chunk in chunks[:20]:
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

        try:
            return json.loads(response)
        except json.JSONDecodeError:
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
        content = "\n\n".join(
            [
                f"--- {c.section_title or 'Section'} ---\n{c.content[:500]}"
                for c in chunks[:10]
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
        """Extract concepts from chunks using LLM."""
        concepts = []
        seen_names = set()

        for chunk in chunks[:15]:
            try:
                extracted = self.llm.extract_concepts(chunk.content)

                for item in extracted:
                    name = item.get("name", "").strip()
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

        return concepts

    def _detect_relations(
        self, concepts: List[Concept], chunks: List[Chunk]
    ) -> List[Relation]:
        """Detect relationships between concepts."""
        relations = []

        chunk_text = "\n\n".join([c.content for c in chunks[:5]])

        for i, concept_a in enumerate(concepts):
            for concept_b in concepts[i + 1 : i + 4]:
                try:
                    result = self.llm.detect_relations(
                        concept_a.name, concept_b.name, chunk_text
                    )

                    if result and result.get("relation_type"):
                        relation = Relation(
                            id=str(uuid.uuid4()),
                            from_concept=concept_a.name,
                            to_concept=concept_b.name,
                            relation_type=RelationType(result["relation_type"]),
                            description=result.get("description"),
                            confidence=result.get("confidence", 0.8),
                        )
                        relations.append(relation)

                except Exception as e:
                    logger.warning(f"Error detecting relation: {str(e)}")
                    continue

        return relations

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
