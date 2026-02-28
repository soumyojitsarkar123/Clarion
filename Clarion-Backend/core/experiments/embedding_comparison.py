"""
Embedding Comparison Experiment Module

Runs relation validation experiments with different embedding models
and compares results.
"""

import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

from utils.config import settings

logger = logging.getLogger(__name__)


EMBEDDING_MODELS = {
    "minilm": {
        "name": "sentence-transformers/all-MiniLM-L6-v2",
        "display": "MiniLM-L6-v2",
        "description": "Lightweight baseline embedding model"
    },
    "bge-base": {
        "name": "BAAI/bge-base-en-v1.5",
        "display": "BGE-Base",
        "description": "BGE base embedding model"
    },
    "bge-large": {
        "name": "BAAI/bge-large-en-v1.5",
        "display": "BGE-Large",
        "description": "BGE large embedding model (stronger)"
    }
}


class EmbeddingComparator:
    """
    Runs relation validation experiments with different embedding models.
    """
    
    def __init__(self, dataset_path: Optional[Path] = None):
        self.dataset_path = dataset_path or (settings.data_dir / "relation_dataset.db")
        self.results: Dict[str, Any] = {}
    
    def load_labeled_data(self) -> List[Dict]:
        """Load labeled relation records."""
        if not self.dataset_path.exists():
            logger.warning(f"Dataset not found at {self.dataset_path}")
            return []
        
        conn = sqlite3.connect(str(self.dataset_path))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM relation_dataset 
            WHERE is_valid IS NOT NULL
        """)
        
        rows = cursor.fetchall()
        conn.close()
        
        records = []
        for row in rows:
            records.append({
                "record_id": row["record_id"],
                "document_id": row["document_id"],
                "relation_id": row["relation_id"],
                "concept_a": row["concept_a"],
                "concept_b": row["concept_b"],
                "relation_type": row["relation_type"],
                "llm_confidence": row["llm_confidence"],
                "cooccurrence_score": row["cooccurrence_score"],
                "semantic_similarity": row["semantic_similarity"],
                "chunk_context": row["chunk_context"],
                "is_valid": bool(row["is_valid"])
            })
        
        return records
    
    def compute_embedding_similarity(
        self,
        concept_a: str,
        concept_b: str,
        model_key: str
    ) -> Optional[float]:
        """
        Compute semantic similarity using embedding model.
        
        Args:
            concept_a: First concept
            concept_b: Second concept
            model_key: Embedding model key
        
        Returns:
            Similarity score (0-1) or None
        """
        try:
            from sentence_transformers import SentenceTransformer
            
            model_info = EMBEDDING_MODELS.get(model_key, {})
            model_name = model_info.get("name", EMBEDDING_MODELS["minilm"]["name"])
            
            # Load model (cached per session)
            cache_key = f"_model_{model_key}"
            if not hasattr(self, cache_key):
                model = SentenceTransformer(model_name, device="cpu")
                setattr(self, cache_key, model)
            
            model = getattr(self, cache_key)
            
            # Encode concepts
            embeddings = model.encode([concept_a, concept_b])
            
            # Compute cosine similarity
            from sklearn.metrics.pairwise import cosine_similarity
            similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
            
            return float(max(0, min(1, similarity)))  # Clamp to 0-1
            
        except Exception as e:
            logger.warning(f"Failed to compute embedding similarity with {model_key}: {e}")
            return None
    
    def run_single_experiment(
        self,
        embedding_key: str,
        use_graph_features: bool = False,
        test_size: float = 0.25
    ) -> Dict[str, Any]:
        """
        Run experiment with a specific embedding model.
        
        Args:
            embedding_key: Key for embedding model
            use_graph_features: Whether to use graph features
            test_size: Test set fraction
        
        Returns:
            Experiment results
        """
        records = self.load_labeled_data()
        
        if not records:
            return {"error": "No labeled data available"}
        
        # Compute embedding similarities
        embedding_sims = []
        for record in records:
            sim = self.compute_embedding_similarity(
                record["concept_a"],
                record["concept_b"],
                embedding_key
            )
            embedding_sims.append(sim)
            record["computed_similarity"] = sim
        
        # Extract features
        X, y = self._extract_features(records)
        
        if len(np.unique(y)) < 2:
            return {"error": "Need at least 2 classes for training"}
        
        # Split
        use_stratify = len(records) >= 6
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42,
            stratify=y if use_stratify else None
        )
        
        # Scale
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train
        model = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000, random_state=42)
        model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test_scaled)
        
        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "f1": float(f1_score(y_test, y_pred, zero_division=0))
        }
        
        # Compute additional metrics
        additional_metrics = self._compute_additional_metrics(
            records, y_test, y_pred, embedding_sims
        )
        
        return {
            "embedding_model": embedding_key,
            "embedding_name": EMBEDDING_MODELS[embedding_key]["display"],
            "train_size": len(X_train),
            "test_size": len(X_test),
            "num_features": X.shape[1],
            "metrics": metrics,
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
            "additional_metrics": additional_metrics
        }
    
    def _extract_features(self, records: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
        """Extract features from records."""
        features = []
        labels = []
        
        for record in records:
            if record.get("is_valid") is None:
                continue
            
            feat = []
            
            # LLM confidence
            feat.append(record.get("llm_confidence", 0.5))
            
            # Co-occurrence score
            cooc = record.get("cooccurrence_score", 0.0)
            feat.append(cooc if cooc else 0.0)
            
            # Semantic similarity (use computed if available, else stored)
            sem_sim = record.get("computed_similarity") or record.get("semantic_similarity")
            feat.append(sem_sim if sem_sim else 0.5)
            
            # Relation type encoding (simplified)
            rel_type = record.get("relation_type", "unknown")
            type_map = {"prerequisite": 0, "part-of": 1, "similar-to": 2, "cause-effect": 3,
                       "example-of": 4, "definition": 5, "derives-from": 6, "explanation": 7}
            feat.append(type_map.get(rel_type, 8) / 8.0)
            
            # Context length
            context = record.get("chunk_context", "")
            feat.append(min(len(context), 1000) / 1000.0)
            
            # Interaction features
            llm_conf = record.get("llm_confidence", 0.5)
            feat.append(llm_conf * (cooc if cooc else 0.0))
            feat.append(llm_conf * (sem_sim if sem_sim else 0.5))
            
            features.append(feat)
            labels.append(1 if record["is_valid"] else 0)
        
        return np.array(features), np.array(labels)
    
    def _compute_additional_metrics(
        self,
        records: List[Dict],
        y_test: np.ndarray,
        y_pred: np.ndarray,
        embedding_sims: List[Optional[float]]
    ) -> Dict[str, Any]:
        """Compute additional experiment metrics."""
        metrics = {}
        
        # Hallucination rate (predicted invalid when actually valid)
        hallucination_count = 0
        valid_count = 0
        for i, record in enumerate(records):
            if record.get("is_valid"):
                valid_count += 1
                if i < len(y_pred) and y_pred[i] == 0:  # Predicted invalid
                    hallucination_count += 1
        
        metrics["hallucination_rate"] = (
            hallucination_count / valid_count if valid_count > 0 else 0.0
        )
        
        # Average embedding similarity
        valid_sims = [s for s in embedding_sims if s is not None]
        metrics["avg_embedding_similarity"] = (
            np.mean(valid_sims) if valid_sims else 0.0
        )
        
        # Confidence calibration
        llm_confs = [r.get("llm_confidence", 0) for r in records]
        metrics["avg_llm_confidence"] = np.mean(llm_confs) if llm_confs else 0.0
        
        # Co-occurrence statistics
        coocs = [r.get("cooccurrence_score", 0) or 0 for r in records]
        metrics["avg_cooccurrence"] = np.mean(coocs) if coocs else 0.0
        
        return metrics
    
    def run_comparison(
        self,
        embedding_keys: List[str] = None,
        output_path: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        Run comparison across multiple embedding models.
        
        Args:
            embedding_keys: List of embedding keys to compare
            output_path: Path to save results
        
        Returns:
            Comparison results
        """
        if embedding_keys is None:
            embedding_keys = list(EMBEDDING_MODELS.keys())
        
        results = {
            "experiment_name": "Embedding Comparison for Relation Validation",
            "timestamp": datetime.now().isoformat(),
            "embeddings_compared": embedding_keys,
            "embedding_info": {
                k: EMBEDDING_MODELS[k] for k in embedding_keys
            },
            "results": {}
        }
        
        for key in embedding_keys:
            if key not in EMBEDDING_MODELS:
                logger.warning(f"Unknown embedding key: {key}")
                continue
            
            logger.info(f"Running experiment with {key}...")
            
            exp_result = self.run_single_experiment(key)
            results["results"][key] = exp_result
            
            if "error" not in exp_result:
                logger.info(f"  Accuracy: {exp_result['metrics']['accuracy']:.3f}")
        
        # Find best model
        best_key = None
        best_f1 = -1
        for key, res in results["results"].items():
            if "metrics" in res and res["metrics"]["f1"] > best_f1:
                best_f1 = res["metrics"]["f1"]
                best_key = key
        
        results["best_model"] = best_key
        results["best_f1"] = best_f1
        
        # Summary table
        results["summary"] = []
        for key, res in results["results"].items():
            if "metrics" in res:
                m = res["metrics"]
                add = res.get("additional_metrics", {})
                results["summary"].append({
                    "embedding": EMBEDDING_MODELS[key]["display"],
                    "accuracy": m["accuracy"],
                    "precision": m["precision"],
                    "recall": m["recall"],
                    "f1": m["f1"],
                    "hallucination_rate": add.get("hallucination_rate", 0),
                    "avg_similarity": add.get("avg_embedding_similarity", 0)
                })
        
        # Save to file
        if output_path is None:
            output_path = settings.data_dir / "embedding_comparison_results.json"
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        
        results["output_file"] = str(output_path)
        
        self.results = results
        return results


def run_embedding_comparison(
    embedding_keys: List[str] = None,
    output_path: str = None
) -> Dict[str, Any]:
    """
    Convenience function to run embedding comparison.
    
    Args:
        embedding_keys: Embedding models to compare
        output_path: Output file path
    
    Returns:
        Comparison results
    """
    comparator = EmbeddingComparator()
    return comparator.run_comparison(
        embedding_keys=embedding_keys,
        output_path=Path(output_path) if output_path else None
    )


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("Embedding Comparison Experiment")
    print("=" * 60)
    
    # Run comparison
    results = run_embedding_comparison(
        embedding_keys=["minilm", "bge-base"],
        output_path="data/embedding_comparison.json"
    )
    
    print("\nResults Summary:")
    print("-" * 60)
    
    if "summary" in results:
        print(f"{'Embedding':<20} {'Acc':>8} {'Prec':>8} {'Rec':>8} {'F1':>8}")
        print("-" * 60)
        for row in results["summary"]:
            print(f"{row['embedding']:<20} {row['accuracy']:>8.3f} {row['precision']:>8.3f} {row['recall']:>8.3f} {row['f1']:>8.3f}")
    
    print("-" * 60)
    print(f"Best Model: {results.get('best_model')} (F1: {results.get('best_f1', 0):.3f})")
    print(f"Results saved to: {results.get('output_file')}")
