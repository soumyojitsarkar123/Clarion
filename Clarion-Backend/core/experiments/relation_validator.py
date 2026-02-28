"""
Relation Validation Experiment Module

A research module for training and evaluating relation validity classifiers
using the collected dataset.
"""

import json
import logging
import pickle
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
from sklearn.preprocessing import LabelEncoder, StandardScaler

from utils.config import settings
from core.experiments.graph_features import GraphFeatureExtractor, GraphEnrichedFeatureExtractor

logger = logging.getLogger(__name__)


class RelationFeatureExtractor:
    """
    Extracts features from relation dataset records for training.
    """
    
    RELATION_TYPES = [
        "prerequisite", "definition", "explanation", 
        "cause-effect", "example-of", "similar-to", 
        "part-of", "derives-from"
    ]
    
    def __init__(self):
        self.relation_encoder = LabelEncoder()
        self.relation_encoder.fit(self.RELATION_TYPES)
        self.scaler = StandardScaler()
    
    def extract_features(self, records: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract features and labels from relation records.
        
        Args:
            records: List of relation dataset records
        
        Returns:
            Tuple of (features, labels)
        """
        features = []
        labels = []
        
        for record in records:
            if record.get("is_valid") is None:
                continue
            
            feature_vector = self._extract_single(record)
            features.append(feature_vector)
            labels.append(1 if record["is_valid"] else 0)
        
        return np.array(features), np.array(labels)
    
    def _extract_single(self, record: Dict) -> List[float]:
        """Extract feature vector for a single record."""
        features = []
        
        # LLM confidence (0-1)
        llm_conf = record.get("llm_confidence", 0.5)
        features.append(llm_conf)
        
        # Co-occurrence score (0-1)
        cooc = record.get("cooccurrence_score", 0.0)
        features.append(cooc if cooc is not None else 0.0)
        
        # Semantic similarity (0-1)
        sem_sim = record.get("semantic_similarity")
        features.append(sem_sim if sem_sim is not None else 0.5)
        
        # Relation type encoding
        rel_type = record.get("relation_type", "unknown")
        try:
            type_encoded = self.relation_encoder.transform([rel_type])[0]
        except ValueError:
            type_encoded = len(self.RELATION_TYPES)
        features.append(type_encoded / len(self.RELATION_TYPES))
        
        # Context length feature
        context = record.get("chunk_context", "")
        context_len = min(len(context), 1000) / 1000.0
        features.append(context_len)
        
        # Number of source chunks
        chunk_ids = record.get("source_chunk_ids", [])
        chunk_count = min(len(chunk_ids), 5) / 5.0
        features.append(chunk_count)
        
        # Confidence * Co-occurrence interaction
        features.append(llm_conf * (cooc if cooc else 0.0))
        
        # Confidence * Semantic similarity interaction
        features.append(llm_conf * (sem_sim if sem_sim else 0.5))
        
        return features
    
    def get_feature_names(self) -> List[str]:
        """Return feature names for interpretation."""
        return [
            "llm_confidence",
            "cooccurrence_score", 
            "semantic_similarity",
            "relation_type_encoded",
            "context_length",
            "chunk_count",
            "conf_x_cooc",
            "conf_x_sem_sim"
        ]


class RelationClassifier:
    """
    Lightweight relation validity classifier.
    
    Uses logistic regression for interpretability and speed.
    Supports optional graph-structural features.
    """
    
    def __init__(
        self, 
        model: Optional[LogisticRegression] = None,
        use_graph_features: bool = True
    ):
        self.model = model or LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=1000,
            random_state=42
        )
        self.feature_extractor = RelationFeatureExtractor()
        self.use_graph_features = use_graph_features
        self.graph_extractor: Optional[GraphEnrichedFeatureExtractor] = None
        self.is_trained = False
        
        if use_graph_features:
            self.graph_extractor = GraphEnrichedFeatureExtractor()
    
    def train(
        self,
        records: List[Dict],
        test_size: float = 0.2,
        cross_validate: bool = True,
        load_graphs: bool = True
    ) -> Dict[str, Any]:
        """
        Train the classifier on relation records.
        
        Args:
            records: Labeled relation records
            test_size: Fraction for test split
            cross_validate: Run cross-validation
            load_graphs: Whether to load graphs for features
        
        Returns:
            Training results and metrics
        """
        if not records:
            raise ValueError("No training records provided")
        
        # Extract features with optional graph features
        if self.use_graph_features and self.graph_extractor and load_graphs:
            X, y, feature_names = self._extract_with_graph_features(records)
        else:
            X, y = self.feature_extractor.extract_features(records)
            feature_names = self.feature_extractor.get_feature_names()
        
        if len(np.unique(y)) < 2:
            raise ValueError("Need at least 2 classes for training")
        
        logger.info(f"Training on {len(X)} samples, {X.shape[1]} features")
        
        # Split data (use stratification only if enough samples)
        use_stratify = len(records) >= 6
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, 
            stratify=y if use_stratify else None
        )
        
        # Scale features
        X_train_scaled = self.feature_extractor.scaler.fit_transform(X_train)
        X_test_scaled = self.feature_extractor.scaler.transform(X_test)
        
        # Train
        self.model.fit(X_train_scaled, y_train)
        self.is_trained = True
        
        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        
        results = {
            "train_size": len(X_train),
            "test_size": len(X_test),
            "features": feature_names,
            "num_features": len(feature_names),
            "graph_features_enabled": self.use_graph_features,
            "test_metrics": {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": float(precision_score(y_test, y_pred, zero_division=0)),
                "recall": float(recall_score(y_test, y_pred, zero_division=0)),
                "f1": float(f1_score(y_test, y_pred, zero_division=0))
            },
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
            "classification_report": str(classification_report(y_test, y_pred, zero_division=0))
        }
        
        # Cross-validation
        if cross_validate and len(X) >= 10:
            X_all_scaled = self.feature_extractor.scaler.fit_transform(X)
            cv_scores = cross_val_score(self.model, X_all_scaled, y, cv=min(5, len(X)//2))
            results["cross_validation"] = {
                "scores": cv_scores.tolist(),
                "mean": float(cv_scores.mean()),
                "std": float(cv_scores.std())
            }
        
        self._last_results = results
        return results
    
    def _extract_with_graph_features(
        self, 
        records: List[Dict]
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Extract features including graph features."""
        # Group records by document
        docs = {}
        for record in records:
            doc_id = record.get("document_id")
            if doc_id not in docs:
                docs[doc_id] = []
            docs[doc_id].append(record)
        
        # Process each document's graph
        all_features = []
        labels = []
        
        for doc_id, doc_records in docs.items():
            # Try to load graph for this document
            graph_loaded = self.graph_extractor.load_graph_for_document(doc_id)
            
            for record in doc_records:
                if record.get("is_valid") is None:
                    continue
                
                # Get basic features
                basic_features = self.feature_extractor._extract_single(record)
                
                # Add graph features if available
                if graph_loaded:
                    graph_features = self.graph_extractor.extract_all_features(
                        {},
                        record.get("concept_a", ""),
                        record.get("concept_b", "")
                    )
                    # Convert graph features to list in correct order
                    feature_dict = {}
                    for name in self.graph_extractor.get_feature_names():
                        feature_dict[name] = graph_features.get(name, 0.0)
                    graph_feature_list = [feature_dict.get(n, 0.0) for n in self.graph_extractor.get_feature_names()]
                    
                    # Combine
                    all_features.append(basic_features + graph_feature_list)
                else:
                    # Add zeros for graph features
                    graph_feature_count = len(self.graph_extractor.get_feature_names())
                    all_features.append(basic_features + [0.0] * graph_feature_count)
                
                labels.append(1 if record["is_valid"] else 0)
        
        X = np.array(all_features)
        y = np.array(labels)
        feature_names = self.feature_extractor.get_feature_names() + self.graph_extractor.get_feature_names()
        
        return X, y, feature_names
    
    def predict(self, record: Dict) -> Tuple[int, float]:
        """
        Predict validity for a single record.
        
        Args:
            record: Relation record
        
        Returns:
            Tuple of (prediction, probability)
        """
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        features = self.feature_extractor._extract_single(record)
        features_scaled = self.feature_extractor.scaler.transform([features])
        
        prediction = self.model.predict(features_scaled)[0]
        probability = self.model.predict_proba(features_scaled)[0]
        
        return int(prediction), float(probability[1])
    
    def predict_batch(self, records: List[Dict]) -> List[Dict[str, Any]]:
        """Predict validity for multiple records."""
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        results = []
        for record in records:
            pred, prob = self.predict(record)
            results.append({
                "record_id": record.get("record_id"),
                "predicted_valid": bool(pred),
                "confidence": prob
            })
        return results
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance coefficients."""
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        feature_names = self.feature_extractor.get_feature_names()
        coefficients = self.model.coef_[0]
        
        return dict(zip(feature_names, coefficients.tolist()))


class ExperimentRunner:
    """
    Runs relation validation experiments using the collected dataset.
    """
    
    def __init__(self, dataset_path: Optional[Path] = None):
        self.dataset_path = dataset_path or (settings.data_dir / "relation_dataset.db")
        self.classifier = RelationClassifier()
        self.results = {}
    
    def load_labeled_data(self, min_samples: int = 10) -> List[Dict]:
        """
        Load labeled relation records from dataset.
        
        Args:
            min_samples: Minimum number of samples required
        
        Returns:
            List of labeled records
        """
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
                "source_chunk_ids": json.loads(row["source_chunk_ids"]),
                "is_valid": bool(row["is_valid"])
            })
        
        logger.info(f"Loaded {len(records)} labeled records")
        
        if len(records) < min_samples:
            logger.warning(f"Only {len(records)} labeled samples (need {min_samples})")
        
        return records
    
    def run_experiment(
        self,
        test_size: float = 0.2,
        save_model: bool = True,
        model_path: Optional[Path] = None,
        use_graph_features: bool = True,
        load_graphs: bool = True
    ) -> Dict[str, Any]:
        """
        Run complete experiment: load data, train, evaluate.
        
        Args:
            test_size: Test set fraction
            save_model: Whether to save trained model
            model_path: Path to save model
            use_graph_features: Whether to use graph features
            load_graphs: Whether to load graphs for feature extraction
        
        Returns:
            Complete experiment results
        """
        # Create classifier with graph features option
        self.classifier = RelationClassifier(use_graph_features=use_graph_features)
        
        records = self.load_labeled_data()
        
        if not records:
            return {
                "error": "No labeled data available",
                "message": "Please label some relations using the API first"
            }
        
        results = self.classifier.train(
            records, 
            test_size=test_size,
            load_graphs=load_graphs
        )
        
        if save_model:
            model_path = model_path or (settings.data_dir / "relation_model.pkl")
            self.save_model(model_path)
            if results:
                results["model_saved"] = str(model_path)
        
        if results:
            feature_importance = self.classifier.get_feature_importance()
            results["feature_importance"] = feature_importance
        
        self.results = results
        return results
    
    def save_model(self, path: Path) -> None:
        """Save trained model to disk."""
        model_data = {
            "model": self.classifier.model,
            "feature_extractor": self.classifier.feature_extractor,
            "trained_at": datetime.now().isoformat(),
            "version": "1.0.0"
        }
        
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, "wb") as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: Path) -> None:
        """Load trained model from disk."""
        with open(path, "rb") as f:
            model_data = pickle.load(f)
        
        self.classifier.model = model_data["model"]
        self.classifier.feature_extractor = model_data["feature_extractor"]
        self.classifier.is_trained = True
        
        logger.info(f"Model loaded from {path}")
    
    def calibrate_confidence(
        self,
        records: List[Dict],
        threshold: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Use trained model to calibrate relation confidence scores.
        
        Args:
            records: Relation records
            threshold: Decision threshold
        
        Returns:
            Records with calibrated confidence
        """
        if not self.classifier.is_trained:
            raise ValueError("Model not trained")
        
        calibrated = []
        
        for record in records:
            pred, model_prob = self.classifier.predict(record)
            
            original_conf = record.get("llm_confidence", 0.5)
            cooc = record.get("cooccurrence_score", 0.5)
            
            calibrated_conf = (
                0.4 * original_conf + 
                0.3 * model_prob + 
                0.3 * (cooc if cooc else 0.5)
            )
            
            calibrated.append({
                **record,
                "original_confidence": original_conf,
                "model_probability": model_prob,
                "calibrated_confidence": round(calibrated_conf, 3),
                "predicted_valid": bool(pred),
                "above_threshold": calibrated_conf >= threshold
            })
        
        return calibrated


def run_quick_experiment(
    dataset_path: Optional[Path] = None,
    test_size: float = 0.2,
    use_graph_features: bool = True,
    load_graphs: bool = True
) -> Dict[str, Any]:
    """
    Convenience function to run a quick experiment.
    
    Args:
        dataset_path: Path to relation dataset
        test_size: Test set fraction
        use_graph_features: Whether to use graph features
        load_graphs: Whether to load graphs for feature extraction
    
    Returns:
        Experiment results
    """
    runner = ExperimentRunner(dataset_path)
    return runner.run_experiment(
        test_size=test_size,
        use_graph_features=use_graph_features,
        load_graphs=load_graphs
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("Relation Validation Experiment")
    print("=" * 60)
    
    results = run_quick_experiment()
    
    print("\nResults:")
    print(f"  Train size: {results.get('train_size')}")
    print(f"  Test size: {results.get('test_size')}")
    
    if "test_metrics" in results:
        metrics = results["test_metrics"]
        print(f"\nTest Metrics:")
        print(f"  Accuracy:  {metrics['accuracy']:.3f}")
        print(f"  Precision: {metrics['precision']:.3f}")
        print(f"  Recall:    {metrics['recall']:.3f}")
        print(f"  F1 Score:  {metrics['f1']:.3f}")
    
    if "feature_importance" in results:
        print(f"\nFeature Importance:")
        for name, coef in results["feature_importance"].items():
            print(f"  {name}: {coef:+.3f}")
    
    print(f"\nModel saved: {results.get('model_saved', 'Not saved')}")
