#!/usr/bin/env python
"""
Relation Validation Experiment Script

Usage:
    python experiments/train_relation_model.py --help
    python experiments/train_relation_model.py
    python experiments/train_relation_model.py --cross-validate
    python experiments/train_relation_model.py --calibrate
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.experiments.relation_validator import (
    ExperimentRunner,
    RelationClassifier,
    RelationFeatureExtractor,
    run_quick_experiment
)
from utils.config import settings

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Train and evaluate relation validity classifier"
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path to relation dataset SQLite file"
    )
    
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test set fraction (default: 0.2)"
    )
    
    parser.add_argument(
        "--cross-validate",
        action="store_true",
        help="Run cross-validation"
    )
    
    parser.add_argument(
        "--model-output",
        type=str,
        default=None,
        help="Path to save trained model"
    )
    
    parser.add_argument(
        "--model-input",
        type=str,
        default=None,
        help="Path to load pretrained model"
    )
    
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="Calibrate confidence scores using trained model"
    )
    
    parser.add_argument(
        "--export-results",
        type=str,
        default=None,
        help="Export results to JSON file"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Relation Validation Experiment")
    print("=" * 60)
    
    # Load or train model
    if args.model_input:
        print(f"\nLoading model from: {args.model_input}")
        runner = ExperimentRunner(args.dataset)
        runner.load_model(Path(args.model_input))
        
        # Load labeled data for evaluation
        records = runner.load_labeled_data()
        if records:
            X, y = runner.classifier.feature_extractor.extract_features(records)
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            from sklearn.preprocessing import StandardScaler
            
            X_scaled = runner.classifier.feature_extractor.scaler.fit_transform(X)
            y_pred = runner.classifier.model.predict(X_scaled)
            
            print(f"\nLoaded model evaluation on {len(records)} labeled samples:")
            print(f"  Accuracy:  {accuracy_score(y, y_pred):.3f}")
            print(f"  Precision: {precision_score(y, y_pred, zero_division=0):.3f}")
            print(f"  Recall:    {recall_score(y, y_pred, zero_division=0):.3f}")
            print(f"  F1:        {f1_score(y, y_pred, zero_division=0):.3f}")
        
    else:
        print(f"\nRunning experiment...")
        print(f"  Test size: {args.test_size}")
        print(f"  Cross-validate: {args.cross_validate}")
        
        results = run_quick_experiment(
            dataset_path=Path(args.dataset) if args.dataset else None,
            test_size=args.test_size
        )
        
        if "error" in results:
            print(f"\nError: {results['error']}")
            print(f"Message: {results.get('message', '')}")
            return 1
        
        print(f"\nResults:")
        print(f"  Train size: {results.get('train_size')}")
        print(f"  Test size: {results.get('test_size')}")
        
        if "test_metrics" in results:
            metrics = results["test_metrics"]
            print(f"\nTest Metrics:")
            print(f"  Accuracy:  {metrics['accuracy']:.3f}")
            print(f"  Precision: {metrics['precision']:.3f}")
            print(f"  Recall:    {metrics['recall']:.3f}")
            print(f"  F1 Score:  {metrics['f1']:.3f}")
        
        if "cross_validation" in results:
            cv = results["cross_validation"]
            print(f"\nCross-Validation ({len(cv['scores'])} folds):")
            print(f"  Mean: {cv['mean']:.3f} (+/- {cv['std']:.3f})")
        
        if "feature_importance" in results:
            print(f"\nFeature Importance:")
            sorted_features = sorted(
                results["feature_importance"].items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )
            for name, coef in sorted_features:
                sign = "+" if coef > 0 else ""
                print(f"  {name}: {sign}{coef:.3f}")
        
        if "confusion_matrix" in results:
            cm = results["confusion_matrix"]
            print(f"\nConfusion Matrix:")
            print(f"  [[{cm[0][0]:3d}, {cm[0][1]:3d}],")
            print(f"   [{cm[1][0]:3d}, {cm[1][1]:3d}]]")
        
        if "model_saved" in results:
            print(f"\nModel saved: {results['model_saved']}")
        
        if args.export_results:
            output_path = Path(args.export_results)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(results, f, indent=2, default=str)
            print(f"\nResults exported to: {output_path}")
    
    # Calibration
    if args.calibrate:
        print("\n" + "=" * 60)
        print("Confidence Calibration")
        print("=" * 60)
        
        runner = ExperimentRunner(args.dataset)
        
        if args.model_input:
            runner.load_model(Path(args.model_input))
        else:
            model_path = args.model_output or str(settings.data_dir / "relation_model.pkl")
            if Path(model_path).exists():
                runner.load_model(Path(model_path))
            else:
                print("No trained model available. Run training first.")
                return 1
        
        # Get unlabeled records
        conn = runner.load_labeled_data()
        
        conn = None
        if args.dataset:
            db_path = Path(args.dataset)
        else:
            db_path = settings.data_dir / "relation_dataset.db"
        
        if db_path.exists():
            import sqlite3
            sqlite_conn = sqlite3.connect(str(db_path))
            sqlite_conn.row_factory = sqlite3.Row
            cursor = sqlite_conn.cursor()
            cursor.execute("SELECT * FROM relation_dataset LIMIT 10")
            rows = cursor.fetchall()
            sqlite_conn.close()
            
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
                    "is_valid": row["is_valid"]
                })
            
            calibrated = runner.calibrate_confidence(records)
            
            print(f"\nCalibrated {len(calibrated)} relations:")
            for rec in calibrated[:5]:
                print(f"  {rec['concept_a']} -> {rec['concept_b']}")
                print(f"    Original: {rec['original_confidence']:.2f} -> Calibrated: {rec['calibrated_confidence']:.2f}")
                print(f"    Predicted: {rec['predicted_valid']}, Above threshold: {rec['above_threshold']}")
    
    print("\n" + "=" * 60)
    print("Experiment Complete")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
