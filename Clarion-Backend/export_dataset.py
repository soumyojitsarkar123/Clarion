import sys
from pathlib import Path

# Add the backend root to the path so we can import modules
backend_dir = Path(__file__).parent.resolve()
sys.path.append(str(backend_dir))

from services.relation_dataset_service import RelationDatasetService

def export():
    print("Connecting to Relation Dataset...")
    service = RelationDatasetService()
    
    stats = service.get_dataset_stats()
    print(f"Found {stats['total_records']} total records in the dataset.")
    print("Exporting to JSON...")
    
    export_data = service.export_dataset(format="json")
    
    # Clean up the output for research (remove internal IDs and noisy metadata)
    clean_records = []
    for record in export_data.get("records", []):
        clean_record = {
            "concept_a": record.get("concept_a"),
            "concept_a_normalized": record.get("concept_a_normalized"),
            "relation_type": record.get("relation_type"),
            "concept_b": record.get("concept_b"),
            "concept_b_normalized": record.get("concept_b_normalized"),
            "llm_confidence": record.get("llm_confidence"),
            "extraction_confidence": record.get("extraction_confidence"),
            "confidence_source": record.get("confidence_source"),
            "cooccurrence_score": record.get("cooccurrence_score"),
            "semantic_similarity": record.get("semantic_similarity"),
            "chunk_context": record.get("chunk_context"),
            "source_chunk_ids": record.get("source_chunk_ids"),
            "quality_score": record.get("quality_score"),
            "quality_flags": record.get("quality_flags"),
            "relation_description": record.get("relation_description"),
            "created_at": record.get("created_at")
        }
        clean_records.append(clean_record)
        
    final_output = {
        "format": export_data.get("format"),
        "count": export_data.get("count"),
        "quality_summary": export_data.get("quality_summary", {}),
        "records": clean_records
    }
    
    output_file = backend_dir / "data" / "research_dataset_export.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        import json
        json.dump(final_output, f, indent=2, ensure_ascii=False)
        
    print(f"Done! Dataset exported successfully to: {output_file}")

if __name__ == "__main__":
    export()
