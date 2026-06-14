import json
import random
import sys
from pathlib import Path

# Insert project root into sys.path to allow importing from rq1_experiment
sys.path.insert(0, str(Path(__file__).parent))

from rq1_experiment.rag_system.embedder import DocumentEmbedder
import numpy as np

def cosine_similarity(vec1, vec2):
    return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))

def verify_outputs():
    print("=== Automated Results Verification ===")
    
    raw_outputs_dir = Path("rq1_experiment/results/raw_outputs")
    queries_dir = Path("rq1_experiment/query_bank/queries")
    
    if not raw_outputs_dir.exists() or not queries_dir.exists():
        print("Could not find raw_outputs or query_bank directories. Has the experiment run?")
        return

    # Load all queries into a dictionary
    queries_db = {}
    print("Loading Query Banks...")
    for q_file in queries_dir.glob("*.json"):
        with open(q_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for q in data:
                queries_db[q["query_id"]] = q
    print(f"Loaded {len(queries_db)} total queries from the query bank.")

    # Load all raw outputs
    outputs = []
    print("Loading Generated Outputs...")
    for out_file in raw_outputs_dir.glob("*.json"):
        with open(out_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            outputs.extend(data)
    print(f"Loaded {len(outputs)} total responses.")

    if not outputs:
        print("No outputs found to verify.")
        return

    # Sample 10 random outputs to verify
    sample_size = min(10, len(outputs))
    sampled_outputs = random.sample(outputs, sample_size)

    # Initialize Embedder (will use DO API if local fails)
    embedder = DocumentEmbedder()
    
    print("\n" + "="*80)
    print("VERIFICATION RESULTS (Sample size: 10)")
    print("="*80)

    score_sum = 0.0
    valid_count = 0

    for idx, out in enumerate(sampled_outputs, 1):
        q_id = out["query_id"]
        response = out.get("response", "").strip()
        
        q_data = queries_db.get(q_id)
        if not q_data:
            print(f"[{idx}] Error: Query {q_id} not found in query banks!")
            continue
            
        question = q_data.get("question", "")
        ref_answer = q_data.get("reference_answer", "")
        
        print(f"\n--- Output {idx} [{q_id}] ---")
        print(f"Q: {question}")
        print(f"Reference: {ref_answer[:150]}...")
        print(f"Generated: {response[:150]}...")
        
        if not response or response.startswith("I cannot determine"):
            print("Status: Model indicated missing context (Expected behavior for some conditions).")
            continue
            
        # Check alignment via embeddings
        try:
            vec_ref = embedder.embed_query(ref_answer, normalize=True)[0]
            vec_gen = embedder.embed_query(response, normalize=True)[0]
            sim = cosine_similarity(vec_ref, vec_gen)
            
            alignment = "HIGH" if sim >= 0.7 else "MODERATE" if sim >= 0.5 else "LOW (Possible Hallucination)"
            print(f"Alignment Score: {sim:.3f} -> {alignment}")
            
            score_sum += sim
            valid_count += 1
        except Exception as e:
            print(f"Error calculating alignment: {e}")
            
    print("\n" + "="*80)
    if valid_count > 0:
        avg_score = score_sum / valid_count
        print(f"Average Semantic Alignment (Response vs Reference): {avg_score:.3f}")
        if avg_score > 0.65:
            print(f"VERDICT: PASS. Responses are highly aligned with the dataset and are NOT random generation. Score: {avg_score:.3f}")
        else:
            print(f"VERDICT: WARNING. Responses show low alignment with dataset. Manual review recommended. Score: {avg_score:.3f}")
    else:
        print("No valid responses to score (either none generated, or all were 'Cannot determine').")

if __name__ == "__main__":
    verify_outputs()
