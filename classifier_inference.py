from transformers import pipeline
import os

# Model path
MODEL_PATH = 'outputs/classifier'

# Global pipeline instance
_clf_pipe = None

def get_classifier():
    global _clf_pipe
    if _clf_pipe is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Classifier model not found at {MODEL_PATH}. Run training first.")
        
        print(f"Loading Temporal Classifier from {MODEL_PATH}...")
        _clf_pipe = pipeline(
            'text-classification',
            model=MODEL_PATH,
            tokenizer=MODEL_PATH,
            top_k=None # Returns scores for all labels
        )
    return _clf_pipe

def classify(claims: list[str]) -> list[dict]:
    """
    Classifies a list of claims into stable, volatile, or uncertain.
    Returns a list of dictionaries with labels, probabilities, and volatility scores.
    """
    if not claims:
        return []
        
    clf = get_classifier()
    results = clf(claims, truncation=True, max_length=128)
    
    output = []
    for claim_text, scores in zip(claims, results):
        # Convert list of {label, score} to a map
        score_map = {s['label']: s['score'] for s in scores}
        
        # Get top label
        top_label = max(score_map, key=score_map.get)
        
        # Compute volatility score: volatile weight = 1.0, uncertain weight = 0.5
        volatility = (score_map.get('volatile', 0) + 
                     0.5 * score_map.get('uncertain', 0))
        
        output.append({
            'claim': claim_text,
            'temporal_type': top_label,
            'type_probs': score_map,
            'volatility_score': round(volatility, 4),
        })
        
    return output

if __name__ == "__main__":
    test_claims = [
        "Sundar Pichai is the CEO of Google",
        "The capital of France is Paris",
        "What is the current price of Bitcoin?"
    ]
    print("Running classification test...")
    results = classify(test_claims)
    for res in results:
        print(f"\nClaim: {res['claim']}")
        print(f"Type: {res['temporal_type']} (Volatility: {res['volatility_score']})")
        print(f"Probs: {res['type_probs']}")
