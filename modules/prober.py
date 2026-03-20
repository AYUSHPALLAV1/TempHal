from bert_score import score as bscore
from modules.paraphraser import paraphrase
import numpy as np

def probe_uncertainty(query: str, gemini_fn, rate_limiter) -> dict:
    """
    Computes uncertainty for a claim by paraphrasing the query 7 times 
    and measuring semantic consistency across Gemini responses.
    """
    paraphrases = paraphrase(query, n=7)
    responses = []
    
    for p in paraphrases:
        rate_limiter.wait()
        responses.append(gemini_fn(p, temp=0.7))
        
    if len(responses) < 2:
        return {
            'uncertainty_score': 0.5,
            'responses': responses,
            'paraphrases': paraphrases,
        }
        
    # Semantic consistency via BERTScore
    # Reference is the first response, hypotheses are the rest
    ref = [responses[0]] * (len(responses) - 1)
    hyp = responses[1:]
    
    try:
        # Explicitly use bert-base-uncased as specified in project details
        # to avoid RobertaTokenizer compatibility issues
        _, _, F = bscore(hyp, ref, model_type='bert-base-uncased', lang='en', verbose=False)
        # Uncertainty = 1 - mean(F1)
        uncertainty = round(1.0 - F.mean().item(), 4)
    except Exception as e:
        print(f"BERTScore failed: {e}")
        uncertainty = 0.5
    
    return {
        'uncertainty_score': uncertainty,
        'responses': responses,
        'paraphrases': paraphrases,
    }
