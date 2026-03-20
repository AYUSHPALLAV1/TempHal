import requests
import json
import time
from duckduckgo_search import DDGS

def _wiki_search(query: str) -> str:
    """Queries Wikipedia search for snippets."""
    try:
        r = requests.get(
            'https://en.wikipedia.org/w/api.php',
            params={
                'action': 'query',
                'list': 'search',
                'srsearch': query,
                'format': 'json',
                'srlimit': 3
            },
            timeout=8
        )
        r.raise_for_status()
        data = r.json()
        hits = data['query']['search']
        return ' '.join(h['snippet'] for h in hits)
    except Exception as e:
        print(f"Wikipedia search failed: {e}")
        return ""

def _ddg_search(query: str) -> str:
    """Fallback search using DuckDuckGo."""
    try:
        # DDG has soft rate limits, wait 2s
        time.sleep(2)
        results = DDGS().text(query, max_results=4)
        return ' '.join(r['body'] for r in results)
    except Exception as e:
        print(f"DuckDuckGo search failed: {e}")
        return ""

def retrieve(claim: str, entities: list) -> str:
    """
    Retrieves evidence for a claim, first from Wikipedia, 
    then fallback to DuckDuckGo if insufficient.
    """
    # Join entity text as the query
    query = ' '.join(e['text'] for e in entities) if entities else claim
    
    evidence = _wiki_search(query)
    
    # If less than 30 words, fallback to DDG
    if len(evidence.split()) < 30:
        print(f"Wikipedia insufficient ({len(evidence.split())} words), trying DuckDuckGo...")
        evidence = _ddg_search(query)
        
    return evidence

GROUND_PROMPT = '''
Claim: {claim}

Retrieved evidence:
{evidence}

Assess whether the claim is accurate based on the evidence. Respond in exactly this format:
corrected_claim: <the claim, corrected if needed>
hallucination_type: <temporal_staleness | parametric_absence | none>
confidence: <high | medium | low>
'''.strip()

def ground_claim(claim: str, evidence: str, gemini_fn, rate_limiter) -> dict:
    """
    Grounds a claim using Gemini and the retrieved evidence.
    Returns a structured dictionary.
    """
    rate_limiter.wait()
    resp = gemini_fn(
        GROUND_PROMPT.format(claim=claim, evidence=evidence),
        temp=0.0 # Deterministic
    )
    
    # Parse the response lines
    lines = {}
    for line in resp.splitlines():
        if ':' in line:
            k, v = line.split(':', 1)
            lines[k.strip().lower()] = v.strip()
            
    return {
        'grounded_claim': lines.get('corrected_claim', claim),
        'hallucination_type': lines.get('hallucination_type', 'none'),
        'confidence': lines.get('confidence', 'low'),
        'evidence_used': evidence,
    }
