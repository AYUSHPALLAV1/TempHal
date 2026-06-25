import gradio as gr
import yaml
import os
import json
import traceback
from dotenv import load_dotenv
from modules.db import DB
from nlp_utils import extract_claims # Reusing nlp_utils for claim extraction
from classifier_inference import classify # Reusing classifier_inference for classification
from modules.prober import probe_uncertainty
from modules.retriever import retrieve, ground_claim
from modules.gemini_client import (
    safe_gemini_call, RateLimiter, get_gemini_model)

# 1. Setup Configuration
load_dotenv()
if not os.path.exists('config.yaml'):
    raise FileNotFoundError("config.yaml not found.")
cfg = yaml.safe_load(open('config.yaml'))

# 2. Initialize Clients
gemini = get_gemini_model(cfg['llm']['model'], mock=cfg['llm'].get('mock', False))
limiter = RateLimiter(cfg['llm']['calls_per_minute'])
db = DB(cfg['db']['path'])

def call(prompt, temp=0.0):
    """Wrapper for safe_gemini_call."""
    return safe_gemini_call(gemini, prompt, temp)


def run_pipeline(query: str):
    """
    Main entry point for the TempHal pipeline.
    """
    try:
        # 3.1 Cache Check
        cached = db.cache_get(query)
        if cached:
            print("Returning cached result.")
            return (cached['raw'], cached['highlights'], 
                    cached['scores_df'], 
                    cached['grounded'])

        # 3.2 Raw Generation
        print(f"Generating raw response for query: {query}")
        limiter.wait()
        raw = call(query, temp=cfg['llm']['temperature_generation'])
        if not raw:
            return "Error: No response from Gemini.", [], [], "Please check your API key and connection."

        # 3.3 Claim Extraction & Classification
        print("Extracting and classifying claims...")
        claims = extract_claims(raw)
        if not claims:
            return raw, [(raw, "stable")], [], raw

        texts = [c['text'] for c in claims]
        typed_results = classify(texts)

        # 3.4 Uncertainty Probing & Retrieval
        scores = []
        grounded_parts = []
        threshold = cfg['probing']['risk_threshold'] # Define threshold outside the loop
        
        for claim, t in zip(claims, typed_results):
            print(f"Processing claim: {claim['text'][:50]}...")
            
            # Only probe volatile/uncertain claims
            if t['temporal_type'] in ('volatile', 'uncertain'):
                print(f"Probing uncertainty for {t['temporal_type']} claim...")
                probe = probe_uncertainty(query, call, limiter)
                
                # Combine scores: volatility (0.4) + uncertainty (0.6)
                risk_score = round(
                    cfg['probing']['volatility_weight'] * t['volatility_score'] +
                    cfg['probing']['uncertainty_weight'] * probe['uncertainty_score'], 
                    3
                )
            else:
                probe = {'uncertainty_score': 0.0}
                risk_score = 0.0

            # 3.5 Retrieval Path for High Risk
            if risk_score >= threshold:
                print(f"High risk ({risk_score} >= {threshold}), triggering retrieval...")
                entities = claim.get('entities', [])
                evidence = retrieve(claim['text'], entities)
                
                print("Grounding claim with retrieved evidence...")
                grounded_res = ground_claim(claim['text'], evidence, call, limiter)
                label = 'high_risk'
            else:
                grounded_res = {
                    'grounded_claim': claim['text'],
                    'hallucination_type': 'none',
                    'confidence': 'n/a'
                }
                label = t['temporal_type']

            # Collect scores for display
            scores.append({
                'claim': claim['text'], # Use full text for highlighting
                'type': t['temporal_type'],
                'vol': t['volatility_score'],
                'unc': round(probe['uncertainty_score'], 3),
                'risk': risk_score,
                'hal_type': grounded_res['hallucination_type'],
                'final_label': label # Store label for highlights
            })
            grounded_parts.append(grounded_res['grounded_claim'])

        # 3.6 Assembly and Cache
        highlights = [(s['claim'], s['final_label']) for s in scores]
        grounded_text = ' '.join(grounded_parts)
        
        output = {
            'raw': raw,
            'claims': claims,
            'highlights': highlights,
            'scores_df': scores,
            'grounded': grounded_text
        }
        
        db.cache_set(query, output)
        return raw, highlights, scores, grounded_text
        
    except Exception as e:
        error_msg = f"Error in pipeline: {str(e)}"
        print(error_msg)
        traceback.print_exc()
        return error_msg, [], [], "Traceback printed to console."

# 4. Gradio Interface
EXAMPLES = [
    ['Who is the CEO of OpenAI and when was it founded?'],
    ['What is the largest language model available today?'],
    ['When did World War II end and what year was the United Nations founded?'],
    ['Who leads the European Central Bank and what is the current Eurozone inflation rate?'],
]

with gr.Blocks(title='TempHal — Temporal Hallucination Attribution') as demo:
    gr.Markdown('## TempHal — Temporal Hallucination Attribution')
    gr.Markdown('Enter any factual question. The system identifies which claims are time-sensitive, '
                'probes the model uncertainty, and grounds only the high-risk claims via free Wikipedia retrieval.')
    
    with gr.Row():
        query_box = gr.Textbox(label='Your query', lines=2)
        btn = gr.Button('Analyze', variant='primary')
    
    raw_out = gr.Textbox(label='Raw Gemini response')
    hl_out = gr.HighlightedText(
        label='Claims by risk level',
        color_map={'stable': 'green', 'volatile': 'orange', 'high_risk': 'red'}
    )
    df_out = gr.JSON(label='Claim scores')
    gr_out = gr.Textbox(label='Grounded response')
    
    gr.Examples(EXAMPLES, query_box)
    
    btn.click(run_pipeline, query_box, [raw_out, hl_out, df_out, gr_out])

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port, share=False)

