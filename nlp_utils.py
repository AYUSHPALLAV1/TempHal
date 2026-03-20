import spacy
from sentence_transformers import SentenceTransformer
import numpy as np
import json

# Initialize models
_spacy_nlp = None
_sentence_model = None

ENTITY_TYPES = {'PERSON', 'ORG', 'GPE', 'LAW', 'PRODUCT', 'WORK_OF_ART'}

def get_spacy_nlp():
    global _spacy_nlp
    if _spacy_nlp is None:
        # Using en_core_web_sm as default for efficiency on CPU
        # User can switch to 'en_core_web_trf' if GPU is available
        model_name = "en_core_web_sm"
        try:
            _spacy_nlp = spacy.load(model_name)
        except OSError:
            print(f"Spacy model '{model_name}' not found. Downloading...")
            spacy.cli.download(model_name)
            _spacy_nlp = spacy.load(model_name)
    return _spacy_nlp

def get_sentence_model():
    global _sentence_model
    if _sentence_model is None:
        print("Loading Sentence Transformer model (all-MiniLM-L6-v2)...")
        _sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
    return _sentence_model

def extract_claims(text: str) -> list[dict]:
    """
    Advanced claim extraction using spaCy's NER and dependency parser.
    Targets subject-verb-object triples containing at least one named entity.
    """
    nlp = get_spacy_nlp()
    doc = nlp(text)
    claims = []
    
    for sent in doc.sents:
        # Check for named entities in the sentence
        ents = [e for e in sent.ents if e.label_ in ENTITY_TYPES]
        if not ents:
            continue
            
        # Find root verb
        root = [t for t in sent if t.dep_ == 'ROOT']
        if not root:
            continue
        root = root[0]
        
        # Ensure root is a verb or auxiliary
        if root.pos_ not in ('VERB', 'AUX'):
            continue
            
        # Skip interrogative sentences
        if sent.text.strip().endswith('?'):
            continue
            
        claims.append({
            'text': sent.text.strip(),
            'entities': [{
                'text': e.text,
                'label': e.label_,
                'start': e.start_char,
                'end': e.end_char
            } for e in ents],
        })
    return claims

def get_embedding(text: str):
    """Computes embedding for a given text."""
    model = get_sentence_model()
    return model.encode(text)

def cosine_similarity(v1, v2):
    """Computes cosine similarity between two vectors."""
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

if __name__ == "__main__":
    test_text = "Sundar Pichai is the CEO of Google. He joined the company in 2004. What is the weather like?"
    print("Extracting advanced claims...")
    extracted = extract_claims(test_text)
    for i, c in enumerate(extracted):
        print(f"Claim {i+1}: {c['text']}")
        print(f"  Entities: {[e['text'] for e in c['entities']]}")
