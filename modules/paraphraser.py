from transformers import PegasusTokenizer, PegasusForConditionalGeneration
import torch

# Load models
PEGASUS_MODEL = 'tuner007/pegasus_paraphrase'
_ptok = None
_pmod = None

def get_paraphrase_model():
    global _ptok, _pmod
    if _ptok is None or _pmod is None:
        print(f"Loading local PEGASUS model: {PEGASUS_MODEL}...")
        _ptok = PegasusTokenizer.from_pretrained(PEGASUS_MODEL)
        _pmod = PegasusForConditionalGeneration.from_pretrained(PEGASUS_MODEL)
        _pmod.eval()
        
        # Move to GPU if available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _pmod.to(device)
    return _ptok, _pmod

def paraphrase(text: str, n: int = 7) -> list[str]:
    """
    Generates n paraphrases of a given text using a local PEGASUS model.
    """
    tok, mod = get_paraphrase_model()
    device = next(mod.parameters()).device
    
    batch = tok(
        [text], truncation=True, 
        padding='longest', max_length=128, 
        return_tensors='pt'
    ).to(device)
    
    with torch.no_grad():
        outputs = mod.generate(
            **batch,
            max_length=128,
            num_beams=n,
            num_return_sequences=n,
            temperature=1.5,
            do_sample=True,
            early_stopping=True
        )
    return [tok.decode(o, skip_special_tokens=True) for o in outputs]
