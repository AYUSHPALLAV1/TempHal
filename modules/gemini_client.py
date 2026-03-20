import google.generativeai as genai
import google.api_core.exceptions as gexc
import time
import os
from dotenv import load_dotenv

load_dotenv()

class RateLimiter:
    def __init__(self, calls_per_minute: int = 15):
        # 60s / 15 calls = 4s. Adding 0.5s safety buffer = 4.5s
        self.min_interval = 60.0 / calls_per_minute + 0.5
        self._last_call = 0.0

    def wait(self):
        elapsed = time.time() - self._last_call
        gap = self.min_interval - elapsed
        if gap > 0:
            time.sleep(gap)
        self._last_call = time.time()

MOCK_RESPONSES = { 
  'generate': 'Sam Altman is the CEO of OpenAI. OpenAI was founded in 2015.', 
  'ground':   'corrected_claim: Sam Altman is CEO\nhallucination_type: none\nconfidence: high', 
} 

class MockGeminiClient: 
  def generate_content(self, prompt, **kwargs): 
    class R: 
      text = MOCK_RESPONSES.get('generate','mock response') 
    return R()

def safe_gemini_call(model, prompt, temp, retries=3) -> str:
    # Handle Mock client
    if isinstance(model, MockGeminiClient):
        return model.generate_content(prompt).text

    for attempt in range(retries):
        try:
            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=temp,
                    max_output_tokens=512
                )
            )
            return response.text.strip()
        except gexc.ResourceExhausted:
            print(f'Rate limit hit, sleeping 60s (Attempt {attempt+1}/{retries})...')
            time.sleep(60)
        except Exception as e:
            print(f'Gemini error attempt {attempt+1}: {e}')
            time.sleep(5)
    return ''

def get_gemini_model(model_name='gemini-3-flash-preview', mock=False):
    if mock:
        return MockGeminiClient()
        
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in environment")
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(model_name)
