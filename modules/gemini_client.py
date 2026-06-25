import google.genai as genai
import google.genai.errors as gerrors
import time
import os
from dotenv import load_dotenv

load_dotenv()

class RateLimiter:
    def __init__(self, calls_per_minute: int = 15):
        # 60s / calls_per_minute + 0.5s safety buffer
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


class MockGeminiModel:
    """Mock model used when cfg.llm.mock = true."""
    def generate_content(self, prompt, **kwargs):
        class R:
            text = MOCK_RESPONSES.get('generate', 'mock response')
        return R()


class _RealGeminiModel:
    """
    Thin wrapper around google.genai Client that exposes the same
    generate_content(prompt, temperature, ...) interface used by
    safe_gemini_call and the rest of the codebase.
    """
    def __init__(self, client: genai.Client, model_name: str):
        self._client = client
        self._model_name = model_name

    def generate_content(self, prompt: str, temperature: float = 0.0,
                         max_output_tokens: int = 512):
        config = genai.types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )
        return self._client.models.generate_content(
            model=self._model_name,
            contents=prompt,
            config=config,
        )


def safe_gemini_call(model, prompt: str, temp: float, retries: int = 3) -> str:
    """
    Calls model.generate_content with retry logic.
    Works for both MockGeminiModel and _RealGeminiModel.
    """
    if isinstance(model, MockGeminiModel):
        return model.generate_content(prompt).text

    for attempt in range(retries):
        try:
            response = model.generate_content(prompt, temperature=temp)
            return response.text.strip()
        except gerrors.ClientError as e:
            # 429 Resource Exhausted
            if '429' in str(e) or 'quota' in str(e).lower() or 'exhausted' in str(e).lower():
                wait = 60 * (attempt + 1)
                print(f'Rate limit hit, sleeping {wait}s (attempt {attempt+1}/{retries})...')
                time.sleep(wait)
            else:
                print(f'Gemini ClientError attempt {attempt+1}: {e}')
                time.sleep(5)
        except Exception as e:
            print(f'Gemini error attempt {attempt+1}: {e}')
            time.sleep(5)
    return ''


def get_gemini_model(model_name: str = 'models/gemini-3-flash-preview', mock: bool = False):
    """
    Returns a model object compatible with safe_gemini_call().
    Uses MockGeminiModel if mock=True, otherwise _RealGeminiModel.
    """
    if mock:
        return MockGeminiModel()

    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in environment. Check your .env file.")

    # Ensure model name has the required 'models/' prefix
    if not model_name.startswith('models/'):
        model_name = f'models/{model_name}'

    client = genai.Client(api_key=api_key)
    return _RealGeminiModel(client, model_name)
