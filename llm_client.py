"""
llm_client.py — Legacy Gemini client (kept for compatibility).
The main app uses modules/gemini_client.py — this file is retained
but updated to use the current google.genai SDK.
"""
import google.genai as genai
import time
import os
from dotenv import load_dotenv

load_dotenv()


class RateLimiter:
    def __init__(self, min_gap_seconds: float):
        self.min_gap_seconds = min_gap_seconds
        self.last_call_time = 0

    def wait_if_needed(self):
        current_time = time.time()
        elapsed_time = current_time - self.last_call_time
        if elapsed_time < self.min_gap_seconds:
            sleep_time = self.min_gap_seconds - elapsed_time
            print(f"Rate limit: waiting {sleep_time:.2f}s...")
            time.sleep(sleep_time)
        self.last_call_time = time.time()


class GeminiClient:
    def __init__(self, api_key: str = None, model_name: str = 'models/gemini-3-flash-preview',
                 min_gap: float = 4.0):
        if not api_key:
            api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("Gemini API key not found. Set GEMINI_API_KEY in .env file.")

        # Ensure model name has the required 'models/' prefix
        if not model_name.startswith('models/'):
            model_name = f'models/{model_name}'

        self._client = genai.Client(api_key=api_key)
        self._model_name = model_name
        self.rate_limiter = RateLimiter(min_gap)

    def generate_content(self, prompt: str, temperature: float = 0.0,
                         max_tokens: int = 512) -> str:
        self.rate_limiter.wait_if_needed()
        try:
            config = genai.types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
            )
            response = self._client.models.generate_content(
                model=self._model_name,
                contents=prompt,
                config=config,
            )
            return response.text.strip()
        except Exception as e:
            print(f"Error calling Gemini API: {e}")
            return f"Error: {e}"


# Global instance for easy use
_client = None


def get_gemini_client():
    global _client
    if _client is None:
        _client = GeminiClient()
    return _client


def gemini_chat(prompt: str, temperature: float = 0.0) -> str:
    client = get_gemini_client()
    return client.generate_content(prompt, temperature)


if __name__ == "__main__":
    print("First call...")
    print(gemini_chat("Hello!"))
    print("\nSecond call (should wait)...")
    print(gemini_chat("How are you?"))
