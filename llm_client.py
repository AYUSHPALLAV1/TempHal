import google.generativeai as genai
import os
import time
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
            print(f"Rate limit hit. Waiting for {sleep_time:.2f} seconds...")
            time.sleep(sleep_time)
        self.last_call_time = time.time()

class GeminiClient:
    def __init__(self, api_key: str = None, model_name: str = 'gemini-1.5-flash', min_gap: float = 4.0):
        if not api_key:
            api_key = os.getenv('GEMINI_API_KEY')
        
        if not api_key:
            raise ValueError("Gemini API key not found. Set GEMINI_API_KEY in .env file.")

        genai.configure(api_key=api_key)
        
        # Try to use the requested model, but fall back if it fails
        try:
            self.model = genai.GenerativeModel(model_name)
            # Test call to verify model
            self.model.generate_content("test")
        except Exception as e:
            print(f"Warning: Model {model_name} not available. Falling back to gemini-3-flash-preview.")
            self.model = genai.GenerativeModel('gemini-3-flash-preview')
        
        self.rate_limiter = RateLimiter(min_gap)

    def generate_content(self, prompt: str, temperature: float = 0.0, max_tokens: int = 512) -> str:
        self.rate_limiter.wait_if_needed()
        
        try:
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                )
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
    # Test with two calls to verify rate limiting
    print("First call...")
    print(gemini_chat("Hello!"))
    print("\nSecond call (should wait)...")
    print(gemini_chat("How are you?"))
