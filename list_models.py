"""Lists available Gemini models that support content generation."""
import google.genai as genai
import os
from dotenv import load_dotenv

load_dotenv()
client = genai.Client(api_key=os.getenv('GEMINI_API_KEY'))

for m in client.models.list():
    if hasattr(m, 'supported_actions') and 'generateContent' in (m.supported_actions or []):
        print(m.name)
    elif not hasattr(m, 'supported_actions'):
        # Fallback: print all models
        print(m.name)
