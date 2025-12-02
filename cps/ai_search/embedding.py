from google import genai
import os

GEMINI_API_KEY = os.environ['GEMINI_API_KEY']


class GeminiEmbedding:
    def __init__(self):
        self.client = genai.Client(api_key=GEMINI_API_KEY)

    def embed(self, text : list[str] | str):
        result = self.client.models.embed_content(
            model="gemini-embedding-001",
            contents = text
        )

        return result.embeddings