from google import genai
import os

class GeminiEmbedding:
    def __init__(self):
        self.client = genai.Client()

    def embed(self, text : list[str] | str):
        result = self.client.models.embed_content(
            model="gemini-embedding-001",
            contents = text
        )

        return result.embeddings

    def batch_embed(self, text : list[str] | str):
        pass
