import logging
import os
from abc import ABC, abstractmethod

from tenacity import retry, stop_after_attempt, wait_random_exponential
from ..providers import ProviderClient

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)


class BaseSummarizationModel(ABC):
    @abstractmethod
    def summarize(self, context, max_tokens=150):
        pass


class CommSummarizationModel(BaseSummarizationModel):
    def __init__(self, model=None, llm_base_url="https://api.example.com/v1",llm_api_key="your-api-key-here"):
        """
        Initialize class with support for custom model and API base URL.
        
        :param model: Model name
        :param llm_base_url: Base URL for LLM API
        """
        self.model = model
        self.llm_base_url = llm_base_url
        self.llm_api_key = llm_api_key
        self.provider_client = ProviderClient(
            provider=None,
            api_key=self.llm_api_key,
            base_url=self.llm_base_url,
            model_name=self.model,
        )

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def summarize(self, context, max_completion_tokens=5000, stop_sequence=None):
        """
        Generate text from a pre-built summarization prompt.

        Note: call sites are responsible for assembling task-specific prompt content.
        This avoids double-wrapping instructions when upstream already provides
        structured summarization guidance.

        :param context: Text that needs to be summarized
        :param max_tokens: Maximum number of summary tokens
        :param stop_sequence: Optional stop sequence
        :return: Generated summary
        """
        try:
            result = self.provider_client.chat_completion(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是一位阅读专家，能快速准确的理解书籍内容."},
                    {
                        "role": "user",
                        "content": context,
                    },
                ],
                max_tokens=max_completion_tokens,
                temperature=0,
                top_p=1,
                stop=stop_sequence,
            )
            return result.text

        except Exception as e:
            print(f"An error occurred: {e}")
            return str(e)
