
import os
from dotenv import load_dotenv

from llm_client import (
    GeminiProvider,
    OllamaProvider,
    LLMClient
)

from usage_logger import UsageLogger

# Load environment variables from the project directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"), override=True)

# Connect to MongoDB
usage_logger = UsageLogger()

try:
    # Select provider
    provider_name = input(
        "Choose provider (gemini/ollama): "
    ).strip().lower()

    if provider_name == "gemini":

        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY not found in .env"
            )

        provider = GeminiProvider(api_key)

    elif provider_name == "ollama":

        provider = OllamaProvider("llama3.2")

    else:

        raise ValueError("Invalid provider selected")

    # Create common LLM client
    client = LLMClient(provider, usage_logger)

    # Get user prompt
    prompt = input("Enter your question: ")

    # Generate response
    answer = client.generate(prompt)

    print("\nLLM Response:")
    print(answer)

finally:
    # Close MongoDB connection
    usage_logger.close()
