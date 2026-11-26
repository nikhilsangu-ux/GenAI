import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

import requests # type: ignore
from google import genai
from google.genai import types


# Store LLM response and token information
@dataclass
class LLMResult:
    text: str
    input_tokens: int
    output_tokens: int


# Common provider interface
class LLMProvider(ABC):

    @abstractmethod
    def generate(self, prompt):
        pass


# Gemini Provider
class GeminiProvider(LLMProvider):

    provider_name = "Gemini"

    def __init__(self, api_key, model="gemini-2.5-flash"):

        self.model = model

        self.client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(
                timeout=60000
            )
        )

    def generate(self, prompt):

        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt
        )

        usage = response.usage_metadata

        input_count = getattr(
            usage, "prompt_token_count", 0
        ) or 0

        output_count = getattr(
            usage, "candidates_token_count", 0
        ) or 0

        return LLMResult(
            text=response.text or "",
            input_tokens=input_count,
            output_tokens=output_count
        )


# Ollama Provider
class OllamaProvider(LLMProvider):

    provider_name = "Ollama"

    def __init__(self, model="llama3.2"):

        self.model = model

        self.url = "http://localhost:11434/api/generate"

    def generate(self, prompt):

        response = requests.post(
            self.url,
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False
            },
            timeout=(5, 60)
        )

        response.raise_for_status()

        data = response.json()

        return LLMResult(
            text=data.get("response", ""),
            input_tokens=data.get("prompt_eval_count", 0) or 0,
            output_tokens=data.get("eval_count", 0) or 0
        )


# Common LLM Client
class LLMClient:

    def __init__(self, provider, usage_logger):

        self.provider = provider
        self.usage_logger = usage_logger

    def generate(self, prompt):

        start = time.perf_counter()

        max_attempts = 3

        for attempt in range(max_attempts):

            try:

                # Generate response
                result = self.provider.generate(prompt)

                latency_ms = round(
                    (time.perf_counter() - start) * 1000,
                    2
                )

                # Read Gemini pricing from environment
                input_rate = float(
                    os.getenv(
                        "GEMINI_INPUT_PRICE_PER_MILLION",
                        "0"
                    )
                )

                output_rate = float(
                    os.getenv(
                        "GEMINI_OUTPUT_PRICE_PER_MILLION",
                        "0"
                    )
                )

                # Calculate estimated cost
                if self.provider.provider_name == "Gemini":

                    cost = (
                        result.input_tokens * input_rate
                        + result.output_tokens * output_rate
                    ) / 1_000_000

                    cost_type = "estimated_hosted_api_cost"

                else:

                    # Ollama runs locally
                    cost = 0.0

                    cost_type = (
                        "local_inference_cost_not_included"
                    )

                # Save successful request
                self.usage_logger.log_call({

                    "provider": self.provider.provider_name,

                    "model": self.provider.model,

                    "input_tokens": result.input_tokens,

                    "output_tokens": result.output_tokens,

                    "total_tokens": (
                        result.input_tokens
                        + result.output_tokens
                    ),

                    "latency_ms": latency_ms,

                    "estimated_cost_usd": round(cost, 8),

                    "cost_type": cost_type,

                    "status": "success",

                    "attempts": attempt + 1

                })

                print(f"Latency: {latency_ms} ms")

                print(
                    f"Input tokens: {result.input_tokens}"
                )

                print(
                    f"Output tokens: {result.output_tokens}"
                )

                print(
                    f"Estimated cost (USD): {cost}"
                )

                return result.text

            except Exception as error:

                status = getattr(error, "status_code", None)

                if status is None:
                    status = getattr(error, "code", None)

                # Handle HTTP errors from Ollama
                if status is None:
                    response = getattr(error, "response", None)

                    if response is not None:
                        status = getattr(
                            response, "status_code", None
                        )

                error_name = type(error).__name__.lower()

                # Identify temporary errors
                retryable = (
                    isinstance(
                        error,
                        (
                            requests.exceptions.Timeout,
                            requests.exceptions.ConnectionError
                        )
                    )
                    or status in (
                        408, 429, 500, 502, 503, 504
                    )
                    or "timeout" in error_name
                    or "unavailable" in error_name
                    or "resourceexhausted" in error_name
                )

                # Log final failure
                if not retryable or attempt == max_attempts - 1:

                    latency_ms = round(
                        (time.perf_counter() - start) * 1000,
                        2
                    )

                    self.usage_logger.log_call({

                        "provider": self.provider.provider_name,

                        "model": self.provider.model,

                        "input_tokens": None,

                        "output_tokens": None,

                        "total_tokens": None,

                        "latency_ms": latency_ms,

                        "estimated_cost_usd": None,

                        "status": "failed",

                        "attempts": attempt + 1,

                        "error": str(error)

                    })

                    raise

                # Retry with exponential backoff
                delay = 2 ** attempt

                print(
                    f"Temporary error. Retry "
                    f"{attempt + 1} of {max_attempts - 1} "
                    f"in {delay} seconds."
                )

                time.sleep(delay)