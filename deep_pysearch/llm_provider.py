#!/usr/bin/env python3

import os
import asyncio
import json
from typing import Any, Dict, Optional
from datetime import datetime

import aiohttp
from dotenv import load_dotenv

from system_prompt import system_prompt

# Load environment variables
load_dotenv()

# Constants
MIN_CHUNK_SIZE = 140
DEFAULT_CONTEXT_SIZE = 128000

# Get configuration from environment variables
API_KEY = os.getenv("API_KEY")
BASE_URL = os.getenv("BASE_URL", "https://api.openai.com/v1")
MODEL = os.getenv("MODEL", "gpt-4o-mini")

async def get_model() -> Dict[str, Any]:
    """
    Returns the model configuration to use for LLM requests.

    Returns:
        Dict containing model info including model_id
    """
    model_id = MODEL

    # This is a placeholder for model version/capabilities detection
    # In a real implementation, you might check model capabilities
    return {
        "modelId": model_id,
        "endpoint": BASE_URL
    }

async def generate_with_schema(
    model: Dict[str, Any],
    prompt: str,
    schema: Dict[str, Any],
    temperature: float = 0.7
) -> Dict[str, Any]:
    """
    Generate a response from the LLM with structured output based on a JSON schema.

    Args:
        model: Model configuration dict
        prompt: The prompt to send to the model
        schema: JSON schema defining the structure of the expected response
        temperature: Controls randomness in generation

    Returns:
        Parsed JSON object matching the schema
    """
    model_id = model.get("modelId", MODEL)
    endpoint = model.get("endpoint", BASE_URL)

    # Prepare headers for API request
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }

    # Prepare request payload
    payload = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": system_prompt()},
            {"role": "user", "content": prompt}
        ],
        "temperature": temperature,
        "response_format": {"type": "json_object", "schema": schema}
    }

    # Make API request
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{endpoint}/chat/completions",
            headers=headers,
            json=payload
        ) as response:
            if response.status != 200:
                raise Exception(f"API request failed with status {response.status}: {await response.text()}")

            result = await response.json()
            try:
                print(result)
                content = result["choices"][0]["message"]["content"]
                print(content)
                # Parse JSON response
                return json.loads(content)
            except (KeyError, json.JSONDecodeError) as e:
                raise Exception(f"Failed to parse response: {str(e)}")

def trim_prompt(prompt: str, context_size: int = DEFAULT_CONTEXT_SIZE) -> str:
    """
    Trim a prompt to fit within the specified context size.

    Args:
        prompt: The prompt to trim
        context_size: Maximum number of characters to allow

    Returns:
        Trimmed prompt
    """
    if not prompt:
        return ""

    # Simple character-based trimming
    # In a production implementation, this would use proper tokenization
    if len(prompt) <= context_size:
        return prompt

    # If exceeding context size, trim to fit
    overflow = len(prompt) - context_size
    chunk_size = max(MIN_CHUNK_SIZE, len(prompt) - overflow)

    # Simple recursive trimming
    return trim_prompt(prompt[:chunk_size], context_size)
