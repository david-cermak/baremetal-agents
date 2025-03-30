#!/usr/bin/env python3

import os
import json
from typing import Any, Dict, Optional
from datetime import datetime

from openai import OpenAI
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

def create_openai_client():
    """
    Creates an OpenAI client using environment variables.

    Returns:
        OpenAI client instance
    """
    return OpenAI(
        api_key=API_KEY,
        base_url=BASE_URL
    )

def get_model() -> Dict[str, Any]:
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

def generate_with_schema(
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

    # Create OpenAI client
    client = create_openai_client()

    # Prepare request
    messages = [
        {"role": "system", "content": system_prompt() + "\nPlease format your response as JSON, **exactly** according to the provided schema."},
        {"role": "user", "content": prompt + f"\n\nRespond using JSON format with this schema provided\n {schema}"}
    ]

    # Make API request
    response = client.chat.completions.create(
        model=model_id,
        messages=messages,
        temperature=temperature,
        response_format={"type": "json_object", "schema": schema}
    )

    try:
        # print(response)
        content = response.choices[0].message.content
        print(content)
        # The 'reasoning' attribute may not exist in all models/responses
        if hasattr(response.choices[0].message, 'reasoning'):
            print(response.choices[0].message.reasoning)

        # Improved JSON parsing that extracts JSON even if other text is present
        return extract_and_parse_json(content)
    except Exception as e:
        raise Exception(f"Failed to parse response: {str(e)}")

def extract_and_parse_json(text: str) -> Dict[str, Any]:
    """
    Extract and parse JSON from text that might contain other content.

    Args:
        text: String that may contain JSON

    Returns:
        Parsed JSON object
    """
    # Try direct parsing first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Look for JSON within markdown code blocks
    import re
    json_block_pattern = r"```(?:json)?\s*([\s\S]*?)```"
    matches = re.findall(json_block_pattern, text)

    for match in matches:
        try:
            return json.loads(match.strip())
        except json.JSONDecodeError:
            continue

    # Try to find JSON object by braces
    try:
        # Find the first opening brace
        start_idx = text.find('{')
        if start_idx >= 0:
            # Find the matching closing brace
            brace_count = 0
            for i in range(start_idx, len(text)):
                if text[i] == '{':
                    brace_count += 1
                elif text[i] == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        # Extract and parse the JSON object
                        json_str = text[start_idx:i+1]
                        return json.loads(json_str)
    except:
        pass

    # If all methods fail, raise exception
    raise json.JSONDecodeError("Failed to extract valid JSON from the response", text, 0)

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
