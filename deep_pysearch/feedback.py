#!/usr/bin/env python3

from typing import List

from llm_provider import generate_with_schema, get_model
from system_prompt import system_prompt

async def generate_feedback(
    query: str,
    num_questions: int = 3
) -> List[str]:
    """
    Generate follow-up questions to clarify the research direction.

    Args:
        query: The user's initial research query
        num_questions: Maximum number of follow-up questions to generate

    Returns:
        A list of follow-up questions
    """
    model = await get_model()

    prompt = f"""Given the following query from the user, ask some follow up questions to clarify the research direction. Return a maximum of {num_questions} questions, but feel free to return less if the original query is clear: <query>{query}</query>"""

    schema = {
        "type": "object",
        "properties": {
            "questions": {
                "type": "array",
                "description": f"Follow up questions to clarify the research direction, max of {num_questions}",
                "items": {"type": "string"}
            }
        },
        "required": ["questions"]
    }

    result = await generate_with_schema(model, prompt, schema)

    return result["questions"][:num_questions]
