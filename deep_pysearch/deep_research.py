#!/usr/bin/env python3

import asyncio
import os
from typing import Dict, List, Any, Optional, Callable

from llm_provider import generate_with_schema, get_model, trim_prompt

# Function for consistent logging
def log(*args):
    print(*args)

# Type for research progress tracking
class ResearchProgress:
    def __init__(self, depth: int, breadth: int):
        self.current_depth = depth
        self.total_depth = depth
        self.current_breadth = breadth
        self.total_breadth = breadth
        self.current_query = None
        self.total_queries = 0
        self.completed_queries = 0

# Function to generate research queries based on the user's input
async def generate_research_queries(
    query: str,
    num_queries: int = 3,
    learnings: Optional[List[str]] = None
) -> List[Dict[str, str]]:
    """Generate research queries based on the user's input."""
    model = await get_model()

    learnings_text = ""
    if learnings:
        learnings_text = f"Here are some learnings from previous research, use them to generate more specific queries: {chr(10).join(learnings)}"

    prompt = f"""Given the following prompt from the user, generate a list of research queries to investigate the topic.
Return a maximum of {num_queries} queries, but feel free to return less if the original prompt is clear.
Make sure each query is unique and not similar to each other:

<prompt>{query}</prompt>

{learnings_text}"""

    schema = {
        "type": "object",
        "properties": {
            "queries": {
                "type": "array",
                "description": f"List of research queries, max of {num_queries}",
                "items": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The research query"
                        },
                        "researchGoal": {
                            "type": "string",
                            "description": "First talk about the goal of the research that this query is meant to accomplish, then go deeper into how to advance the research once the results are found, mention additional research directions. Be as specific as possible, especially for additional research directions."
                        }
                    },
                    "required": ["query", "researchGoal"]
                }
            }
        },
        "required": ["queries"]
    }

    result = await generate_with_schema(model, prompt, schema)
    log(f"Created {len(result['queries'])} queries", result["queries"])

    return result["queries"][:num_queries]

# Function to process search results
async def process_research_results(
    query: str,
    results: List[str],
    num_learnings: int = 3,
    num_follow_up_questions: int = 3
) -> Dict[str, Any]:
    """Process research results and extract learnings and follow-up questions."""
    model = await get_model()

    # Format the results for processing
    contents = [trim_prompt(content, 25000) for content in results if content]
    log(f"Processing research for '{query}', found {len(contents)} results")
    stuff=chr(10).join([f"<content>\n{content}\n</content>" for content in contents])
    prompt = f"""Given the following contents from a research query <query>{query}</query>,
generate a list of learnings from the contents. Return a maximum of {num_learnings} learnings,
but feel free to return less if the contents are clear. Make sure each learning is unique and not similar to each other.
The learnings should be concise and to the point, as detailed and information dense as possible.
Make sure to include any entities like people, places, companies, products, things, etc in the learnings,
as well as any exact metrics, numbers, or dates. The learnings will be used to research the topic further.

<contents>
{stuff}
</contents>"""

    schema = {
        "type": "object",
        "properties": {
            "learnings": {
                "type": "array",
                "description": f"List of learnings, max of {num_learnings}",
                "items": {"type": "string"}
            },
            "followUpQuestions": {
                "type": "array",
                "description": f"List of follow-up questions to research the topic further, max of {num_follow_up_questions}",
                "items": {"type": "string"}
            }
        },
        "required": ["learnings", "followUpQuestions"]
    }

    result = await generate_with_schema(model, prompt, schema)
    log(f"Created {len(result['learnings'])} learnings", result["learnings"])

    return result

# Mock function that would be replaced with actual local search functionality
async def search_local_data(query: str) -> List[str]:
    """
    Search local data sources based on a query.
    This is a placeholder that would be replaced with actual local search logic.
    """
    # This is where you'd implement your local search functionality
    # For example, searching files, databases, local knowledge bases, etc.
    log(f"Searching local data for: {query}")

    # Mock result - in real implementation, this would return actual data
    return [
        f"This is a mock result for the query: {query}. " +
        "In a real implementation, this would contain text from local files, " +
        "databases, or other local data sources.",

        f"This is another mock result for: {query}. " +
        "You would replace this with your actual local search implementation " +
        "that searches your computer or local data sources."
    ]

# Function to write the final report
async def write_final_report(
    prompt: str,
    learnings: List[str],
    sources: List[str]
) -> str:
    """Write a final research report based on the learnings."""
    model = await get_model()

    learnings_text = chr(10).join([f"<learning>\n{learning}\n</learning>" for learning in learnings])

    prompt_text = f"""Given the following prompt from the user, write a final report on the topic using the learnings from research.
Make it as detailed as possible, aim for 3 or more pages, include ALL the learnings from research:

<prompt>{prompt}</prompt>

Here are all the learnings from previous research:

<learnings>
{learnings_text}
</learnings>"""

    schema = {
        "type": "object",
        "properties": {
            "reportMarkdown": {
                "type": "string",
                "description": "Final report on the topic in Markdown"
            }
        },
        "required": ["reportMarkdown"]
    }

    result = await generate_with_schema(model, prompt_text, schema)

    # Append the sources section to the report
    sources_section = f"\n\n## Sources\n\n{chr(10).join([f'- {source}' for source in sources])}"

    return result["reportMarkdown"] + sources_section

# Function to write a concise answer
async def write_final_answer(
    prompt: str,
    learnings: List[str]
) -> str:
    """Write a concise final answer based on the learnings."""
    model = await get_model()

    learnings_text = chr(10).join([f"<learning>\n{learning}\n</learning>" for learning in learnings])

    prompt_text = f"""Given the following prompt from the user, write a final answer on the topic using the learnings from research.
Follow the format specified in the prompt. Do not yap or babble or include any other text than the answer besides the format specified in the prompt.
Keep the answer as concise as possible - usually it should be just a few words or maximum a sentence.
Try to follow the format specified in the prompt (for example, if the prompt is using Latex, the answer should be in Latex.
If the prompt gives multiple answer choices, the answer should be one of the choices).

<prompt>{prompt}</prompt>

Here are all the learnings from research on the topic that you can use to help answer the prompt:

<learnings>
{learnings_text}
</learnings>"""

    schema = {
        "type": "object",
        "properties": {
            "exactAnswer": {
                "type": "string",
                "description": "The final answer, make it short and concise, just the answer, no other text"
            }
        },
        "required": ["exactAnswer"]
    }

    result = await generate_with_schema(model, prompt_text, schema)

    return result["exactAnswer"]

# Main deep research function
async def deep_research(
    query: str,
    breadth: int,
    depth: int,
    learnings: List[str] = None,
    sources: List[str] = None,
    on_progress: Callable[[ResearchProgress], None] = None
) -> Dict[str, List[str]]:
    """
    Perform deep research by iteratively generating queries, searching local data,
    and processing results to generate learnings.
    """
    learnings = learnings or []
    sources = sources or []

    # Initialize progress tracking
    progress = ResearchProgress(depth, breadth)

    def report_progress(update: Dict[str, Any]):
        for key, value in update.items():
            setattr(progress, key, value)
        if on_progress:
            on_progress(progress)

    # Generate research queries based on the initial query
    research_queries = await generate_research_queries(
        query=query,
        num_queries=breadth,
        learnings=learnings
    )

    report_progress({
        "total_queries": len(research_queries),
        "current_query": research_queries[0]["query"] if research_queries else None
    })

    # Process each research query
    all_results = []
    for i, research_query in enumerate(research_queries):
        try:
            # Search local data sources for information
            search_results = await search_local_data(research_query["query"])

            # Process the search results to extract learnings and follow-up questions
            processed_results = await process_research_results(
                query=research_query["query"],
                results=search_results,
                num_follow_up_questions=max(1, breadth // 2)
            )

            # Update the lists of learnings and sources
            new_learnings = processed_results["learnings"]
            all_learnings = list(set(learnings + new_learnings))
            all_sources = list(set(sources + [f"Local search for: {research_query['query']}"]))

            # If we still have depth to go, continue researching
            new_breadth = max(1, breadth // 2)
            new_depth = depth - 1

            if new_depth > 0:
                log(f"Researching deeper, breadth: {new_breadth}, depth: {new_depth}")

                report_progress({
                    "current_depth": new_depth,
                    "current_breadth": new_breadth,
                    "completed_queries": progress.completed_queries + 1,
                    "current_query": research_query["query"]
                })

                next_query = f"""
Previous research goal: {research_query["researchGoal"]}
Follow-up research directions: {chr(10).join(processed_results["followUpQuestions"])}
                """.strip()

                # Recursive call to continue the research process
                deeper_results = await deep_research(
                    query=next_query,
                    breadth=new_breadth,
                    depth=new_depth,
                    learnings=all_learnings,
                    sources=all_sources,
                    on_progress=on_progress
                )

                all_results.append(deeper_results)
            else:
                report_progress({
                    "current_depth": 0,
                    "completed_queries": progress.completed_queries + 1,
                    "current_query": research_query["query"]
                })

                all_results.append({
                    "learnings": all_learnings,
                    "sources": all_sources
                })

        except Exception as e:
            log(f"Error processing query '{research_query['query']}': {str(e)}")
            all_results.append({
                "learnings": [],
                "sources": []
            })

    # Combine results from all research paths
    combined_learnings = list(set(sum([r.get("learnings", []) for r in all_results], [])))
    combined_sources = list(set(sum([r.get("sources", []) for r in all_results], [])))

    return {
        "learnings": combined_learnings,
        "sources": combined_sources
    }
