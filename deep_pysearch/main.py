#!/usr/bin/env python3

import os
from dotenv import load_dotenv
import asyncio

from llm_provider import get_model
from deep_research import deep_research, write_final_report, write_final_answer
from feedback import generate_feedback

# Load environment variables
load_dotenv()

def log(*args):
    """Helper function for consistent logging"""
    print(*args)

async def main():
    """Run the research agent"""
    model_id = await get_model()
    log(f"Using model: {model_id}")

    # Get initial query
    initial_query = input("What would you like to research? ")

    # Get breadth and depth parameters
    breadth_input = input("Enter research breadth (recommended 2-10, default 4): ")
    breadth = int(breadth_input) if breadth_input.strip() else 4

    depth_input = input("Enter research depth (recommended 1-5, default 2): ")
    depth = int(depth_input) if depth_input.strip() else 2

    report_type = input("Do you want to generate a long report or a specific answer? (report/answer, default report): ")
    is_report = report_type.lower() != "answer"

    combined_query = initial_query

    if is_report:
        log("Creating research plan...")

        # Generate follow-up questions
        follow_up_questions = await generate_feedback(initial_query)

        log("\nTo better understand your research needs, please answer these follow-up questions:")

        # Collect answers to follow-up questions
        answers = []
        for question in follow_up_questions:
            answer = input(f"\n{question}\nYour answer: ")
            answers.append(answer)

        # Combine all information for deep research
        stuff=chr(10).join([f"Q: {q}\nA: {a}" for q, a in zip(follow_up_questions, answers)])
        combined_query = f"""
Initial Query: {initial_query}
Follow-up Questions and Answers:
{stuff}
"""

    log("\nStarting research...\n")

    # Run the deep research process
    result = await deep_research(
        query=combined_query,
        breadth=breadth,
        depth=depth
    )

    learnings = result["learnings"]
    sources = result["sources"]  # Instead of visitedUrls since we're not using web search

    log(f"\n\nLearnings:\n\n{chr(10).join(learnings)}")
    log(f"\n\nSources ({len(sources)}):\n\n{chr(10).join(sources)}")
    log("Writing final report...")

    if is_report:
        report = await write_final_report(
            prompt=combined_query,
            learnings=learnings,
            sources=sources
        )

        with open("report.md", "w", encoding="utf-8") as f:
            f.write(report)

        log(f"\n\nFinal Report:\n\n{report}")
        log("\nReport has been saved to report.md")
    else:
        answer = await write_final_answer(
            prompt=combined_query,
            learnings=learnings
        )

        with open("answer.md", "w", encoding="utf-8") as f:
            f.write(answer)

        log(f"\n\nFinal Answer:\n\n{answer}")
        log("\nAnswer has been saved to answer.md")

if __name__ == "__main__":
    asyncio.run(main())
