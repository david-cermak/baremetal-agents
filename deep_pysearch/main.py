#!/usr/bin/env python3

import os
from dotenv import load_dotenv
import asyncio

from llm_provider import get_model
from deep_research import deep_research, write_final_report
from feedback import generate_feedback

# Load environment variables
load_dotenv()

def log(*args):
    """Helper function for consistent logging"""
    print(*args)

async def main():
    """Run the research agent"""
    model_id = get_model()
    log(f"Using model: {model_id}")

    # Get initial query
    initial_query = """
Below is a description of a specific TCP/IP stack feature.
Your goal is to research the feature and check if it's implemented in lwip stack.
---
Congestion Control (RFC 5681)
Congestion control [RFC 5681] is required because network congestion, when unmitigated, can
cause packet loss, which severely reduces performance and degrades the user experience.
Although congestion control mechanisms have been known to degrade performance over
wireless networks, research has shown that the impact is far less severe over low-bandwidth
wireless networks such as IEEE 802.15.4.

"""
    # Get breadth and depth parameters
    breadth = 2
    depth = 2

    combined_query = initial_query

    log("\nStarting pysearch...\n")

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

    report = await write_final_report(
        prompt=combined_query,
        learnings=learnings,
        sources=sources
    )

    with open("report.md", "w", encoding="utf-8") as f:
        f.write(report)

    log(f"\n\nFinal Report:\n\n{report}")
    log("\nReport has been saved to report.md")

if __name__ == "__main__":
    asyncio.run(main())
