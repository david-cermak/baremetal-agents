import os
import re
from openai import OpenAI  # Assuming you have the OpenAI Python package

# --- OpenAI Client Setup ---
def create_openai_client():
    """Creates an OpenAI client using environment variables."""
    return OpenAI(
        api_key=os.environ["API_KEY"],
        base_url=os.environ["BASE_URL"]
    )

# --- Define a Tool class ---
class Tool:
    def __init__(self, name, function):
        self.name = name
        self.function = function

    def __call__(self, *args, **kwargs):
        return self.function(*args, **kwargs)

# --- Define the full-text search tool ---
def full_text_search(query: str, directory: str) -> str:
    """Searches for a query string in all *.c and *.h files under the given directory."""
    results = []
    context_lines = 3  # Number of lines to show before and after the match
    
    # Skip build directories and other common directories to ignore
    skip_dirs = {'build', 'build_esp32_default', '.git', 'cmake-build'}
    
    for root, dirs, files in os.walk(directory):
        # Skip build directories
        dirs[:] = [d for d in dirs if d not in skip_dirs]
        
        for file in files:
            if file.endswith(".c") or file.endswith(".h"):
                filepath = os.path.join(root, file)
                # Skip files in build directories
                if any(skip_dir in filepath for skip_dir in skip_dirs):
                    continue
                    
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        lines = f.readlines()
                        for i, line in enumerate(lines):
                            if query.lower() in line.lower():
                                # Calculate start and end indices for context
                                start_idx = max(0, i - context_lines)
                                end_idx = min(len(lines), i + context_lines + 1)
                                
                                # Get the context lines
                                context = lines[start_idx:end_idx]
                                
                                # Format the output with line numbers and highlight the match
                                context_str = "".join([
                                    f"{j+1:4d} | {line.rstrip()}\n"
                                    for j, line in enumerate(context, start=start_idx)
                                ])
                                
                                results.append(f"Found match in: {filepath}\n{context_str}")
                except Exception as e:
                    # Only report errors if the file actually exists
                    if os.path.exists(filepath):
                        results.append(f"Error reading {filepath}: {e}")
    return "\n".join(results) if results else "No matches found."

# Instantiate the search tool
search_tool = Tool("search", full_text_search)

# --- Base Agent class ---
class Agent:
    def __init__(self, client, model):
        self.client = client
        self.model = model

# --- Researcher Agent ---
class Researcher(Agent):
    def __init__(self, client, model, tool: Tool, directory: str):
        super().__init__(client, model)
        self.tool = tool
        self.directory = directory
    
    def research(self, query: str) -> str:
        """Uses the search tool to perform research on the given query."""
        return self.tool(query, self.directory)

# --- Summarizer Agent ---
class Summarizer(Agent):
    def summarize(self, text: str, query: str) -> str:
        """Summarizes the given text using the OpenAI API."""
        messages = [
            {
                "role": "system",
                "content": ( 
f"""
You are an experienced SW architect, that gives a summary of the partial code search
given this initial query:
{query}
"""
                )
            },
            {
                "role": "user",
                "content": f"Summarize the following text:\n\n{text}"
            } ]
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            # max_tokens=150
        )
        return response.choices[0].message.content

if __name__ == "__main__":
    directory = "/home/david/esp/idf/components/lwip"  # start with lwip
    query = "congestion"

    # Create separate OpenAI clients
    researcher_client = create_openai_client()
    summarizer_client = create_openai_client()

    researcher_model = os.environ["MODEL"]
    summarizer_model = os.environ["MODEL"]

    # Instantiate the agents
    researcher = Researcher(researcher_client, researcher_model, search_tool, directory)
    summarizer = Summarizer(summarizer_client, summarizer_model)

    # Perform research
    search_results = researcher.research(query)
    print(search_results)

    # Summarize results
    summary = summarizer.summarize(search_results, query)

    # Output results
    print("=== Search Results ===")
    print(search_results)
    print("\n=== Summary ===")
    print(summary)
