# pip install pandas langchain langchain-community sentence-transformers faiss-cpu --upgrade
#
import os
from dotenv import load_dotenv
from glob import glob
from tqdm import tqdm
from langchain_community.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy
from func_ranges import get_function_ranges
from openai import OpenAI
import re
import csv
import time
import random

# Load environment variables from .env file
load_dotenv()

class Agent:
    def __init__(self, system_prompt=None, model=None):
        self.client = OpenAI(
            api_key=os.environ.get("API_KEY", ""),
            base_url=os.environ.get("BASE_URL", "https://api.openai.com/v1")
        )
        self.model = model or os.environ.get("MODEL", "gpt-4-0125-preview")
        self.system_prompt = system_prompt or os.environ.get("SYSTEM_PROMPT", "You are a helpful assistant specializing in code analysis.")

    def generate_response(self, user_prompt):
        # Check if API key is provided
        if not os.environ.get("API_KEY") and not self.client.api_key:
            print("Warning: No API_KEY provided in environment variables or constructor.")
            print("Please set API_KEY in your .env file or provide it when initializing the Agent.")
            return "Error: No API key provided. Set API_KEY in .env file or provide it when initializing."

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        # Implement backoff strategy for rate limiting
        max_retries = 5
        retry_count = 0
        base_delay = 5  # Start with 5 seconds delay

        while retry_count < max_retries:
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.5
                )
                return response.choices[0].message.content
            except Exception as e:
                retry_count += 1
                if retry_count >= max_retries:
                    print(f"Failed after {max_retries} attempts. Error: {str(e)}")
                    return f"Error: {str(e)}"

                # Calculate exponential backoff with jitter
                delay = base_delay * (2 ** (retry_count - 1)) + random.uniform(0, 1)
                print(f"API error: {str(e)}. Retrying in {delay:.2f} seconds (attempt {retry_count}/{max_retries})...")
                time.sleep(delay)
                print("Retrying now...")

class Reviewer(Agent):
    def __init__(self, model=os.environ["MODEL"]):
        system_prompt = """You are an expert code reviewer specializing in C programming, networking protocols, and MDNS implementation.
Your expertise includes:
- Deep understanding of C language patterns, memory management, and optimization techniques
- Comprehensive knowledge of network programming, socket APIs, and protocol implementation
- Specific expertise in multicast DNS (MDNS) protocol specifications, service discovery, and Zero-configuration networking
- Ability to identify refactoring patterns, code structure changes, and function renaming
- Experience analyzing complex codebases and tracing function relationships

When reviewing code changes between original and refactored implementations:
1. Focus on functional equivalence rather than syntactic differences
2. Identify when functions are split, merged, renamed, or otherwise restructured
3. Pay special attention to error handling, resource management, and protocol-specific logic
4. Consider implementation details specific to embedded systems and constrained environments

Provide clear, precise answers with high confidence when possible, or structured analytical responses when further investigation is needed."""

        super().__init__(system_prompt=system_prompt, model=model)

# Function to load all C and header files as functions using func_ranges.py
def load_functions_from_files(directory_path):
    # Get all .c and .h files from the directory
    c_files = glob(os.path.join(directory_path, '**/*.c'), recursive=True)
    h_files = glob(os.path.join(directory_path, '**/*.h'), recursive=True)
    file_paths = c_files + h_files
    documents = []

    for file_path in tqdm(file_paths, desc="Processing files"):
        try:
            # Get function ranges and content using the updated func_ranges.py
            function_ranges = get_function_ranges(file_path)

            # Create a Document for each function
            for func_name, (start_line, end_line, func_content) in function_ranges.items():
                doc = Document(
                    page_content=func_content,
                    metadata={
                        "source": os.path.basename(file_path),
                        "function": func_name,
                        "start_line": start_line,
                        "end_line": end_line
                    }
                )
                documents.append(doc)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    return documents

class FunctionEmbedder:
    def __init__(self, model_name=None, save_directory="db"):
        self.model_name = model_name or os.environ.get("EMBEDDING_MODEL", "thenlper/gte-small")
        self.save_directory = save_directory
        self.embedding_model = HuggingFaceEmbeddings(model_name=self.model_name)
        self.vectordb = self._load_or_create_db()

    def _load_or_create_db(self):
        # Check if the database already exists
        if os.path.exists(self.save_directory):
            print(f"Loading existing vector database from {self.save_directory}...")
            vectordb = FAISS.load_local(self.save_directory, self.embedding_model, allow_dangerous_deserialization=True)
            print("Vector store loaded successfully!")
            return vectordb
        else:
            return None

    def create_db_from_directory(self, source_dir):
        # Create new database if it doesn't exist
        print(f"Loading functions from {source_dir}...")
        docs_processed = load_functions_from_files(source_dir)
        print(f"Loaded {len(docs_processed)} functions as documents")

        # Filter out any duplicates if needed
        unique_texts = {}
        filtered_docs = []
        for doc in docs_processed:
            if doc.page_content not in unique_texts:
                unique_texts[doc.page_content] = True
                filtered_docs.append(doc)

        if len(filtered_docs) < len(docs_processed):
            print(f"Filtered out {len(docs_processed) - len(filtered_docs)} duplicate functions")
            docs_processed = filtered_docs

        # Build the vector store using FAISS with cosine similarity
        print("Embedding functions... This may take several minutes.")
        self.vectordb = FAISS.from_documents(
            documents=docs_processed,
            embedding=self.embedding_model,
            distance_strategy=DistanceStrategy.COSINE,
        )

        print("Vector store created successfully!")
        self.vectordb.save_local(self.save_directory)
        print(f"Vector store saved to {self.save_directory}")
        return self.vectordb

    def search_functions(self, query, top_k=5):
        """
        Search for functions related to the query in the vector database.

        Args:
            query (str): The search query
            top_k (int): Number of results to return

        Returns:
            Tuple containing:
            - List of function documents with their metadata
            - List of formatted result strings for display
        """
        if not self.vectordb:
            print("No vector database loaded. Please create one first.")
            return [], []

        # Perform the search using the already loaded vectordb with scores
        docs_and_scores = self.vectordb.similarity_search_with_score(query, k=top_k)

        # Create formatted results list
        formatted_results = []

        # Display and collect the results
        print(f"\nTop {top_k} results for query: '{query}'\n")
        for i, (doc, score) in enumerate(docs_and_scores):
            # Convert score to similarity percentage (FAISS returns distance, so 1-distance for similarity)
            # Cosine distance is between 0-2, where 0 is identical and 2 is opposite
            # Convert to 0-100% scale where 100% is identical
            similarity_pct = (1 - (score / 2)) * 100

            # Build result string
            result_str = f"Result {i+1}: [Similarity: {similarity_pct:.2f}%]\n"
            result_str += f"  Function: {doc.metadata['function']}\n"
            result_str += f"  Source: {doc.metadata['source']} (lines {doc.metadata['start_line']}-{doc.metadata['end_line']})\n"
            result_str += f"  Content:\n{doc.page_content}\n"
            result_str += "-" * 80

            # Add to formatted results
            formatted_results.append(result_str)

            # Print for immediate feedback
            print(result_str)

        # Return both the docs for backward compatibility and the formatted results
        return [doc for doc, _ in docs_and_scores], formatted_results

# --- Define the full-text search tool ---
def full_text_search(query: str, directory: str, max_results: int = 20) -> str:
    """Searches for a query string in all *.c and *.h files under the given directory."""
    results = []
    context_lines = 3  # Number of lines to show before and after the match

    # Also look for inline function definition patterns if the query appears to be a function name
    inline_patterns = [
        f"static\\s+inline\\s+[\\w\\*]+\\s+{query}\\s*\\(",  # static inline return_type function_name(
        f"inline\\s+static\\s+[\\w\\*]+\\s+{query}\\s*\\(",  # inline static return_type function_name(
        f"IRAM_ATTR\\s+[\\w\\*]+\\s+{query}\\s*\\(",         # IRAM_ATTR return_type function_name(
        f"INLINE_FN\\s+[\\w\\*]+\\s+{query}\\s*\\("          # INLINE_FN return_type function_name(
    ]

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

                        # First look for exact matches
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

                                # Check if we've reached the maximum number of results
                                if len(results) >= max_results:
                                    break_message = f"Reached maximum of {max_results} results. Consider refining your search."
                                    return "\n".join(results) + f"\n\n{break_message}"

                        # For function names, also look for inline function definitions using regex
                        if len(query.split()) == 1:  # Likely a function name if it's a single word
                            file_content = ''.join(lines)
                            for pattern in inline_patterns:
                                for match in re.finditer(pattern, file_content, re.MULTILINE | re.IGNORECASE):
                                    match_pos = match.start()

                                    # Find the line number of the match
                                    line_num = file_content[:match_pos].count('\n')

                                    # Calculate start and end indices for context
                                    start_idx = max(0, line_num - context_lines)
                                    end_idx = min(len(lines), line_num + context_lines + 1)

                                    # Get the context lines
                                    context = lines[start_idx:end_idx]

                                    # Format the output with line numbers
                                    context_str = "".join([
                                        f"{j+1:4d} | {line.rstrip()}\n"
                                        for j, line in enumerate(context, start=start_idx)
                                    ])

                                    results.append(f"Found inline function in: {filepath}\n{context_str}")

                                    # Check if we've reached the maximum number of results
                                    if len(results) >= max_results:
                                        break_message = f"Reached maximum of {max_results} results. Consider refining your search."
                                        return "\n".join(results) + f"\n\n{break_message}"
                except Exception as e:
                    # Only report errors if the file actually exists
                    if os.path.exists(filepath):
                        results.append(f"Error reading {filepath}: {e}")

                # Break out of file loop if max results reached
                if len(results) >= max_results:
                    break

        # Break out of directory loop if max results reached
        if len(results) >= max_results:
            break

    return "\n".join(results) if results else "No matches found."

# Example usage
if __name__ == "__main__":
    # Check for environment variables
    if not os.environ.get("API_KEY"):
        print("Warning: No API_KEY found in environment variables.")
        print("Copy .env.example to .env and add your API key to proceed with AI code review.")
        print("You can still use embedding and search functionality without an API key.")
        print()

    # Test code for examining header functions - Can be commented out for normal operation
    # function_ranges = get_function_ranges("/home/david/repos/proto/components/mdns/private_include/mdns_utils.h")
    # for func_name, (start_line, end_line, func_content) in function_ranges.items():
    #     print(f"Function: {func_name} (lines {start_line}-{end_line})")
    # exit()

    # Example search
    print("\nRunning example search...")
    # full_search = full_text_search("mdns_init", "/home/david/repos/proto/components/mdns")
    # print(full_search)
    # exit()

    # Set paths from environment variables or use defaults
    original_code_path = os.environ.get("ORIGINAL_CODE_PATH", "/home/david/repos/proto/components/mdns_old")
    refactored_code_path = os.environ.get("REFACTORED_CODE_PATH", "/home/david/repos/proto/components/mdns")

    # Create embedder instance
    embedder = FunctionEmbedder()

    # If database doesn't exist, create it
    if not embedder.vectordb:
        embedder.create_db_from_directory(refactored_code_path)

    # Process both .c and .h files
    # Define file types to process
    files_to_process = [ ] #os.path.join(original_code_path, 'mdns.c') ]

    # Add .c files
    c_files = glob(os.path.join(original_code_path, '**/*.c'), recursive=True)
    files_to_process.extend(c_files)

    # # Add .h files
    h_files = glob(os.path.join(original_code_path, '**/*.h'), recursive=True)
    files_to_process.extend(h_files)

    # Keep track of all mappings
    all_mappings = []

    # Process each file
    for file_path in files_to_process:
        print(f"\nProcessing file: {file_path}")
        try:
            function_ranges = get_function_ranges(file_path)

            if not function_ranges:
                print(f"No functions found in {file_path}")
                continue

            # Create a Document for each function
            for func_name, (start_line, end_line, func_content) in function_ranges.items():
                print(f"\nFunction(lines {start_line}-{end_line}): {func_name}")
                # if func_name != "mdns_if_from_preset_if":
                #     continue

                # Search for original function references
                full_text = full_text_search(func_name, original_code_path)
                original_context = f"""
Please review the following refactoring of the function {func_name}. Mostly code structure changes and renaming.
The below context shows the original function and the refactored code containing multiple functions that might replace the original function.
Your goal is to find the actual refactored function. If unsure, please summarize what you learned and give follow up questions.
In the next iteration, you can run contextual search or full-text search of the original and refactored code.

Format your response as follows:
* If you have 95% confidence, just state the name of the function you think is the refactored function. Do not say anything else just the function name withing xml tags, for example:
```xml
<refactored_function>
mdns_init
</refactored_function>
```
* If you have 95% confidence that the original function is not present in the refactored code, just say empty xml tags:
```xml
<refactored_function>
</refactored_function>
```
* If you have 95% confidence that the original function has been split into several functions, just say the refactored function names within xml tags:
```xml
<refactored_function>
mdns_init_internal
mdns_receiver_init
</refactored_function>
```
* Otherwise, summarize what you have learned and give follow up questions and search queries, within xml tags, for example:
```xml
<summary>
I learned that the function mdns_init might have been split into multiple functions, mdns_init_internal and mdns_receiver_init.
</summary>
<follow_up>
You have to verify if mdns_init_internal and mdns_receiver_init are the replacements for the original function.
</follow_up>
<search_original>
mdns_init
</search_original>
<search_refactored>
mdns_init_internal
mdns_receiver_init
</search_refactored>
```

## Original function

```c
{func_content}
```

### Function references in the original codebase
{full_text}

"""
                # Search for similar functions in the refactored codebase
                docs, formatted_results = embedder.search_functions(func_content, 3)
                context = ""
                for result in formatted_results:
                    context += result
                refactored_context = f"""
### Refactored code

{context}
"""
                # print(original_context)
                # print(refactored_context)

                # Create a reviewer and generate a response
                reviewer = Reviewer()
                full_prompt = original_context + refactored_context
                review_response = reviewer.generate_response(full_prompt)
                print("\n=== REVIEWER RESPONSE ===")
                print(review_response)

                # Skip further processing if we got an error
                if review_response.startswith("Error:"):
                    print("Received error response. Continuing to next function.")

                    # Write to CSV file with a special ERROR marker
                    csv_file = "refactoring.csv"
                    csv_exists = os.path.exists(csv_file)

                    with open(csv_file, 'a', newline='') as f:
                        writer = csv.writer(f, delimiter=';')

                        # Write header if file doesn't exist
                        if not csv_exists:
                            writer.writerow(['original_func_name', 'refactored_func_name'])

                        # Write the entry with "ERROR" to indicate API error
                        writer.writerow([func_name, "ERROR"])
                        print(f"Added error mapping: {func_name} → ERROR")
                        all_mappings.append((func_name, "ERROR"))

                    continue

                # Post-process the reviewer's response to extract refactored function names
                match = re.search(r'<refactored_function>(.*?)</refactored_function>', review_response, re.DOTALL)

                if match:
                    # Get function names (might be multiple functions separated by newlines)
                    content = match.group(1).strip()
                    refactored_func_names = [name.strip() for name in content.split('\n') if name.strip()]

                    # Write to CSV file
                    csv_file = "refactoring.csv"
                    csv_exists = os.path.exists(csv_file)

                    with open(csv_file, 'a', newline='') as f:
                        writer = csv.writer(f, delimiter=';')

                        # Write header if file doesn't exist
                        if not csv_exists:
                            writer.writerow(['original_func_name', 'refactored_func_name'])

                        # Write each refactored function
                        for refactored_name in refactored_func_names:
                            writer.writerow([func_name, refactored_name])
                            print(f"Added mapping: {func_name} → {refactored_name}")
                            all_mappings.append((func_name, refactored_name))
                else:
                    print("No refactored function found in the response. Continuing with next function.")
                    # Now let's try to add additional context to the prompt
                    summary = None
                    search_original_terms = None
                    search_refactored_terms = None
                    follow_up_questions = None

                    # Extract summary if available
                    match = re.search(r'<summary>(.*?)</summary>', review_response, re.DOTALL)
                    if match:
                        summary = match.group(1).strip()
                        print(f"Summary: {summary}")

                    # Extract follow-up questions if available
                    match = re.search(r'<follow_up>(.*?)</follow_up>', review_response, re.DOTALL)
                    if match:
                        follow_up_questions = match.group(1).strip()
                        print(f"Follow-up questions: {follow_up_questions}")

                    # Extract search terms for original code
                    match = re.search(r'<search_original>(.*?)</search_original>', review_response, re.DOTALL)
                    if match:
                        search_original_terms = match.group(1).strip()
                        print(f"Search terms for original code: {search_original_terms}")

                    # Extract search terms for refactored code
                    match = re.search(r'<search_refactored>(.*?)</search_refactored>', review_response, re.DOTALL)
                    if match:
                        search_refactored_terms = match.group(1).strip()
                        print(f"Search terms for refactored code: {search_refactored_terms}")

                    # If we have search terms, try up to 3 more iterations with additional context
                    if summary and (search_original_terms or search_refactored_terms):
                        max_retries = 3
                        retry_count = 0

                        while retry_count < max_retries:
                            retry_count += 1
                            print(f"\n=== RETRY ATTEMPT {retry_count}/{max_retries} ===")

                            additional_context = f"""
## Additional context from previous analysis

### Summary of previous findings
{summary}

"""
                            # Add follow-up questions if available
                            if follow_up_questions:
                                additional_context += f"""
### Questions to consider
{follow_up_questions}

"""

                            # Search for original terms if provided
                            if search_original_terms:
                                original_search_results = full_text_search(search_original_terms, original_code_path)
                                additional_context += f"""
### Additional search results from original codebase for "{search_original_terms}"
{original_search_results}

"""

                            # Search for refactored terms if provided
                            if search_refactored_terms:
                                refactored_search_results = full_text_search(search_refactored_terms, refactored_code_path)
                                additional_context += f"""
### Additional search results from refactored codebase for "{search_refactored_terms}"
{refactored_search_results}

"""

                            # Add a reminder about the confidence requirement
                            additional_context += """
Based on this additional context, please reconsider your analysis and provide a more confident answer if possible.
Remember, only use the <refactored_function> tag if you have 95% confidence in your determination.
"""

                            # Generate a new response with the additional context
                            enhanced_prompt = full_prompt + additional_context
                            print("Generating new response with additional context...")
                            review_response = reviewer.generate_response(enhanced_prompt)
                            print(f"\n=== REVIEWER RESPONSE (RETRY {retry_count}) ===")
                            print(review_response)

                            # If we got an error response, check if we should continue or stop
                            if review_response.startswith("Error:"):
                                print(f"Received error response on retry {retry_count}. Will try again in next iteration.")
                                # Update the retry counter but continue with the loop
                                continue

                            # Check if we found a refactored function this time
                            match = re.search(r'<refactored_function>(.*?)</refactored_function>', review_response, re.DOTALL)
                            if match:
                                # We found a match! Process it
                                content = match.group(1).strip()
                                refactored_func_names = [name.strip() for name in content.split('\n') if name.strip()]

                                # Write to CSV file
                                csv_file = "refactoring.csv"
                                csv_exists = os.path.exists(csv_file)

                                with open(csv_file, 'a', newline='') as f:
                                    writer = csv.writer(f, delimiter=';')

                                    # Write header if file doesn't exist
                                    if not csv_exists:
                                        writer.writerow(['original_func_name', 'refactored_func_name'])

                                    # Write each refactored function
                                    for refactored_name in refactored_func_names:
                                        writer.writerow([func_name, refactored_name])
                                        print(f"Added mapping: {func_name} → {refactored_name}")
                                        all_mappings.append((func_name, refactored_name))

                                # Found a match, break out of the retry loop
                                break

                            # Extract new information for the next iteration if needed
                            new_summary = re.search(r'<summary>(.*?)</summary>', review_response, re.DOTALL)
                            if new_summary:
                                summary = new_summary.group(1).strip()

                            new_follow_up = re.search(r'<follow_up>(.*?)</follow_up>', review_response, re.DOTALL)
                            if new_follow_up:
                                follow_up_questions = new_follow_up.group(1).strip()

                            new_search_original = re.search(r'<search_original>(.*?)</search_original>', review_response, re.DOTALL)
                            if new_search_original:
                                search_original_terms = new_search_original.group(1).strip()

                            new_search_refactored = re.search(r'<search_refactored>(.*?)</search_refactored>', review_response, re.DOTALL)
                            if new_search_refactored:
                                search_refactored_terms = new_search_refactored.group(1).strip()

                        if retry_count >= max_retries:
                            print(f"Maximum retry attempts ({max_retries}) reached without finding a confident mapping.")
                    else:
                        print("No sufficient information for additional searches. Moving to next function.")

                    # Write to CSV file with a special marker to indicate no mapping found
                    csv_file = "refactoring.csv"
                    csv_exists = os.path.exists(csv_file)

                    with open(csv_file, 'a', newline='') as f:
                        writer = csv.writer(f, delimiter=';')

                        # Write header if file doesn't exist
                        if not csv_exists:
                            writer.writerow(['original_func_name', 'refactored_func_name'])

                        # Write the entry with "???" to indicate no mapping found
                        writer.writerow([func_name, "???"])
                        print(f"Added special mapping for unresolved function: {func_name} → ???")
                        all_mappings.append((func_name, "???"))

        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    # Print summary of all mappings found
    print("\n=== SUMMARY OF REFACTORING MAPPINGS ===")
    if all_mappings:
        for original, refactored in all_mappings:
            print(f"{original} → {refactored}")
        print(f"\nTotal mappings found: {len(all_mappings)}")
        print(f"Mappings saved to refactoring.csv")
    else:
        print("No refactoring mappings were found.")

    # Interactive mode example code (commented out)
    # while True:
    #     user_query = input("\nEnter a search query (or 'q' to quit): ")
    #     if user_query.lower() == 'q':
    #         break
    #     num_results = input("How many results? [5]: ")
    #     try:
    #         num_results = int(num_results) if num_results else 5
    #     except ValueError:
    #         num_results = 5
    #     docs, formatted_results = embedder.search_functions(user_query, num_results)
