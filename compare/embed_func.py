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
import re
import csv
import time
import random

# Import our new modules
from models.reviewer import Reviewer
from utils.response_parser import ResponseParser

# Load environment variables from .env file
load_dotenv()

# Function to write function mapping to either CSV or markdown file
def write_mapping_to_file(original_func_name, refactored_func_name,
                          original_file=None, original_line=None,
                          refactored_file=None, refactored_line=None,
                          output_format="csv", function_locations=None, concern=None):
    """
    Write function mapping to a file in either CSV or markdown format.

    Args:
        original_func_name: Name of the original function
        refactored_func_name: Name of the refactored function (or "???" or "ERROR")
        original_file: Path to the original file containing the function
        original_line: Line number of the original function
        refactored_file: Path to the refactored file containing the function
        refactored_line: Line number of the refactored function
        output_format: "csv" or "markdown"
        function_locations: Dictionary mapping function names to their file:line locations
        concern: Any potential concerns about the refactoring (optional)
    """
    if output_format == "csv":
        # Write to CSV file
        csv_file = "refactoring.csv"
        csv_exists = os.path.exists(csv_file)

        with open(csv_file, 'a', newline='') as f:
            writer = csv.writer(f, delimiter=';')

            # Write header if file doesn't exist
            if not csv_exists:
                writer.writerow(['original_func_name', 'refactored_func_name', 'concern'])

            # Write the mapping entry
            writer.writerow([original_func_name, refactored_func_name, concern or ""])

        return f"Added mapping to CSV: {original_func_name} → {refactored_func_name}" + (f" (with concern)" if concern else "")

    elif output_format == "markdown":
        # Get repository SHA values from environment
        orig_sha = os.environ.get("ORIG_SHA", "main")
        new_sha = os.environ.get("NEW_SHA", "main")

        md_file = "refactoring.md"
        md_exists = os.path.exists(md_file)

        with open(md_file, 'a') as f:
            # Write header if file doesn't exist
            if not md_exists:
                f.write("| Original Function | Refactored Function | Concerns |\n")
                f.write("|------------------|--------------------|---------|\n")

            # Create markdown links if file and line info is available
            if original_file and original_line:
                original_link = f"[{original_func_name}](https://github.com/espressif/esp-protocols/blob/{orig_sha}/components/mdns/{original_file}#L{original_line})"
            else:
                original_link = original_func_name

            # Try to get refactored function location from the dictionary if available
            if refactored_func_name not in ["???", "ERROR"] and function_locations and refactored_func_name in function_locations:
                # Parse the location from the dictionary
                location_parts = function_locations[refactored_func_name].split(':')
                if len(location_parts) == 2:
                    ref_file, ref_line = location_parts
                    refactored_link = f"[{refactored_func_name}](https://github.com/espressif/esp-protocols/blob/{new_sha}/components/mdns/{ref_file}#L{ref_line})"
                else:
                    refactored_link = refactored_func_name
            elif refactored_func_name not in ["???", "ERROR"] and refactored_file and refactored_line:
                refactored_link = f"[{refactored_func_name}](https://github.com/espressif/esp-protocols/blob/{new_sha}/{refactored_file}#L{refactored_line})"
            else:
                refactored_link = refactored_func_name

            # Write the entry with concerns (if any)
            concern_text = concern or ""
            f.write(f"| {original_link} | {refactored_link} | {concern_text} |\n")

        return f"Added mapping to markdown: {original_func_name} → {refactored_func_name}" + (f" (with concern)" if concern else "")

    else:
        return f"Unsupported output format: {output_format}"

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

    # Set paths from environment variables or use defaults
    original_code_path = os.environ.get("ORIGINAL_CODE_PATH", "/home/david/repos/proto/components/mdns_old")
    refactored_code_path = os.environ.get("REFACTORED_CODE_PATH", "/home/david/repos/proto/components/mdns")
    output_format = os.environ.get("OUTPUT_FORMAT", "csv")

    # Create embedder instance
    embedder = FunctionEmbedder()

    # If database doesn't exist, create it
    if not embedder.vectordb:
        embedder.create_db_from_directory(refactored_code_path)

    # Process both .c and .h files
    # Define file types to process
    # files_to_process = [ ] #os.path.join(original_code_path, 'mdns.c') ]
    files_to_process = [ os.path.join(original_code_path, 'mdns.c') ]

    # Add .c files
    # c_files = glob(os.path.join(original_code_path, '**/*.c'), recursive=True)
    # files_to_process.extend(c_files)

    # # Add .h files
    # h_files = glob(os.path.join(original_code_path, '**/*.h'), recursive=True)
    # files_to_process.extend(h_files)

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
                if func_name != "_mdns_get_default_instance_name":
                    continue

                # Search for original function references
                original_context = full_text_search(func_name, original_code_path)

                # Search for similar functions in the refactored codebase
                docs, formatted_results = embedder.search_functions(func_content, 3)
                refactored_context = ""
                function_names_and_lines = {}

                # Create a dictionary mapping function names to their location information
                for i, result in enumerate(formatted_results):
                    refactored_context += result
                    if i < len(docs):  # Safety check to avoid index errors
                        source_file1 = docs[i].metadata.get('source', '')
                        start_line1 = docs[i].metadata.get('start_line', '')
                        function_name1 = docs[i].metadata.get('function', '')
                        function_names_and_lines[function_name1] = f"{source_file1}:{start_line1}"

                # Create a reviewer and build the initial prompt
                reviewer = Reviewer()
                initial_prompt = reviewer.build_initial_prompt(func_name, func_content, original_context, refactored_context, initial=True)
                print("\n=== INITIAL PROMPT ===")
                print(initial_prompt)
                # exit()

                # Generate the initial response
                review_response = reviewer.generate_response(initial_prompt)
                print("\n=== REVIEWER RESPONSE ===")
                print(review_response)

                # Skip further processing if we got an error
                if review_response.startswith("Error:"):
                    print("Received error response. Continuing to next function.")

                    # Use the function to write to file with ERROR marker
                    result_msg = write_mapping_to_file(
                        original_func_name=func_name,
                        refactored_func_name="ERROR",
                        original_file=os.path.relpath(file_path, original_code_path),
                        original_line=start_line,
                        output_format=output_format,
                        function_locations=function_names_and_lines,
                        concern=None
                    )
                    print(result_msg)
                    all_mappings.append((func_name, "ERROR", None))
                    continue

                # Parse the response
                parsed_response = reviewer.parse_response(review_response)
                refactored_func_names = parsed_response.get('refactored_function_names', [])
                concern = parsed_response.get('concern')

                # If we found refactored function names, process them
                if refactored_func_names:
                    for refactored_name in refactored_func_names:
                        # Write mapping to file
                        result_msg = write_mapping_to_file(
                            original_func_name=func_name,
                            refactored_func_name=refactored_name,
                            original_file=os.path.relpath(file_path, original_code_path),
                            original_line=start_line,
                            output_format=output_format,
                            function_locations=function_names_and_lines,
                            concern=concern
                        )
                        print(result_msg)
                        all_mappings.append((func_name, refactored_name, concern))
                else:
                    # We didn't find a confident answer, try additional iterations
                    summary = parsed_response.get('summary')
                    follow_up = parsed_response.get('follow_up')
                    search_original = parsed_response.get('search_original')
                    search_refactored = parsed_response.get('search_refactored')

                    # Try up to 3 more iterations with additional context if we have search terms
                    if summary and (search_original or search_refactored):
                        max_retries = 3
                        retry_count = 0

                        while retry_count < max_retries:
                            retry_count += 1
                            print(f"\n=== RETRY ATTEMPT {retry_count}/{max_retries} ===")

                            # Search for additional context if needed
                            original_search_results = None
                            refactored_search_results = None

                            if search_original:
                                original_search_results = full_text_search(search_original, original_code_path)

                            if search_refactored:
                                refactored_search_results = full_text_search(search_refactored, refactored_code_path)

                            # Build the follow-up prompt
                            initial_prompt = reviewer.build_initial_prompt(func_name, func_content, original_context, refactored_context, initial=False)
                            follow_up_prompt = reviewer.build_follow_up_prompt(
                                initial_prompt,
                                summary,
                                follow_up,
                                original_search_results,
                                refactored_search_results,
                                search_original,
                                search_refactored
                            )

                            # Generate a new response
                            print("Generating new response with additional context...")
                            review_response = reviewer.generate_response(follow_up_prompt)
                            print(f"\n=== REVIEWER RESPONSE (RETRY {retry_count}) ===")
                            print(review_response)

                            # Skip further processing if we got an error
                            if review_response.startswith("Error:"):
                                print(f"Received error response on retry {retry_count}. Will try again in next iteration.")
                                continue

                            # Parse the new response
                            parsed_response = reviewer.parse_response(review_response)
                            refactored_func_names = parsed_response.get('refactored_function_names', [])
                            concern = parsed_response.get('concern')

                            # If we found refactored function names, process them
                            if refactored_func_names:
                                for refactored_name in refactored_func_names:
                                    # Write mapping to file
                                    result_msg = write_mapping_to_file(
                                        original_func_name=func_name,
                                        refactored_func_name=refactored_name,
                                        original_file=os.path.relpath(file_path, original_code_path),
                                        original_line=start_line,
                                        output_format=output_format,
                                        function_locations=function_names_and_lines,
                                        concern=concern
                                    )
                                    print(result_msg)
                                    all_mappings.append((func_name, refactored_name, concern))

                                # Found an answer, break out of the retry loop
                                break

                            # Update search terms for next iteration if needed
                            summary = parsed_response.get('summary', summary)
                            follow_up = parsed_response.get('follow_up', follow_up)
                            search_original = parsed_response.get('search_original', search_original)
                            search_refactored = parsed_response.get('search_refactored', search_refactored)

                        # If we still don't have an answer after max retries, mark as unknown
                        if not refactored_func_names:
                            print(f"Maximum retry attempts ({max_retries}) reached without finding a confident mapping.")
                            # Mark as unknown
                            result_msg = write_mapping_to_file(
                                original_func_name=func_name,
                                refactored_func_name="???",
                                original_file=os.path.relpath(file_path, original_code_path),
                                original_line=start_line,
                                output_format=output_format,
                                function_locations=function_names_and_lines,
                                concern=None
                            )
                            print(result_msg)
                            all_mappings.append((func_name, "???", None))
                    else:
                        # Not enough information for additional searches
                        print("No sufficient information for additional searches. Marking as unknown.")
                        # Mark as unknown
                        result_msg = write_mapping_to_file(
                            original_func_name=func_name,
                            refactored_func_name="???",
                            original_file=os.path.relpath(file_path, original_code_path),
                            original_line=start_line,
                            output_format=output_format,
                            function_locations=function_names_and_lines,
                            concern=None
                        )
                        print(result_msg)
                        all_mappings.append((func_name, "???", None))

        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    # Print summary of all mappings found
    print("\n=== SUMMARY OF REFACTORING MAPPINGS ===")
    if all_mappings:
        for original, refactored, concern in all_mappings:
            if concern:
                print(f"{original} → {refactored} (Concern: {concern})")
            else:
                print(f"{original} → {refactored}")
        print(f"\nTotal mappings found: {len(all_mappings)}")
        if output_format == "csv":
            print(f"Mappings saved to refactoring.csv")
        elif output_format == "markdown":
            print(f"Mappings saved to refactoring.md")
        else:
            print(f"Mappings saved in {output_format} format")
    else:
        print("No refactoring mappings were found.")
