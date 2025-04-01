# pip install pandas langchain langchain-community sentence-transformers faiss-cpu --upgrade
#
import os
from glob import glob
from tqdm import tqdm
from langchain.docstore.document import Document
from langchain.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy
import sys
from func_ranges import get_function_ranges

# Function to load all c files as functions using func_ranges.py
def load_functions_from_files(directory_path):
    file_paths = glob(os.path.join(directory_path, '*.c'))
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

# Create embedding model
embedding_model = HuggingFaceEmbeddings(model_name="thenlper/gte-small")
save_directory = "db"

# Check if the database already exists
if os.path.exists(save_directory):
    print(f"Loading existing vector database from {save_directory}...")
    vectordb = FAISS.load_local(save_directory, embedding_model, allow_dangerous_deserialization=True)
    print("Vector store loaded successfully!")
else:
    # Create new database if it doesn't exist
    # Replace with the path to your text files directory
    source_dir = "/home/david/repos/proto/components/mdns/"
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
    vectordb = FAISS.from_documents(
        documents=docs_processed,
        embedding=embedding_model,
        distance_strategy=DistanceStrategy.COSINE,
    )

    print("Vector store created successfully!")
    vectordb.save_local(save_directory)
    print(f"Vector store saved to {save_directory}")


# Function to perform searches
def search_functions(query, top_k=5):
    """
    Search for functions related to the query in the vector database.

    Args:
        query (str): The search query
        top_k (int): Number of results to return

    Returns:
        List of function documents with their metadata
    """
    # Perform the search using the already loaded vectordb with scores
    docs_and_scores = vectordb.similarity_search_with_score(query, k=top_k)

    # Display the results
    print(f"\nTop {top_k} results for query: '{query}'\n")
    for i, (doc, score) in enumerate(docs_and_scores):
        # Convert score to similarity percentage (FAISS returns distance, so 1-distance for similarity)
        # Cosine distance is between 0-2, where 0 is identical and 2 is opposite
        # Convert to 0-100% scale where 100% is identical
        similarity_pct = (1 - (score / 2)) * 100

        print(f"Result {i+1}: [Similarity: {similarity_pct:.2f}%]")
        print(f"  Function: {doc.metadata['function']}")
        print(f"  Source: {doc.metadata['source']} (lines {doc.metadata['start_line']}-{doc.metadata['end_line']})")
        print(f"  Content:\n{doc.page_content}")
        print("-" * 80)

    # Return just the docs for backward compatibility
    return [doc for doc, _ in docs_and_scores]

# Example usage
if __name__ == "__main__":
    # Example search
    print("\nRunning example search...")
    search_functions("""
static esp_err_t _udp_pcb_main_init(void)
{
    if (_pcb_main) {
        return ESP_OK;
    }
    _pcb_main = udp_new();
    if (!_pcb_main) {
        return ESP_ERR_NO_MEM;
    }
    if (udp_bind(_pcb_main, IP_ANY_TYPE, MDNS_SERVICE_PORT) != 0) {
        udp_remove(_pcb_main);
        _pcb_main = NULL;
        return ESP_ERR_INVALID_STATE;
    }
    _pcb_main->mcast_ttl = 255;
    _pcb_main->remote_port = MDNS_SERVICE_PORT;
    ip_addr_copy(_pcb_main->remote_ip, *(IP_ANY_TYPE));
    udp_recv(_pcb_main, &_udp_recv, NULL);
    return ESP_OK;
}""", 3)

    # Interactive mode
    # while True:
    #     user_query = input("\nEnter a search query (or 'q' to quit): ")
    #     if user_query.lower() == 'q':
    #         break
    #     num_results = input("How many results? [5]: ")
    #     try:
    #         num_results = int(num_results) if num_results else 5
    #     except ValueError:
    #         num_results = 5
    #     search_functions(user_query, num_results)
