# pip install pandas langchain langchain-community sentence-transformers faiss-cpu smolagents --upgrade
#
import os
from glob import glob
from tqdm import tqdm
from transformers import AutoTokenizer
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy

# Function to load all c files from lwip directory into Document objects
def load_text_files(directory_path):
    file_paths = glob(os.path.join(directory_path, '*.c'))
    documents = []
    for file_path in file_paths:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        # Use the filename as the source metadata
        doc = Document(page_content=text, metadata={"source": os.path.basename(file_path)})
        documents.append(doc)
    return documents

# Load the db
# embedding_model = HuggingFaceEmbeddings(model_name="thenlper/gte-small")
# vectordb = FAISS.load_local(save_directory, embedding_model)
# print("Vector store loaded successfully!")

# Replace with the path to your text files directory
source_docs = load_text_files("/home/david/esp/idf/components/lwip/lwip/src/core")

# Initialize a HuggingFace tokenizer and a text splitter that uses it
tokenizer = AutoTokenizer.from_pretrained("thenlper/gte-small")
text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
    tokenizer,
    chunk_size=200,
    chunk_overlap=20,
    add_start_index=True,
    strip_whitespace=True,
    separators=["\n\n", "\n", ".", " ", ""],
)

# Split the loaded documents into chunks while filtering out duplicates
print("Splitting documents...")
docs_processed = []
unique_texts = {}
for doc in tqdm(source_docs):
    new_docs = text_splitter.split_documents([doc])
    for new_doc in new_docs:
        if new_doc.page_content not in unique_texts:
            unique_texts[new_doc.page_content] = True
            docs_processed.append(new_doc)

# Build the vector store using FAISS with cosine similarity
print("Embedding documents... This may take several minutes.")
embedding_model = HuggingFaceEmbeddings(model_name="thenlper/gte-small")
vectordb = FAISS.from_documents(
    documents=docs_processed,
    embedding=embedding_model,
    distance_strategy=DistanceStrategy.COSINE,
)

print("Vector store created successfully!")
save_directory = "db"
vectordb.save_local(save_directory)
print("Vector store saved to", save_directory)

from smolagents import Tool
from langchain_core.vectorstores import VectorStore


class RetrieverTool(Tool):
    name = "retriever"
    # description = "Using semantic similarity, retrieves some documents from the knowledge base that have the closest embeddings to the input query."
    description = "Using semantic similarity, retrieves lwip core sources from the codebase that have the closest embeddings to the input query."
    inputs = {
        "query": {
            "type": "string",
            "description": "The query to perform. This should be semantically close to your target documents. Use the affirmative form rather than a question.",
        }
    }
    output_type = "string"

    def __init__(self, vectordb: VectorStore, **kwargs):
        super().__init__(**kwargs)
        self.vectordb = vectordb

    def forward(self, query: str) -> str:
        assert isinstance(query, str), "Your search query must be a string"

        docs = self.vectordb.similarity_search(
            query,
            k=7,
        )

        return "\nRetrieved documents:\n" + "".join(
            [f"===== Document {str(i)} =====\n" + doc.page_content for i, doc in enumerate(docs)]
        )

from smolagents import CodeAgent, DuckDuckGoSearchTool, VisitWebpageTool, OpenAIServerModel
from smolagents import LiteLLMModel, ToolCallingAgent
import os


model = LiteLLMModel(
    # model_id="deepseek-ai/DeepSeek-V3",
    model_id="deepseek/deepseek-reasoner", 
    api_base="https://api.deepseek.com",

    # "deepseek-ai/deepseek-chat",
    # temperature=0.2,
    api_key=os.environ["API_KEY"]
)

retriever_tool = RetrieverTool(vectordb)
# agent = ToolCallingAgent(tools=[retriever_tool], model=model, verbose=True)
agent = CodeAgent(
    tools=[retriever_tool], model=model, max_steps=4, verbosity_level=2
)

# agent = CodeAgent(tools=[DuckDuckGoSearchTool(), VisitWebpageTool()], model=model)
agent_output = agent.run("Does lwip support SACK (selective acknowledgements)? Please show on the code examples")

print("Final output:")
print(agent_output)