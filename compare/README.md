# Code Refactoring Analyzer

This tool analyzes C code refactoring by identifying matching functions between original and refactored codebases using embeddings and AI review.

## Project Structure

```
.
├── embed_func.py           # Main script
├── func_ranges.py          # Function for extracting functions from C files
├── models/                 # Agent models
│   ├── __init__.py
│   ├── agent.py            # Base Agent class
│   └── reviewer.py         # Specialized Reviewer agent
└── utils/                  # Utility functions
    ├── __init__.py
    └── response_parser.py  # Parser for AI responses
```

## Setup

1. Create a `.env` file based on `.env.example` and add your API keys
2. Install the requirements:
```bash
pip install pandas langchain langchain-community sentence-transformers faiss-cpu openai python-dotenv tqdm
```

## Usage

1. Set the paths to your original and refactored code:
```bash
export ORIGINAL_CODE_PATH="/path/to/original/code"
export REFACTORED_CODE_PATH="/path/to/refactored/code"
```

2. Run the script:
```bash
python embed_func.py
```

The script will generate a mapping between original and refactored functions in either CSV or Markdown format.

## How It Works

1. The tool extracts functions from both codebases
2. For each function in the original codebase:
   - It generates semantic embeddings
   - Searches for similar functions in the refactored codebase
   - Uses a specialized AI model to determine the best match
   - Records the mapping with confidence levels

## Features

- Semantic code search using embeddings
- Full-text search for function references
- AI-powered code review to identify refactored functions
- Custom expert reviewer for C, networking, and MDNS code

## Classes

- `FunctionEmbedder`: Handles the semantic searching functionality
- `Agent`: Base class for AI-powered analysis
- `Reviewer`: Specialized agent for reviewing C/networking/MDNS code

## Customization

You can customize the behavior by:
- Modifying the system prompt in the `Reviewer` class
- Changing the embedding model in `FunctionEmbedder`
- Adjusting search parameters for more or fewer results
