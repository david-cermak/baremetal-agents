# Code Comparison Tool for Refactoring Analysis

This tool helps analyze refactored code by comparing original functions with their refactored counterparts using semantic searching and AI-powered analysis.

## Features

- Semantic code search using embeddings
- Full-text search for function references
- AI-powered code review to identify refactored functions
- Custom expert reviewer for C, networking, and MDNS code

## Setup

### 1. Install Dependencies

```bash
pip install pandas langchain langchain-community sentence-transformers faiss-cpu openai python-dotenv tqdm
```

### 2. Configure Environment Variables

Copy the example environment file and update it with your values:

```bash
cp .env.example .env
```

Edit the `.env` file with your specific configuration:

- `API_KEY`: Your OpenAI API key
- `BASE_URL`: API endpoint (default: https://api.openai.com/v1)
- `MODEL`: The model to use (e.g., gpt-4-0125-preview)
- `SYSTEM_PROMPT`: Default system prompt for the AI
- `ORIGINAL_CODE_PATH`: Path to your original code directory
- `REFACTORED_CODE_PATH`: Path to your refactored code directory

### 3. Running the Tool

```bash
python embed_func.py
```

## How It Works

1. The tool extracts functions from your codebase and builds a vector database
2. It performs semantic searches to find similarities between original and refactored code
3. The AI reviewer analyzes the original function and potential refactored matches
4. The reviewer provides a determination of how functions were refactored with confidence levels

## Classes

- `FunctionEmbedder`: Handles the semantic searching functionality
- `Agent`: Base class for AI-powered analysis
- `Reviewer`: Specialized agent for reviewing C/networking/MDNS code

## Customization

You can customize the behavior by:
- Modifying the system prompt in the `Reviewer` class
- Changing the embedding model in `FunctionEmbedder`
- Adjusting search parameters for more or fewer results
