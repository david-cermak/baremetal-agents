import os
from models.agent import Agent
from utils.response_parser import ResponseParser

class Reviewer(Agent):
    """
    Specialized agent for code review and refactoring analysis.
    """

    def __init__(self, model=None):
        """
        Initialize a Reviewer.

        Args:
            model (str): The OpenAI model to use
        """
        # Default to environment variable if model not provided
        model = model or os.environ.get("MODEL", "gpt-4-0125-preview")

        # Default system prompt for code reviewer
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

    def build_initial_prompt(self, original_func_name, func_content, original_context, refactored_context, initial=True):
        """
        Build the initial prompt for the reviewer.

        Args:
            original_func_name (str): Name of the original function
            func_content (str): Content of the original function
            original_context (str): Context from the original codebase
            refactored_context (str): Context from the refactored codebase

        Returns:
            str: The constructed prompt
        """
        concern_prompt_initial = """
* If you have a concern, that a subtle bug might have been introduced during refactoring, think about what information you need to give a confident response. Format your answer as described in **OTHERWISE** section and do not use `<refactored_function>` tag.

* If you are 100% confident that the given function is the refactored one and no bugs were introduced, just state the name of the function you think is the refactored function within xml tags, for example:
```xml
<refactored_function>
mdns_init
</refactored_function>
(note that the mapping could be 1:many, so there might be more functions within the xml tag, separated by newlines; or 1:0 so an empty xml tag)"""
        concern_prompt_follow_up = """
* If you are confident that the given function is the refactored one and no bugs were introduced, just state the name of the function you think is the refactored function within xml tags, for example:
```xml
<refactored_function>
mdns_init
</refactored_function>
```
(note that the mapping could be 1:many, so there might be more functions within the xml tag, separated by newlines; or 1:0 so an empty xml tag)

* If you have a concern, that a subtle bug have been introduced during refactoring, say so in the `<concern>` tag, for example:
```xml
<refactored_function>
mdns_init
</refactored_function>
<concern>
the new mdns_init function does not check if the pcb is not NULL.
</concern>
```"""
        return f"""
Please review the following refactoring of the function {original_func_name}. Mostly code structure changes and renaming.
The below context shows the original function and the refactored code containing multiple functions that might replace the original function.
Your goal is to find the actual refactored function and point out subtle bugs or concerns, introduced by the refactoring.
Do not jump to conclusions from the short context, if unsure, or have a suspicion, please summarize what you learned and give follow up questions, so the next reviewer has more context information.

Format your response as follows:

{concern_prompt_initial if initial else concern_prompt_follow_up}

* **OTHERWISE**, summarize what you have learned, your concerns and give follow up questions and search queries, within xml tags, for example:
```xml
<summary>
I learned that the function mdns_init might have been split into multiple functions, mdns_init_internal and mdns_receiver_init.
Add more details here, to help the next reviewer understand why you think another search and reasoning is needed.
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
{original_context}

### Refactored code

{refactored_context}
"""

    def build_follow_up_prompt(self, original_prompt, summary, follow_up_questions,
                              original_search_results=None, refactored_search_results=None,
                              search_original_terms=None, search_refactored_terms=None):
        """
        Build a follow-up prompt with additional context.

        Args:
            original_prompt (str): The original prompt
            summary (str): Summary from previous analysis
            follow_up_questions (str): Follow-up questions from previous analysis
            original_search_results (str): Results of searching the original codebase
            refactored_search_results (str): Results of searching the refactored codebase
            search_original_terms (str): Terms used to search the original codebase
            search_refactored_terms (str): Terms used to search the refactored codebase

        Returns:
            str: The constructed prompt with additional context
        """
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
        if search_original_terms and original_search_results:
            additional_context += f"""
### Additional search results from original codebase for "{search_original_terms}"
{original_search_results}

"""

        # Search for refactored terms if provided
        if search_refactored_terms and refactored_search_results:
            additional_context += f"""
### Additional search results from refactored codebase for "{search_refactored_terms}"
{refactored_search_results}

"""

        # Add a reminder about the confidence requirement
        additional_context += """
Based on this additional context, please reconsider your analysis and provide a more confident answer if possible.
Remember, only use the <refactored_function> tag if you have 95% confidence in your determination.
"""
        concern_prompt = """
* If you have 95% confidence that the function the refactored one, but you have a concern, that a subtle bug might have been introduced, say the name of the function and the concern you have, for example:
```xml
<refactored_function>
mdns_init
</refactored_function>
<concern>
the new mdns_init function does not check if the pcb is not NULL.
</concern>
```
"""
        return original_prompt + additional_context + concern_prompt

    def parse_response(self, response):
        """
        Parse the reviewer's response.

        Args:
            response (str): The response from the reviewer

        Returns:
            dict: Parsed information from the response
        """
        return ResponseParser.parse_reviewer_response(response)

    def get_refactored_function_names(self, response):
        """
        Get the refactored function names from a response.

        Args:
            response (str): The reviewer's response

        Returns:
            list: List of refactored function names, or empty list if none found
        """
        result = self.parse_response(response)
        return result['refactored_function_names']

    def get_concern(self, response):
        """
        Get any concerns about the refactoring from the response.

        Args:
            response (str): The reviewer's response

        Returns:
            str or None: The concern if present, None otherwise
        """
        result = self.parse_response(response)
        return result['concern']
