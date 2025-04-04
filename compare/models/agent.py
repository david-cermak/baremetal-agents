import os
import time
import random
from openai import OpenAI

class Agent:
    """Base Agent class for interacting with OpenAI API."""

    def __init__(self, system_prompt=None, model=None):
        """
        Initialize an Agent.

        Args:
            system_prompt (str): The system prompt to use for the agent
            model (str): The OpenAI model to use
        """
        self.client = OpenAI(
            api_key=os.environ.get("API_KEY", ""),
            base_url=os.environ.get("BASE_URL", "https://api.openai.com/v1")
        )
        self.model = model or os.environ.get("MODEL", "gpt-4-0125-preview")
        self.system_prompt = system_prompt or os.environ.get("SYSTEM_PROMPT", "You are a helpful assistant specializing in code analysis.")

    def generate_response(self, user_prompt):
        """
        Generate a response from the OpenAI API.

        Args:
            user_prompt (str): The user prompt to send to the API

        Returns:
            str: The response from the API
        """
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
