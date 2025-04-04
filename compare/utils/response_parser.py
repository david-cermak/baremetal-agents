import re

class ResponseParser:
    """Utility class for parsing structured responses from AI models."""

    @staticmethod
    def extract_xml_content(response, tag_name):
        """
        Extract content from an XML-like tag in the response.

        Args:
            response (str): The response text
            tag_name (str): The name of the tag to extract

        Returns:
            str or None: The content of the tag if found, None otherwise
        """
        pattern = f'<{tag_name}>(.*?)</{tag_name}>'
        match = re.search(pattern, response, re.DOTALL)
        return match.group(1).strip() if match else None

    @staticmethod
    def parse_reviewer_response(response):
        """
        Parse a reviewer response to extract structured information.

        Args:
            response (str): The response text from the reviewer

        Returns:
            dict: A dictionary containing the parsed components
        """
        # Initialize result with default values
        result = {
            'refactored_function': None,
            'concern': None,
            'summary': None,
            'follow_up': None,
            'search_original': None,
            'search_refactored': None
        }

        # Extract all components
        for key in result.keys():
            result[key] = ResponseParser.extract_xml_content(response, key)

        # Process refactored function names if present
        if result['refactored_function']:
            result['refactored_function_names'] = [
                name.strip() for name in result['refactored_function'].split('\n')
                if name.strip()
            ]
        else:
            result['refactored_function_names'] = []

        return result

    @staticmethod
    def has_confident_answer(response):
        """
        Check if the response contains a confident answer (refactored function name).

        Args:
            response (str): The response text

        Returns:
            bool: True if a confident answer was found, False otherwise
        """
        match = ResponseParser.extract_xml_content(response, 'refactored_function')
        return match is not None and match.strip() != ""
