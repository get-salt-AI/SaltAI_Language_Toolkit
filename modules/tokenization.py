class MockTokenizer:
    def __init__(self, max_tokens, char_per_token=4):
        """
        Initializes the tokenizer with a maximum token count and an approximate character per token ratio.
        
        Args:
        max_tokens (int): Maximum number of tokens.
        char_per_token (int): Approximate number of characters per token, default is 4.
        """
        self.max_tokens = max_tokens
        self.char_per_token = char_per_token
        self.max_chars = max_tokens * char_per_token

    def encode(self, data):
        """
        Encodes data by splitting it into tokens of fixed character length.
        
        Args:
        data (str): Input text to be encoded.
        
        Returns:
        list: List of fixed-length tokens.
        """
        return [data[i:i + self.char_per_token] for i in range(0, len(data), self.char_per_token)]
    
    def decode(self, data):
        """
        Decodes data by joining fixed-length tokens into a single string.
        
        Args:
        data (list): List of fixed-length tokens to be decoded.
        
        Returns:
        str: Decoded string.
        """
        return ''.join(data)

    def truncate(self, data):
        """
        Truncates input data to fit within the maximum character limit.
        
        Args:
        data (str or dict or list): Input text, dictionary, or list to be truncated.
        
        Returns:
        str or dict or list: Truncated input.
        """
        if isinstance(data, str):
            return self._truncate_text(data)
        elif isinstance(data, dict):
            return self._truncate_dict(data)
        elif isinstance(data, list):
            return self._truncate_list(data)
        else:
            raise ValueError("Input data must be either a string, dictionary, or list")

    def count(self, data, truncate=False):
        """
        Estimates the token count of the given input data.
        
        Args:
        data (str, dict, or list): Input text, dictionary, or list.
        truncate (bool): Whether to estimate token count after truncation.
        
        Returns:
        int: Estimated token count.
        """
        if truncate:
            data = self.truncate(data)

        if isinstance(data, str):
            return (len(data) + self.char_per_token - 1) // self.char_per_token
        elif isinstance(data, dict):
            total_length = self._count_dict_chars(data)
            return (total_length + self.char_per_token - 1) // self.char_per_token
        elif isinstance(data, list):
            total_length = sum(self._count_list_chars(item) for item in data)
            return (total_length + self.char_per_token - 1) // self.char_per_token
        else:
            raise ValueError("Input data must be either a string, dictionary, or list")

    def _count_dict_chars(self, data):
        """
        Counts the total characters in the dictionary, including keys and values.
        
        Args:
        data (dict): Dictionary to be counted.
        
        Returns:
        int: Total number of characters.
        """
        total_length = 0
        for key, value in data.items():
            total_length += len(key)
            if isinstance(value, str):
                total_length += len(value)
            elif isinstance(value, dict):
                total_length += self._count_dict_chars(value)
            elif isinstance(value, list):
                total_length += sum(self._count_list_chars(item) for item in value)
        return total_length

    def _count_list_chars(self, item):
        """
        Counts the total characters in a list item, which can be a string, dict, or list.
        
        Args:
        item: List item to be counted.
        
        Returns:
        int: Total number of characters.
        """
        if isinstance(item, str):
            return len(item)
        elif isinstance(item, dict):
            return self._count_dict_chars(item)
        elif isinstance(item, list):
            return sum(self._count_list_chars(sub_item) for sub_item in item)
        return 0

    def _truncate_text(self, text):
        """
        Truncates text to fit within the maximum character limit.
        
        Args:
        text (str): Text to be truncated.
        
        Returns:
        str: Truncated text.
        """
        return text[:self.max_chars]

    def _truncate_dict(self, data):
        """
        Truncates dictionary keys and values to fit within the maximum character limit.
        
        Args:
        data (dict): Dictionary whose keys and values are to be truncated.
        
        Returns:
        dict: Dictionary with truncated keys and values.
        """
        truncated_data = {}
        current_chars = 0

        for key, value in data.items():
            if current_chars >= self.max_chars:
                break

            truncated_key = key[:self.max_chars - current_chars]
            truncated_data[truncated_key] = value
            current_chars += len(truncated_key)

            if isinstance(value, str):
                remaining_chars = self.max_chars - current_chars
                truncated_value = value[:remaining_chars]
                truncated_data[truncated_key] = truncated_value
                current_chars += len(truncated_value)
            elif isinstance(value, dict):
                remaining_chars = self.max_chars - current_chars
                truncated_value = self._truncate_dict(value)
                truncated_data[truncated_key] = truncated_value
                current_chars += self._count_dict_chars(truncated_value)
            elif isinstance(value, list):
                remaining_chars = self.max_chars - current_chars
                truncated_value = self._truncate_list(value, remaining_chars)
                truncated_data[truncated_key] = truncated_value
                current_chars += sum(self._count_list_chars(item) for item in truncated_value)

        return truncated_data

    def _truncate_list(self, data, max_chars=None):
        """
        Truncates list to fit within the maximum character limit.
        
        Args:
        data (list): List to be truncated.
        max_chars (int): Maximum number of characters for the truncated list.
        
        Returns:
        list: Truncated list.
        """
        if max_chars is None:
            max_chars = self.max_chars

        current_chars = 0
        truncated_list = []

        for item in data:
            item_length = self._count_list_chars(item)
            if current_chars + item_length > max_chars:
                break
            truncated_list.append(item)
            current_chars += item_length

        return truncated_list