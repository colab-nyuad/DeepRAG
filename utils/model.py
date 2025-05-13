from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union
import json
import os
import ollama
import traceback
from anthropic import Anthropic

class BaseModelProvider(ABC):
    """
    Abstract base class for model providers.
    Defines the interface for different model implementations.
    """
    @abstractmethod
    def __init__(self, model_name: str):
        """
        Initialize the model provider with a specific model.
        
        Args:
            model_name (str): Name of the model to use.
        """
        self.model_name = model_name
    
    @abstractmethod
    def call_model(self,
                 system_prompt: str,
                 user_prompt: str, 
                 stream: bool = False, 
                 **kwargs: Any) -> Union[Dict, Any]:
        """
        Generate a response using the model.
        
        Args:
            prompt (str): Input prompt for the model.
            stream (bool, optional): Whether to stream the response.
            **kwargs: Additional generation parameters.
        
        Returns:
            Model's response
        """
        pass


class OllamaModelProvider(BaseModelProvider):
    """
    Ollama-specific model provider implementation.
    """
    def __init__(self, model_name: str = 'falcon:3b', temperature: float = 0.5):
        try:
            self.client = ollama.Client(
                host="http://10.229.191.26:11434",
            )
        except ImportError:
            raise ImportError("Ollama library is not installed. Please install it using 'pip install ollama'")
        
        super().__init__(model_name)
        self.temperature = temperature
    
    def call_model(self, 
                system_prompt: str,
                user_prompt: str, 
                stream: bool = False, 
                **kwargs: Any) -> Union[Dict, Any]:
        """
        Generate a response using the Ollama model.
        
        Args:
            prompt (str): Input prompt for the model.
            stream (bool, optional): Whether to stream the response.
            **kwargs: Additional generation parameters.
        
        Returns:
            Model's response or generator
        """
        try:
            response = self.client.generate(
                model=self.model_name, 
                prompt=json.dumps([
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ]),
                format='json',
                options={'temperature': self.temperature},
                stream=stream,
                **kwargs
            )
            return response['response']
        except Exception as e:
            print(f"Error generating response with Ollama: {e}")
            return None


class ClaudeModelProvider(BaseModelProvider):
    """
    Anthropic Claude-specific model provider implementation using the official Anthropic Python client.
    """
    def __init__(self, model_name: str = 'claude-3-5-sonnet-20240620', temperature: float = 0.5):
        """
        Initialize the Claude model provider.
        
        Args:
            model_name (str): Name of the Claude model to use. Default is 'claude-3-sonnet-20240229'.
            temperature (float): Sampling temperature. Default is 0.5.
        """
        super().__init__(model_name)
        self.temperature = temperature
        self.api_key = os.environ.get('ANTHROPIC_API_KEY')
        
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set")
        
        try:
            self.client = Anthropic(api_key=self.api_key)
        except ImportError:
            raise ImportError("Anthropic Python client is not installed. Please install it using 'pip install anthropic'")
    
    def call_model(self,
                  system_prompt: str,
                  user_prompt: str,
                  stream: bool = False,
                  max_tokens: int = 200,
                  **kwargs: Any) -> str:
        """
        Generate a response using the Claude model via the Anthropic client library.
        
        Args:
            system_prompt (str): System prompt to guide the model.
            user_prompt (str): User input prompt for the model.
            stream (bool, optional): Whether to stream the response.
            max_tokens (int, optional): Maximum tokens to generate. Defaults to 1024.
            **kwargs: Additional generation parameters.
        
        Returns:
            Claude's response as a string
        """
        try:
            # Prepare parameters for the API call
            params = {
                "model": self.model_name,
                "system": system_prompt,
                "messages": [
                    {"role": "user", "content": user_prompt}
                ],
                "max_tokens": max_tokens,
                "temperature": self.temperature,
                "stream": stream
            }
            
            # Add any additional kwargs to the params
            for key, value in kwargs.items():
                if key not in params and key not in ['stream']:  # Exclude 'stream' as we handle it separately
                    params[key] = value
            
            # Handle streaming separately if implemented
            if stream:
                print("Warning: Streaming not fully implemented for Claude API, falling back to non-streaming")
            
            # Make the API call
            message = self.client.messages.create(**params)
            
            # Extract the text content from the response
            if hasattr(message, 'content') and message.content:
                # Find the first text content block
                for content_block in message.content:
                    if content_block.type == 'text':
                        return content_block.text
                
                # If we didn't find any text content blocks
                print("Warning: No text content found in Claude response")
                return str(message.content)
            else:
                print(f"Unexpected response format: {message}")
                return str(message)
        
        except Exception as e:
            print(f"Error generating response with Claude API: {e}")
            traceback.print_exc()
            return f"Error: {str(e)}"


class ModelManager:
    """
    Centralized model management class to handle different model providers.
    """
    def __init__(self, 
                 provider: str = 'ollama', 
                 model_name: Optional[str] = None, 
                 temperature: float = 0.5):
        """
        Initialize the ModelManager with a specific provider.
        
        Args:
            provider (str, optional): Model provider name. Defaults to 'ollama'.
            model_name (str, optional): Specific model to use. 
                                        If None, uses provider's default.
            temperature (float, optional): Sampling temperature. Defaults to 0.5.
        """
        providers = {
            'ollama': OllamaModelProvider,
            'claude': ClaudeModelProvider
            # Future providers can be added here
            # 'openai': OpenAIModelProvider,
            # 'groq': GroqModelProvider
        }
        
        if provider not in providers:
            raise ValueError(f"Unsupported provider: {provider}. "
                             f"Supported providers: {list(providers.keys())}")
        
        # Use model_name if provided, otherwise use provider's default
        if model_name is None:
            if provider == 'ollama':
                model_name = 'falcon3:10b'
            elif provider == 'claude':
                model_name = 'claude-3-5-sonnet-20240620'  # Using full model name including version
        
        self.provider = providers[provider](model_name, temperature)
    
    def call_model(self, *args, **kwargs):
        """
        Proxy method to generate response using the current provider.
        """
        return self.provider.call_model(*args, **kwargs)