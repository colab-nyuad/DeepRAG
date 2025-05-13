import os
import json
import ollama
from typing import Dict, Any, Union

class FalconLLM:
    """
    A self-contained class for interacting with Falcon LLM through Ollama.
    """
    def __init__(self, 
                 model_name: str = 'falcon:3b',
                 temperature: float = 0.5,
                 host: str = "http://localhost:11434"):
        """
        Initialize the Falcon LLM client.
        
        Args:
            model_name (str): Name of the Falcon model to use ('falcon:3b' or 'falcon3:10b')
            temperature (float): Sampling temperature for generation (0.0 to 1.0)
            host (str): Ollama server host URL
        """
        try:
            self.client = ollama.Client(host=host)
            self.model_name = model_name
            self.temperature = temperature
        except ImportError:
            raise ImportError("Ollama library is not installed. Please install it using 'pip install ollama'")
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Ollama server at {host}: {str(e)}")

    def generate(self, 
                system_prompt: str,
                user_prompt: str,
                stream: bool = False,
                **kwargs: Any) -> Union[Dict, Any]:
        """
        Generate a response using the Falcon model.
        
        Args:
            system_prompt (str): System context/instruction for the model
            user_prompt (str): User input/question
            stream (bool): Whether to stream the response
            **kwargs: Additional generation parameters
            
        Returns:
            Model's response or generator if streaming
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
            print(f"Error generating response with Falcon: {e}")
            return None

def main():
    # Example usage
    try:
        # Initialize the Falcon LLM
        llm = FalconLLM(
            model_name='falcon:3b',  # or 'falcon3:10b' for the larger model
            temperature=0.7,
            host="http://localhost:11434"  # Change this to your Ollama server address
        )
        
        # Example prompts
        system_prompt = "You are a helpful AI assistant that provides clear and concise answers."
        user_prompt = "What are the main differences between Python and JavaScript?"
        
        # Generate response
        response = llm.generate(system_prompt, user_prompt)
        
        if response:
            print("\nSystem: ", system_prompt)
            print("User: ", user_prompt)
            print("\nFalcon's Response:\n", response)
        else:
            print("Failed to generate response")
            
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main() 