import os
import logging
from openai import AsyncOpenAI, APIConnectionError

class LocalAIClient:
    def __init__(self, base_url=None, model=None, api_key=None):
        self.base_url = base_url or os.getenv("LOCAL_AI_BASE_URL", "http://localhost:11434/v1")
        self.model = model or os.getenv("LOCAL_AI_MODEL", "llama3")
        self.api_key = api_key or os.getenv("LOCAL_AI_API_KEY", "ollama")
        
        logging.info(f"Initializing Async Local AI Client at {self.base_url} with model {self.model}")
        
        self.client = AsyncOpenAI(
            base_url=self.base_url,
            api_key=self.api_key
        )

    async def generate_response(self, system_prompt, message_history):
        """
        Generates a response from the local AI model using full chat history.
        """
        try:
            # Construct full messages list
            messages = [{"role": "system", "content": system_prompt}] + message_history
            
            logging.debug(f"Sending async request to Local AI")
            
            # AWAIT the async call
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.6,
            )
            
            content = response.choices[0].message.content
            # Global Sanitization: Remove <|python_tag|> and similar artifacts
            if content:
                import re
                # Strip internal LLM tags
                content = re.sub(r"<\|.*?\|>", "", content)
                # Strip hallucinated user tags like [USER: Logan] anywhere in message
                content = re.sub(r"\[User:.*?\]\s*", "", content, flags=re.IGNORECASE)
                # Strip generic [SYSTEM NOTICE] echoes
                content = re.sub(r"\[SYSTEM.*?\]\s*", "", content, flags=re.IGNORECASE)
                # Strip [CONTEXT: ...] tags
                content = re.sub(r"\[CONTEXT:.*?\]\s*", "", content, flags=re.IGNORECASE)
                
            return content
            
        except APIConnectionError as e:
            logging.error(f"Connection error to Local AI: {e}")
            return "Error: Could not connect to the local AI server. Is it running?"
        except Exception as e:
            logging.error(f"Error generating response: {e}")
            return f"Error: {str(e)}"

    async def generate_stream(self, system_prompt, message_history):
        """
        Yields chunks of text from the AI as they are generated.
        """
        try:
            messages = [{"role": "system", "content": system_prompt}] + message_history
            
            # AWAIT the async stream creation
            stream = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.6,
                stream=True
            )
            
            # Async iterator
            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            logging.error(f"Error in generate_stream: {e}")
            yield f"Error: {str(e)}"
