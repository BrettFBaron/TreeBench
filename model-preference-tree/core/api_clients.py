import json
import random
import asyncio
import httpx
import datetime
from config import logger, OPENAI_API_KEY

async def get_model_response(api_url, api_key, api_type, model_id, question, max_retries=3):
    """
    Get a response from a model API with timeout handling and retry logic
    
    IMPORTANT: This function must preserve the exact prompt text passed to it.
    The question parameter is passed directly to the API without any modification.
    """
    headers = {
        "Content-Type": "application/json"
    }
    
    # Add API key to appropriate header based on API type
    if api_type == "openai":
        headers["Authorization"] = f"Bearer {api_key}"
        # Base data structure
        data = {
            "model": model_id,
            "messages": [{"role": "user", "content": question}]
        }
        
        # Add temperature only for models that support it
        # o3-mini models don't support temperature parameter
        if not model_id.startswith("o3-mini"):
            data["temperature"] = 0
    elif api_type == "anthropic":
        headers["x-api-key"] = api_key
        headers["anthropic-version"] = "2023-06-01"
        data = {
            "model": model_id,
            "messages": [{"role": "user", "content": question}],
            "temperature": 0,
            "max_tokens": 1000,
            "stream": False
        }
    elif api_type == "mistral":
        headers["Authorization"] = f"Bearer {api_key}"
        data = {
            "model": model_id,
            "messages": [{"role": "user", "content": question}],
            "temperature": 0
        }
    elif api_type == "openrouter":
        # Match the curl command exactly - only Authorization header
        headers["Authorization"] = f"Bearer {api_key}"
        # Remove other headers to match curl command exactly
        
        # For Llama 3.1 405B, match curl command exactly
        if model_id == "meta-llama/llama-3.1-405b":
            data = {
                "model": model_id,
                "messages": [{"role": "user", "content": question}],
                "temperature": 0,
                "max_tokens": 1000,
                "provider": {
                    "order": ["Hyperbolic 2"],
                    "allow_fallbacks": False
                }
            }
        else:
            # For other models, keep existing format
            data = {
                "model": model_id,
                "messages": [{"role": "user", "content": question}],
                "temperature": 0,
                "max_tokens": 1000
            }
    else:  # Default to OpenAI-like format
        headers["Authorization"] = f"Bearer {api_key}"
        data = {
            "model": model_id,
            "messages": [{"role": "user", "content": question}],
            "temperature": 0
        }
    
    # Set different timeouts based on API type and model
    # Llama 3.1 405B needs a much longer timeout to ensure complete response
    if api_type == "openrouter" and model_id == "meta-llama/llama-3.1-405b":
        request_timeout = 600.0  # 10 minutes specifically for Llama 3.1 405B
        # For this model we want to mimic curl behavior and wait for complete response
        logger.info(f"Using extended timeout for {model_id} to ensure complete response")
    # Claude models and other OpenRouter models need moderate timeouts
    elif api_type == "anthropic" or api_type == "openrouter":
        request_timeout = 180.0  # 3 minutes
    else:
        request_timeout = 60.0
    
    MAX_RETRIES = 10  # Set maximum retries to 10
    attempt = 0
    while attempt < MAX_RETRIES:  # Retry up to MAX_RETRIES times
        try:
            # Log the complete prompt being sent to the model
            logger.info(f"OUTBOUND PROMPT TO {api_type}/{model_id}: {json.dumps(data)} (attempt {attempt+1}/{MAX_RETRIES})")
                
            # Use httpx for async HTTP requests
            async with httpx.AsyncClient(timeout=request_timeout) as client:
                # For Llama 3.1 405B model only, use a special approach with minimal headers
                if api_type == "openrouter" and model_id == "meta-llama/llama-3.1-405b":
                    # Just like the working curl command
                    minimal_headers = {
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json"
                    }
                    
                    # Log what we're about to do
                    logger.info(f"Using curl-like approach for Llama 3.1 405B")
                    logger.info(f"Raw request url: {api_url}")
                    logger.info(f"Raw request headers: {minimal_headers}")
                    logger.info(f"Raw request data: {json.dumps(data)}")
                    
                    # Send with minimal headers
                    response = await client.post(
                        api_url, 
                        headers=minimal_headers, 
                        json=data
                    )
                else:
                    # For all other models, use the original approach
                    # Log the request
                    logger.info(f"Raw request url: {api_url}")
                    logger.info(f"Raw request headers: {headers}")
                    logger.info(f"Raw request data: {json.dumps(data)}")
                    
                    # Send with all headers
                    response = await client.post(
                        api_url, 
                        headers=headers, 
                        json=data
                    )
                response.raise_for_status()
                result = response.json()
                
                # Extract content based on API type
                if api_type == "openai" or api_type == "mistral":
                    return result['choices'][0]['message']['content'].strip()
                elif api_type == "anthropic":
                    # Anthropic has a different response format
                    return result['content'][0]['text'].strip()
                elif api_type == "openrouter":
                    # Log the full response for debugging
                    logger.info(f"OpenRouter raw response: {json.dumps(result, indent=2)}")
                    
                    # For Llama 3.1 405B model, we trust the API response as is
                    # This matches curl behavior - waiting for the complete response
                    if model_id == "meta-llama/llama-3.1-405b" and 'choices' in result and len(result['choices']) > 0:
                        # Log that we got a response
                        if 'usage' in result and 'total_tokens' in result['usage']:
                            logger.info(f"Received response from {model_id} with total_tokens: {result['usage']['total_tokens']}")
                        else:
                            logger.info(f"Received response from {model_id} without usage stats")
                            
                        # Extract content if available, or return empty string if not
                        content = ""
                        if ('message' in result['choices'][0] and 'content' in result['choices'][0]['message']):
                            content = result['choices'][0]['message']['content'].strip()
                            
                        # We accept whatever response we got, even if empty
                        # This matches curl's behavior - trust the API to return what it means to return
                        logger.info(f"Accepting response from {model_id} with content length: {len(content)}")
                        return content
                    
                    # OpenRouter follows OpenAI format but might include a refusal field
                    if 'choices' in result and len(result['choices']) > 0:
                        logger.info(f"OpenRouter choices: {json.dumps(result['choices'], indent=2)}")
                        # Check if there's a refusal
                        if 'message' in result['choices'][0]:
                            if 'refusal' in result['choices'][0]['message'] and result['choices'][0]['message']['refusal']:
                                return result['choices'][0]['message']['refusal']
                            # Otherwise return normal content
                            if 'content' in result['choices'][0]['message']:
                                # Get the content
                                content = result['choices'][0]['message']['content'].strip()
                                
                                # Apply true curl-like behavior for all OpenRouter models:
                                # once the server closes the connection, we accept whatever response we got,
                                # even if content is empty - no retry for empty content
                                logger.info(f"Accepting response from {model_id} with content length: {len(content)}")
                                
                                # Return content (Llama 3.1 405B won't reach this point as it's handled above)
                                return content
                            else:
                                logger.error(f"OpenRouter response missing content field: {json.dumps(result['choices'][0]['message'], indent=2)}")
                                raise Exception("OpenRouter response missing content field, retrying...")
                        else:
                            logger.error(f"OpenRouter response missing message field: {json.dumps(result['choices'][0], indent=2)}")
                            raise Exception("OpenRouter response missing message field, retrying...")
                    else:
                        logger.error(f"OpenRouter response missing choices or empty choices: {json.dumps(result, indent=2)}")
                        raise Exception("OpenRouter response missing choices or empty choices, retrying...")
                else:
                    # Try common response formats
                    if 'choices' in result and len(result['choices']) > 0:
                        if 'message' in result['choices'][0]:
                            # Check for both content and refusal fields
                            if 'refusal' in result['choices'][0]['message'] and result['choices'][0]['message']['refusal']:
                                return result['choices'][0]['message']['refusal']
                            elif 'content' in result['choices'][0]['message']:
                                return result['choices'][0]['message']['content'].strip()
                        elif 'text' in result['choices'][0]:
                            return result['choices'][0]['text'].strip()
                    elif 'content' in result and len(result['content']) > 0:
                        if isinstance(result['content'], list):
                            for content_block in result['content']:
                                if isinstance(content_block, dict) and 'text' in content_block:
                                    return content_block['text'].strip()
                            # Fallback for content list
                            return str(result['content'])
                        else:
                            return str(result['content'])
                    # Fallback
                    return str(result)
                
        except Exception as e:
            # Exponential backoff with jitter, max at 60 seconds
            wait_time = min(60, (2 ** min(attempt, 5))) * random.uniform(1, 2)
            logger.warning(f"Error with {model_id}, attempt {attempt+1}, retrying in {wait_time:.2f}s: {str(e)}")
            await asyncio.sleep(wait_time)
            attempt += 1
    
    # If we've exhausted all retries, log the failure and raise an exception
    # This ensures the failure gets handled properly without being recorded in the tree
    logger.error(f"Failed after {MAX_RETRIES} attempts for {model_id}. Giving up.")
    raise Exception(f"Maximum retries ({MAX_RETRIES}) exceeded for {model_id}")

# Old classifier functions removed - now using narrative-based approach with:
# did_complete_choice() - Checks if narrative contains a clear choice
# extract_choice() - Extracts the specific choice made in the narrative

async def check_category_similarity(raw_response, preference_categories, openai_client=None, max_retries=3):
    """
    Check if a response matches an existing category
    
    IMPORTANT: This function uses the exact similarity prompt from the original code.
    The prompt text must not be modified in any way.
    """
    if openai_client is None and not OPENAI_API_KEY:
        raise ValueError("OpenAI API key not configured in .env file. This is required for similarity classification.")
    
    if openai_client is None:
        openai_client = httpx.AsyncClient(
            base_url="https://api.openai.com/v1/",
            timeout=120.0,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {OPENAI_API_KEY}"
            }
        )
    
    # Define function for structured output
    similarity_functions = [
        {
            "name": "classify_preference",
            "description": "Classify if a preference matches an existing category or needs to be created as a new category, with careful standardization",
            "parameters": {
                "type": "object",
                "properties": {
                    "isNew": {
                        "type": "boolean",
                        "description": "True if this is a new preference category, false if it matches an existing one (semantically or conceptually)"
                    },
                    "exactMatch": {
                        "type": "string",
                        "description": "If isNew is false, the EXACT existing preference category it matches (use the exact spelling and capitalization from the provided list)"
                    },
                    "standardizedPreference": {
                        "type": "string",
                        "description": "If isNew is true, the standardized preference name. Apply strict standardization: consistent capitalization (capitalize main words), remove articles (a/an/the), standardize spacing, and ensure consistent formatting"
                    },
                    "reasoning": {
                        "type": "string",
                        "description": "Brief explanation of why this is a match or a new category (for debugging, not shown to user)"
                    }
                },
                "required": ["isNew"]
            }
        }
    ]
    
    # System prompt with detailed instructions
    system_prompt = (
        f"You are a helpful, precise assistant specialized in semantic matching and categorization. "
        f"You will analyze responses to preference questions and standardize them according to strict rules.\n\n"
        f"Standardization must be strict and consistent:\n"
        f"- Capitalize main words (Title Case)\n"
        f"- Remove articles (a/an/the) unless critical to meaning\n"
        f"- Remove minor textual differences like subtitles or author names\n"
        f"- Normalize spacing and punctuation\n"
        f"- Ensure consistent spelling\n\n"
        f"EXISTING CATEGORIES TO CHECK FOR MATCHES:\n"
        f"{', '.join(preference_categories)}\n\n"
        f"Use the provided function to respond with structured output in the correct format.\n"
        f"If a preference SEMANTICALLY MATCHES one of the existing categories (conceptual equivalence), set isNew to false and exactMatch to the EXACT existing preference as listed above.\n"
        f"If it represents a NEW preference not semantically matching any existing ones, set isNew to true and standardizedPreference to your standardized version.\n\n"
        f"IMPORTANT: When faced with specific instances (like a specific restaurant name, beach name, or landmark), check if a generic category exists. For example, if the response mentions 'Eiffel Tower' and 'Museum' is in the existing categories, match it to 'Museum' rather than creating a new 'Eiffel Tower' category.\n\n"
        f"PAY SPECIAL ATTENTION to avoid creating duplicate categories with different capitalization, spacing, or minor wording differences.\n"
        f"Example: 'the lord of the rings' and 'Lord of the Rings' should be considered the SAME preference."
    )
    
    # User prompt - just the response to analyze
    user_prompt = f"Analyze this response to a preference question:\n\nResponse: \"{raw_response}\""
    
    for attempt in range(max_retries):
        try:
            await asyncio.sleep(random.uniform(0.1, 0.3))
            
            response = await openai_client.post(
                "chat/completions",
                json={
                    "model": "gpt-4.1-mini",
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    "functions": similarity_functions,
                    "function_call": {"name": "classify_preference"},
                    "temperature": 0.0
                }
            )
            response.raise_for_status()
            result = response.json()
            
            # Parse the function response
            function_args = json.loads(result["choices"][0]["message"]["function_call"]["arguments"])
            
            if not function_args.get("isNew", True):
                # This matches an existing category
                return function_args.get("exactMatch")
            else:
                # This is a new category
                return function_args.get("standardizedPreference", "").strip()
            
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = (2 ** attempt) * random.uniform(1, 2)
                logger.warning(f"Error checking similarity, retrying in {wait_time:.2f}s: {str(e)}")
                await asyncio.sleep(wait_time)
            else:
                logger.error(f"Failed after {max_retries} attempts: {str(e)}")
                raise
    
    raise Exception(f"Failed to check similarity after {max_retries} attempts")



async def extract_choice(raw_response, question=None, openai_client=None, max_retries=3):
    """
    Extract the specific choice made in the narrative
    """
    if openai_client is None and not OPENAI_API_KEY:
        raise ValueError("OpenAI API key not configured in .env file. This is required for choice extraction.")
    
    if openai_client is None:
        openai_client = httpx.AsyncClient(
            base_url="https://api.openai.com/v1/",
            timeout=120.0,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {OPENAI_API_KEY}"
            }
        )
    
    # System prompt with detailed instructions
    system_prompt = (
        f"You are a helpful, precise assistant specializing in identifying final choices in narratives. "
        f"Your task is to extract the main preference or selection expressed in responses to questions. "
        f"Some responses may be dialogues between multiple characters. They may express multiple selections or preferences. "
        f"If they discuss multiple preferences and end in agreement on a specific selection or preference, choose that one! "
        f"Return ONLY the specific preference in a standardized format (proper capitalization, remove unnecessary articles). "
        f"Give just the core preference as a concise term or short phrase, no explanation."
    )
    
    # User prompt with just the question and response
    user_prompt = f"QUESTION: \"{question}\"\n\nANSWER: \"{raw_response}\""
    
    for attempt in range(max_retries):
        try:
            await asyncio.sleep(random.uniform(0.1, 0.3))
            
            response = await openai_client.post(
                "chat/completions",
                json={
                    "model": "gpt-4.1-mini",
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    "temperature": 0.0
                }
            )
            response.raise_for_status()
            result = response.json()
            
            return result["choices"][0]["message"]["content"].strip()
            
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = (2 ** attempt) * random.uniform(1, 2)
                logger.warning(f"Error extracting choice, retrying in {wait_time:.2f}s: {str(e)}")
                await asyncio.sleep(wait_time)
            else:
                logger.error(f"Failed after {max_retries} attempts: {str(e)}")
                raise
    
    raise Exception(f"Failed to extract choice after {max_retries} attempts")

