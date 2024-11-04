from typing import Optional
import wikipediaapi
import json

import logging

from duckduckgo_search import DDGS

def wiki_search(query: str, logger: logging.LogRecord) -> Optional[str]:
    """ 
    Search Wikipedia for a given query and return as JSON.

    Args:
        query (str): The search query.

    Returns:
        Optional[str]:  A JSON string containing the query, title, and summary, or None if no result is found.

    """
    # Initialize the Wikipedia API
    wiki = wikipediaapi.Wikipedia(user_agent='react-agent-from-scratch (kestler.andre.code@gmail.com)', language='en')

    try:
        logger.info(f"Searching Wikipedia for: {query}")
        page = wiki.page(query)

        if page.exists():
            # Create a dictionary with query title and summary
            result = {
                "query": query,
                "title": page.title,
                "summary": page.summary
            }

            logger.info(f"Successfully retrieved summary for: {query}")
            return json.dumps(result, ensure_ascii=False, indent=2)
        else:
            logger.info(f"No result found for: {query}")
            return None

    except Exception as e:
        logger.exception(f"An error occurred while processing the Wikipedia search for: {query}")
        return None
    


def ddgs_search(query: str) -> Optional[str]:
    """
    Search DuckDuckGo for a given query and return the first result.

    Args:
        query (str): The search query.

    Returns:
        Optional[str]:  Return a JSON object containing the query, title, and abstract, or None if no result is found.
    """
    try:
        responses = DDGS().text(query, max_results=5)
        
        results = []
        for response in responses:
            print(response.keys())
            results.append({
                "query": query,
                "title": response['title'],
                "href": response['href'],
                "body": response['body']
            })
        
        return json.dumps(results, ensure_ascii=False, indent=2)

    except Exception as e:
        print(f"Request to DuckDuckGo failed: {e}")
        return None