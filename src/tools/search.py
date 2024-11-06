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
    


def ddgs_search(query: str, logger: logging.LogRecord) -> Optional[str]:
    """
    Search DuckDuckGo for a given query and return the first result.

    Args:
        query (str): The search query.

    Returns:
        Optional[str]:  Return a JSON object containing the query, title, and abstract, or None if no result is found.
    """
    try:
        logger.info(f"Searching DuckDuckGo for: {query}")
        responses = DDGS().text(query, max_results=1)
        
        results = []
        if responses:
            for response in responses:
                print(response.keys())
                results.append({
                    "query": query,
                    "title": response['title'],
                    "href": response['href'],
                    "body": response['body']
                })

            logger.info(f"Successfully retrieved summary for {len(results)} sites: {query}")
            return results[0]['body']
            # return json.dumps(results, ensure_ascii=False, indent=2)['body']
        else:
            logger.info(f"No result found for: {query}")
            return None

    except Exception as e:
        logger.exception(f"An error occurred while processing the DuckDuckGo search for: {query}")
        return None
