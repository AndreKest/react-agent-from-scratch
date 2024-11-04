from utils.logger import logger
from tools.search import ddgs_search, wiki_search

if __name__ == "__main__":
    queries = ["Geoffrey Hinton", "Hugging Face"]


    # for query in queries:
    #     result = wiki_search(query, logger=logger)
        
    #     if result:
    #         print(f"JSON result for {query}:\n{result}")
    #     else:
    #         print(f"No result found for: {query}\n")

    for query in queries:
        result = ddgs_search(query)
        
        if result:
            print(f"JSON result for {query}:\n{result}")
        else:
            print(f"No result found for: {query}\n")