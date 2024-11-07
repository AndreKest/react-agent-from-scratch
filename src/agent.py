from enum import Enum, auto
from typing import Union, Callable, List, Dict
import json
import logging
import regex as re
from pydantic import BaseModel, Field

from llm.llm import LargeLanguageModel
from utils.logger import setup_logger
from tools.search import ddgs_search, wiki_search
from tools.calc import add, multiply
from utils.io import read_file, write_to_file

import torch

import pudb

# Debugging
# pudb.set_trace()

# Set GPU device
torch.cuda.set_device(2)


model_ids = {
    "Llama3.1:8B": "meta-llama/Llama-3.1-8B-Instruct",          # 8.03 B
    "Llama3.1:70B": "meta-llama/Llama-3.1-70B-Instruct",        # 70.6 B
    "Llama3.2:1B": "meta-llama/Llama-3.2-1B-Instruct",          # 1.24 B
    "Llama3.2:3B": "meta-llama/Llama-3.2-3B-Instruct",          # 3.21 B
    "Mixtral-8x7B": "mistralai/Mixtral-8x7B-Instruct-v0.1",     # 46.7 B
    "Mixtral-8x22B": "mistralai/Mixtral-8x22B-Instruct-v0.1",   # 141 B
    "Mistral-7B": "mistralai/Mistral-7B-Instruct-v0.3",         # 7.25 B
    "Mistral-Small": "Mistral-Small-Instruct-2409",             # 22.2 B
    "Mistral-Large": "Mistral-Large-Instruct-2407",             # 123 B
    "Mistral-Nemo": "mistralai/Mistral-Nemo-Instruct-2407",     # 12.2 B

}

model_id = "Mistral-7B"
# model_id = "Mixtral-8x7B"

# PROMPT_TEMPLATE_PATH = "./data/input/prompt_llama.txt"
PROMPT_TEMPLATE_PATH = "./data/input/prompt_mistral.txt"
OUTPUT_TRACE_PATH = f"./data/output/trace_{model_id}.txt"
logger = setup_logger(log_filename=f"app_{model_id}.log")


Observation = Union[str, Exception]

class Name(Enum):
    """ Enumeration for available tools. """
    WIKIPEDIA = auto()
    DUCKDUCKGO = auto()
    ADD = auto()
    MULTIPLY = auto()
    NONE = auto()

    def __str__(self) -> str:
        return self.name.lower()

class Message(BaseModel):
    """ Represents a message with a sender role and content. """
    role: str = Field(..., title="The role of the message sender.")
    content: str = Field(..., title="The content of the message.")

class Choice(BaseModel):
    """ Represents a choice of tool and the reason for choosing it. """
    name: Name = Field(..., title="The name of the tool.")
    message: Message = Field(..., title="The reason for choosing the tool.")


class Tool:
    """ Tool class for executing tool with name and callable function. """
    def __init__(self, name: Name, func: Callable[[str], str]):
        self.name = name
        self.func = func

    def use(self, query: str) -> Observation:
        """ Execute the tool function with the given query. """
        try:
            return self.func(query)
        except Exception as e:
            logger.error(f"Error executing {self.name}: {e}")
            return str(e)


class ReActAgent:
    """ ReAct agent class for managing tools and messages. """
    def __init__(self, model, logger):
        self.model = model

        self.tools: Dict[Name, Tool] = {}
        self.messages: List[Message] = []

        self.max_iterations = 20
        self.current_iteration = 0

        self.logger = logger

        self.query = ""
        self.prompt = self.load_prompt()

    
    def load_prompt(self) -> str:
        """ Load the prompt from the specified file. """
        return read_file(PROMPT_TEMPLATE_PATH, self.logger)
        
    def register(self, name: Name, func: Callable[[str], str]) -> None:
        """
        Registers a tool to the agent.

        Args:
            name (Name): The name of the tool.
            func (Callable[[str], str]): The function associated with the tool.
            description (str): The description of the tool.
        """
        self.tools[name] = Tool(name, func)

    def trace(self, role: str, content: str) -> None:
        """ Logs the message with the specified role and content and writes to file. """
        if role != "system":
            self.messages.append(Message(role=role, content=content))
        write_to_file(path=OUTPUT_TRACE_PATH, content=f"{role}: {content}\n", logger=self.logger)

    def get_history(self) -> str:
        """ Return the history of messages as a string. """
        return "\n".join([f"{msg.role}: {msg.content}" for msg in self.messages])


    def think(self) -> None:
        """
        Run the agent with the given query. 
        - Process the current query
        - Decide the action based on the response
        - Execute the action
        - Write the trace/observation
        """
        self.prompt_r = self.prompt.replace("{history}", self.get_history())


        # Ask the model with the prompt
        response = self.ask_model(self.prompt_r)


        response = response.replace(self.prompt_r, "")

        if "[Final Answer]" in response or "Final Answer" in response:
            print("Final Answer")
            # str_finalAnswer = re.search(r"\[Final Answer\](.*)", response).group(1)
            str_finalAnswer = re.search(r"Final Answer(.*)", response).group(1)

            self.logger.info(f"Final Answer => {str_finalAnswer}")
            self.trace("assistant", f"Final Answer: {str_finalAnswer}")
            self.messages.append(Message(role="assistant", content=f"Final Answer: {str_finalAnswer}"))
            
            return "", True

        else:
            # [THOUGHT] and \n
            if "[Thought]" in response or "Thought" in response:
                # str_thought = re.search(r"\[Thought\](.*)\n", response).group(1)
                str_thought = re.search(r"Thought(.*)", response).group(1)
            else:
                # Error No [Thought] found try again TODO
                self.logger.error("No [Thought] found in response. Try again and take care of the structure.")
                self.trace("assistant", "Wrong response structure (add [Thought]). Try again and take care of the structure.")
                return None, False        

        self.logger.info(f"Thinking => {str_thought}")
        self.trace("assistant", f"Thought: {str_thought}")

        # Decide the action based on the response
        return response, False
        

    def acting(self, response: str) -> None:
        """
        Processes the agent's response, deciding actions or final answers.

        Args:
            response (str): The response generated by the model.
        """
        try:
            if "[Action]" in response or "Action" in response:
                # str_action = re.search(r"\[Action\](.*)", response).group(1)
                str_action = re.search(r"Action(.*)", response).group(1)
            else:
                raise Exception("No [Action] found in response. Try again and add [Action].")
            
            # ====================================================================================
            # Clean the action string for function calling
            # Remove [ if exists at the beginning
            if str_action.startswith("["):
                str_action = str_action[1:]
            # Remove ] if exists at the end
            if str_action.endswith("]"):
                str_action = str_action[:-1]
            # Remove leading : if exists
            if str_action.startswith(":"):
                str_action = str_action[1:]
            
            # Remove leading and trailing whitespaces
            str_action = str_action.lstrip().rstrip()
            # ====================================================================================


            self.logger.info(f"Action => {str_action}")
            self.trace("assistant", f"Action: {str_action}")

            str_action_with_logger = str_action[:-1] + ', logger=self.logger)'

            return str_action_with_logger

        except Exception as e:
            logger.error(f"Error processing reasoning response: {str(e)}")
            self.trace("assistant", "I encountered an unexpected error. Let me try a different approach.")
            return None

    def observe(self, acting: str) -> Observation:
        """
        Observes the agent's action and updates the message history.

        Args:
            acting (str): The action taken by the agent.

        Returns:
            Observation: The observation or result of the action.
        """
        # Execute the action
        try:
            observation = eval(acting)

            # TODO: Error handling
            if observation is None:
                observation = f"Error - No site with Wikipedia API and the search query is found. Try another search tool or query."
                self.trace("tool", f"Error: {observation}")

            self.logger.info(f"Observation => {observation}")
            self.trace("user", f"Observation: {observation}")

            return 

        except Exception as e:
            logger.error(f"Error executing action: {e}")
            self.trace("assistant", f"I encountered an error that the action could not be executed. Try it again and beware of the response structure: {e}")
            
            return 


    def execute(self, query: str) -> str:
        """
        Executes the agent's query-processing workflow.

        Args:
            query (str): The query to be processed.

        Returns:
            str: The final answer or last recorded message content.
        """
        # Set the input query
        self.query = query
        write_to_file(path=OUTPUT_TRACE_PATH, content=f"\n{'='*50}\nStart\n{'='*50}\n", logger=self.logger)
        self.trace(role="user", content=query)
        
        # Reasoning and Acting Loop (ReAct Loop)
        while self.current_iteration < self.max_iterations:
            self.current_iteration += 1
            self.logger.info(f"Starting iteration {self.current_iteration}")
            write_to_file(path=OUTPUT_TRACE_PATH, content=f"\n{'='*50}\nIteration {self.current_iteration}\n{'='*50}\n", logger=self.logger)
            
            # Reasoning
            reasoning, f_answer = self.think()

            if reasoning is None:
                continue

            # Final Answer was found
            if f_answer:
                return self.messages[-1].content

            # Acting
            acting = self.acting(reasoning)

            # Observation
            observation = self.observe(acting)
        

        # Maximum number of iterations reached
        self.logger.info("Maximum number of iterations reached. Stopping the process.")
        self.trace("assistant", "I'm sorry, but I couldn't find a satisfactory answer within the allowed number of iterations. Here's what I know so far: " + self.get_history())
        return
        
    def ask_model(self, prompt: str) -> str:
        """
        Ask the model with the given prompt.
        
        Args:
            prompt (str): The prompt to ask the model.

        Returns:
            str: The response from the model.
        """
        response = self.model.predict(prompt)
        response = response[0]['generated_text']
        return str(response) if response is not None else "No response from Model"
       


def run(queries: [str], logger: logging.LogRecord) -> str:
    # Load model
    model = LargeLanguageModel(model_ids[model_id])

    for query in queries:
        # Create agent
        agent = ReActAgent(model=model, logger=logger)

        # Register tools
        agent.register(Name.WIKIPEDIA, wiki_search)
        agent.register(Name.ADD, add)
        agent.register(Name.MULTIPLY, multiply)
        # agent.register(Name.NONE, None, "No action needed.")

        # Execute query
        answer = agent.execute(query)

def create_queries():
    queries = []
    queries.append("What is the capital of France?")
    queries.append("Calculate 10 + 10")
    queries.append("Calculate 20 - 10")
    queries.append("200 * 10")
    queries.append("100 + -10")
    queries.append("When was the release of the movie Iron Man?")
    queries.append("Who is the president of Germany?")
    queries.append("Who is the president of the United States?")
    queries.append("When was Python first released? Add 2000 to the release year.")
    queries.append("In which year was the fall of the Berlin Wall? Then add 10 to the year.")
    queries.append("In what year was Jonas Vingegaard born? Add the result of 50 + 50 to the year.")

    return queries


if __name__ == "__main__":
    queries = create_queries()
    run(queries=queries, logger=logger)