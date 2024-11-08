import os
import dotenv
dotenv.load_dotenv("./my-env/.env")

import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from transformers import BitsAndBytesConfig

import torch

model_ids = {
    "Llama3.1:8B": "meta-llama/Llama-3.1-8B-Instruct",          # 8.03 B
    "Llama3.1:70B": "meta-llama/Llama-3.1-70B-Instruct",        # 70.6 B
    "Llama3.1:405B": "meta-llama/Llama-3.1-405B-Instruct",      # 406 B
    "Llama3.2:1B": "meta-llama/Llama-3.2-1B-Instruct",          # 1.24 B
    "Llama3.2:3B": "meta-llama/Llama-3.2-3B-Instruct",          # 3.21 B
    "Mixtral-8x7B": "mistralai/Mixtral-8x7B-Instruct-v0.1",     # 46.7 B
    "Mixtral-8x22B": "mistralai/Mixtral-8x22B-Instruct-v0.1",   # 141 B
    "Mistral-7B": "mistralai/Mistral-7B-Instruct-v0.3",         # 7.25 B
    "Mistral-Small": "mistralai/Mistral-Small-Instruct-2409",   # 22.2 B
    "Mistral-Large": "mistralai/Mistral-Large-Instruct-2407",   # 123 B
    "Mistral-Nemo": "mistralai/Mistral-Nemo-Instruct-2407",     # 12.2 B
    "Microsoft-Phi-3-mini": "microsoft/Phi-3-mini-128k-instruct", # 3.82 B
    "Microsoft-Phi-3-small": "microsoft/Phi-3-small-128k-instruct", # 7.39 B
    "Microsoft-Phi-3-medium": "microsoft/Phi-3-medium-128k-instruct", # 14 B
    "Microsoft-Phi-3.5-mini": "microsoft/Phi-3-mini-128k-instruct", #  3.82 B
    "Microsoft-Phi-3.5-MoE": "microsoft/Phi-3.5-MoE-instruct", # 41.9 B
}

class LargeLanguageModel:
    """ A large language model from the Hugging Face Transformers library. """
    def __init__(self, model_id: str):
        self.server = True
        
        # Cache dir for Hugging Face Models
        self.cache_dir = "./hf_cache"
        
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,  # Enable 4-bit quantization
            bnb_4bit_compute_dtype='float16',  # Optional: control compute precision, can be float16 or bfloat16
            bnb_4bit_use_double_quant=True,  # Optional: double quantization
            bnb_4bit_quant_type="nf4",  # Type of quantization, e.g., "nf4" or "fp4"
            llm_int8_threshold=6.0  # Optional: for mixed 8-bit/4-bit quantization
        )

        """ Initialize the language model. """
        self.model_id = model_id
        
        if self.server:
            self.cache_dir = "./hf_cache"
            self.tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=self.cache_dir,trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(model_id,quantization_config=quant_config, token=os.environ["HF_TOKEN"], cache_dir=self.cache_dir, trust_remote_code=True)
        else:
            self.tokenizer =  AutoTokenizer.from_pretrained(model_id, cache_dir=self.cache_dir, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(model_id,quantization_config=quant_config, trust_remote_code=True)

        self.model.eval()

        self.pipeline = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer, max_new_tokens=200)

    def predict(self, x: str) -> str:
        """ Predict the masked token in the input text. """

        with torch.no_grad():
            response =  self.pipeline(x)

        return response
        


if __name__ == "__main__":
    model = LargeLanguageModel(model_ids["Llama3.1:8B"])

    query = "What is the capital of France?"
    query = "Which football player is bigger, Messi or Ronaldo?"

    # read react_prompt_3.txt
    with open("data/input/react_prompt4.txt", "r") as f:
        prompt = f.read()

    # Add input to {input}
    prompt = prompt.replace("{input}", query)
    
    # print(model.model)
    # print(model.tokenizer)
    response = model.predict(prompt)
    response = response[0]["generated_text"]

    # Make response without the prompt
    response = response.replace(prompt, "")

    print("Query: ", query)
    print("Response: ", response)
