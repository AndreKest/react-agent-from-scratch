import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

model_ids = {
    "Llama3.1:8B": "meta-llama/Llama-3.1-8B-Instruct",
    "Llama3.2:1B": "meta-llama/Llama-3.2-1B-Instruct",
    "Llama3.2:3B": "meta-llama/Llama-3.2-3B-Instruct",
}

class LargeLanguageModel:
    """ A large language model from the Hugging Face Transformers library. """
    def __init__(self, model_id: str):
        """ Initialize the language model. """
        self.model_id = model_id
        
        self.model = AutoModelForCausalLM.from_pretrained(model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

        self.pipeline = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer, device=0, max_new_tokens=512)  

    def predict(self, x: str) -> str:
        """ Predict the masked token in the input text. """
        response =  self.pipeline(x)

        return response
        


if __name__ == "__main__":
    model = LargeLanguageModel(model_ids["Llama3.2:1B"])
    
    print(model.model)
    print(model.tokenizer)
    print(model.predict("What is Hugging Face?"))
