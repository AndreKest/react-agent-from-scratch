import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
from transformers import BitsAndBytesConfig

model_ids = {
    "Llama3.1:8B": "meta-llama/Llama-3.1-8B-Instruct",
    "Llama3.2:1B": "meta-llama/Llama-3.2-1B-Instruct",
    "Llama3.2:3B": "meta-llama/Llama-3.2-3B-Instruct",
}

class LargeLanguageModel:
    """ A large language model from the Hugging Face Transformers library. """
    def __init__(self, model_id: str):
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,  # Enable 4-bit quantization
            bnb_4bit_compute_dtype='float16',  # Optional: control compute precision, can be float16 or bfloat16
            bnb_4bit_use_double_quant=True,  # Optional: double quantization
            bnb_4bit_quant_type="nf4",  # Type of quantization, e.g., "nf4" or "fp4"
            llm_int8_threshold=6.0  # Optional: for mixed 8-bit/4-bit quantization
        )

        """ Initialize the language model. """
        self.model_id = model_id
        
        # self.model = AutoModelForCausalLM.from_pretrained(model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quant_config,  # Optional, for quantization
            device_map="auto",  # Automatically assign devices (e.g., GPU))  # Optional, for further optimization
        )

        self.pipeline = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer, max_new_tokens=200)

        # llm = HuggingFacePipeline(model_id=model_id, pipeline=self.pipeline)
        # self.model = ChatHuggingFace(llm=llm)

    def predict(self, x: str) -> str:
        """ Predict the masked token in the input text. """

        print(x)

        response =  self.pipeline(x)
        # response = self.model.invoke(x)

        print(response)

        return response
        


if __name__ == "__main__":
    model = LargeLanguageModel(model_ids["Llama3.2:3B"])
    
    print(model.model)
    print(model.tokenizer)
    print(model.predict("What is Hugging Face?"))
