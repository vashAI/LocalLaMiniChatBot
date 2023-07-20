from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from langchain.llms import HuggingFacePipeline
import torch

checkpoint = "./model/"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
base_model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint,
                                                    device_map='auto',
                                                    torch_dtype=torch.float32)
llm = HuggingFacePipeline.from_model_id(model_id=checkpoint,
                                        task = 'text2text-generation',
                                        model_kwargs={"temperature":0.60,"min_length":30, "max_length":600, "repetition_penalty": 5.0})
                                        
from langchain import PromptTemplate, LLMChain
template = """{text}"""
prompt = PromptTemplate(template=template, input_variables=["text"])
chat = LLMChain(prompt=prompt, llm=llm)

yourprompt = input("Enter your prompt: ")

reply = chat.run(yourprompt)
print(reply)
