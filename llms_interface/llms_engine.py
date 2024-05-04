# import openai

# from langchain_openai import OpenAI
# from langchain.llms import openai
# from langchain_openai import ChatOpenAI
# from langchain.prompts import PromptTemplate, MessagesPlaceholder, HumanMessagePromptTemplate, ChatPromptTemplate
# from langchain.prompts import ChatPromptTemplate
# from langchain.memory import ConversationBufferMemory
# from langchain.chains import LLMChain
# from dotenv import load_dotenv
# from huggingface_hub import hf_hub_download
# from llama_cpp import Llama
# from transformers import BertTokenizer, BertModel
import torch


# from langchain.llms import HuggingFacePipeline
from huggingface_hub import login
from transformers import AutoTokenizer
# from langchain.chains import ConversationChain
# from langchain.chains import SimpleSequentialChain
import transformers
import torch
import warnings
warnings.filterwarnings('ignore')



class LLMEngine:
    def __init__(self, model, temperature = 0):
        self.model = model
        self.temperature = temperature
        self.llm = None # will be updated when the model is loaded
        self.prompt = None # will be update when first prompt created - do we really need it?

        self.hf_login()
        self.load_model()

    def debug_print(self):
        print("Hello from LLMEngine")

    def hf_login(self):
        login(token="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")

    # def load_model(self):
    #     print(f"******** Using Model {self.model} ********")
    #     # torch.cuda.empty_cache()
    #     tokenizer=AutoTokenizer.from_pretrained(self.model, legacy=False)
    #     pipeline=transformers.pipeline(
    #         "text-generation",           # Type of task to perform. In this case, generating text.
    #         model=self.model,
    #         tokenizer=tokenizer,         # Tokenizer to convert text inputs into a format the model can understand.
    #         torch_dtype=torch.bfloat16,  # Data type for processing, 'bfloat16' is used for efficient computation on supported hardware.
    #         trust_remote_code=True,      # Allows execution of custom code from the model (be cautious with this setting in production).
    #         device_map="auto",           # Automatically assigns the model's layers to available devices (like GPU or CPU).
    #         max_length=200,                # Maximum length of the sequence to be generated (will crash & fail if exceeded)
    #         max_new_tokens=250,            # Stop criteria - The maximum numbers of tokens to generate, ignoring the number of tokens in the prompt.
    #         do_sample=True,              # Enables sampling; picks words based on their probability distribution.
    #         top_k=1,                     # Limits the sampling pool to the top-k words.
    #         num_return_sequences=1,      # Number of sequences to return.
    #         eos_token_id=tokenizer.eos_token_id,   # End-of-sequence token ID used to signify the end of a generated sequence.
    #         return_full_text = False
    #         )
    #     self.llm=HuggingFacePipeline(pipeline=pipeline, model_kwargs={'temperature':self.temperature})

    def load_model(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model)
        pipeline = transformers.pipeline(
            "text-generation",
            model=self.model,
            torch_dtype=torch.float16,
            device_map="auto",
        )

        sequences = pipeline(
            'I liked "Breaking Bad" and "Band of Brothers". Do you have any recommendations of other shows I might like?\n',
            do_sample=True,
            top_k=10,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
            max_length=200,
        )
        for seq in sequences:
            print(f"Result: {seq['generated_text']}")

    # def build_prompt(self):
    #     prompt_template = """<s>[INST] <<SYS>>
    #     {{ You are a helpful AI Assistantl. Your answers should be brief up to 10 words only.}}<<SYS>>
    #     ###

    #     Previous Conversation:
    #     '''
    #     {history}
    #     '''

    #     {{{input}}}[/INST]

    #     """
    #     self.prompt = PromptTemplate(template=prompt_template, input_variables=['input', 'history'])

    # def build_prompt2(self):
    #     prompt_template = """<s>[INST] <<SYS>>
    #     {{You are recommender system for movies expert. I need you to answer the questions I am going to ask. Your answer should be movie name only in xml format}}<<SYS>>
    #     ###

    #     Previous Conversation:
    #     '''
    #     {history}
    #     '''

    #     {{{input}}}[/INST]

    #     """
    #     self.prompt = PromptTemplate(template=prompt_template, input_variables=['input', 'history'])

    # # def build_prompt3(self):
    # #     self.prompt = ChatPromptTemplate(
    # #         input_variables = ["content"],
    # #         messages = [
    # #             HumanMessagePromptTemplate.from_template("How are you today?")
    # #         ]
    # #     )

    # #     pass
    # def build_prompt3(self):
    #     # Assuming no input variables are actually needed
    #     # self.prompt1 = ChatPromptTemplate(
    #     self.prompt1 = PromptTemplate(
    #         input_variables = [],  # This needs to be an empty list if no input variables are expected
    #         messages = [
    #             HumanMessagePromptTemplate.from_template("How are you today?")
    #         ]
    #     )

    # # Basic Example
    # def build_prompt_chaining(self):
    #     template = "You are a naming consultant for new companies. What is a good \
    #         name for a company that makes {product}? Answer only with suggested company name."
    #     prompt = PromptTemplate.from_template(template)
    #     chain = LLMChain(llm = self.llm, prompt=prompt)
    #     print(chain.run("Cars"))

    # # Multiple variables
    # def build_prompt_chaining2(self):
    #     template = "You are a naming consultant for new companies. What is a good "\
    #                 + "name for a {company} that makes {product}?"
    #     prompt = PromptTemplate.from_template(template)
    #     chain = LLMChain(llm = self.llm, prompt=prompt)
    #     print(chain.run({'company': "ABC", 'product':"Cars"}))

    # # Sequential Chains
    # def build_prompt_chaining3(self):
    #     template = "What is a good name for a company that makes {product}? Answer very briefly"
    #     first_prompt = PromptTemplate.from_template(template)
    #     first_chain = LLMChain(llm=self.llm, prompt=first_prompt)
    #     print(first_chain.run("Cars"))
    #     print("--------------------------------------------------------------")
    #     second_template = "Write a catch phrase for the following company: {company_name}. Answer very briefly"
    #     second_prompt = PromptTemplate.from_template(second_template)
    #     second_chain = LLMChain(llm=self.llm, prompt=second_prompt)

    #     # overall_chain = SimpleSequentialChain(chains=[first_chain, second_chain], verbose=True)
    #     overall_chain = SimpleSequentialChain(chains=[first_chain, second_chain], verbose=False)

    #     catchphrase = overall_chain.run("Cars")
    #     print(catchphrase)

    #     # prompt = PromptTemplate.from_template(template)
    #     # chain = LLMChain(llm = self.llm, prompt=prompt)
    #     # print(chain.run({'company': "ABC", 'product':"Cars"}))

    # def build_prompt_chat_memory(self):
    #     memory = ConversationBufferMemory(memory_key="messages", return_messages=True)
    #     prompt = ChatPromptTemplate(
    #         input_variables=["content", "messages"],
    #         messages=[
    #             MessagesPlaceholder(variable_name="messages"),
    #             HumanMessagePromptTemplate.from_template("{content}")
    #         ]
    #     )
    #     chain = LLMChain(
    #         llm=self.llm,
    #         prompt=prompt,
    #         memory=memory
    #     )
    #     # result = chain({"content" : "Very brief answer, recommend the best movie from the 90s"})
    #     # print(result["text"])

    #     result = chain({"content" : "Act as caclculator and print result only. How much is 2+5?"})
    #     print(result["text"])
    #     print("-------------------------------------------------------------------------------------")
    #     result = chain({"content" : "And another 3?"})
    #     print(result["text"])
    #     print("-------------------------------------------------------------------------------------")
    #     result = chain({"content" : "I like Rocky and Rambo movies. Should I watch Terminator? Answer yes or no only."})
    #     print(result["text"])
    #     print("-------------------------------------------------------------------------------------")
    #     # chain.predict_and_parse()

    #     # Print available methods and attributes of the object
    #     # print(dir(chain.to_json()))
    #     # print(chain.to_json())

    #     # # If it’s a custom object with a dictionary storing data, perhaps you can access it directly:
    #     # if hasattr(chain, 'data') and isinstance(chain.data, dict):
    #     #     print(chain.data['text'])
    #     # print(chain.get_text())



    # def do_conversation3(self):
    #     chain = ConversationChain(llm=self.llm, prompt=self.prompt1)
    #     res = chain.run({})
    #     print(res)


    # def do_conversation2(self):
    #     chain = ConversationChain(llm=self.llm, prompt=self.prompt)
    #     res = chain.run("Recommend the top action movie from the 90s")
    #     print(res)


    # def do_conversation(self):
    #     chain = ConversationChain(llm=self.llm, prompt=self.prompt)
    #     res = chain.run("I want you to meet my friend, John Rambo. Please welcome him.")
    #     print(res)
    #     print("-------------------------------------------------------------------------------------")
    #     res = chain.run("What is my friend first name?")
    #     print(res)
    #     print("-------------------------------------------------------------------------------------")
    #     res = chain.run("What is my friend last name?")
    #     print(res)
    #     print("-------------------------------------------------------------------------------------")


    def debug_print(self):
        print(f"Model = {self.model}")


# Maybe all LLAMA is same interface, consider to remove 7B and make it generic
class Llama2_7B(LLMEngine):

    def __init__(self,):
        LLMEngine.__init__(self, "meta-llama/Llama-2-7b-chat-hf")

    def llm_config(self):
        pass

class Llama3_13B(LLMEngine):

    def __init__(self,):
        LLMEngine.__init__(self, "meta-llama/Llama-2-13b-chat-hf")

    def llm_config(self):
        pass


class Llama3_8B(LLMEngine):

    def __init__(self,):
        # LLMEngine.__init__(self, "meta-llama/Meta-Llama-3-8B")
        LLMEngine.__init__(self,"meta-llama/Meta-Llama-3-8B-Instruct")

    def llm_config(self):
        pass


class Mistral_7B(LLMEngine):

    def __init__(self,):
        LLMEngine.__init__(self, "mistralai/Mistral-7B-Instruct-v0.2")

    def llm_config(self):
        pass

    # def debug_print(self):
    #     print("Hello from MistralPromptEngine_7B")


class Vicuna_7B(LLMEngine):
    def __init__(self):
        LLMEngine.__init__(self, "lmsys/vicuna-7b-v1.3")

    def llm_config(self):
        pass

    # def debug_print(self):
    #     print("Hello from VicunaPromptEngine_7B") 

# class Prompt_Engineering_OpenAI_Handler(LLMEngine):
#     def debug_print(self):
#         print("Hello from Prompt_Engineering_OpenAI_Handler") 
    
#     def llm_config(self):
#         # llm = OpenAI(model="gpt-3.5-turbo")
#         llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

#         code_prompt = PromptTemplate(
#             template = "Write a very {language} function that will {task}",
#             input_variables = ["language", "task"]
#         )

#         code_chain = LLMChain(
#             llm = llm,
#             prompt = code_prompt
#         )
#         result = code_chain({
#             "language": "Python",
#             "task": "Return a list of numbers"
#         })

#         print(result)
