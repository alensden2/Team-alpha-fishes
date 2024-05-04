import argparse
import os

# from datasets_handler.datasets_main import *
# from prompts.prompts_engine import *
from llms_interface.llms_engine import *

# gpustat -cp
# nvidia-smi
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


def parse_input_args():
    # Create the parser
    parser = argparse.ArgumentParser(description="Prompt Engineering Recommender Inputs")
    
    # Define arguments
    parser.add_argument("-t", "--temperature", type=float, default=0,
                        help="Optional. Specify the temperature value ranging from 0 to 1.\n"
                             "Default is 0.")
    parser.add_argument("-l", "--llm", type=str, default="gpt-3.5-turbo", choices=["gpt-3.5-turbo", "llama2"],
                        help="Optional. Choose the type of language learning model:\n"
                             "- 'gpt-3.5-turbo' (Default)\n"
                             "- 'llama2'\n")
    parser.add_argument("-d", "--dataset", type=str, default="ml_small_100k", choices=["ml_small_100k", "ml_full_33M"],
                        help="Optional. Select the dataset to be used:\n"
                             "- 'ml_small_100k' (Default)\n"
                             "- 'ml_full_33M'")

    # Parse arguments
    args = parser.parse_args()
    return args





def main():
    args = parse_input_args()
    # datasets_main(args.dataset)
    # dataset_handler = Dataset_Handler(args.dataset)


    # print(create_prompt("5", "Scream"))


    # tests_llama_2()
    # tests_llama_3()

    llama_3_13B_obj = Llama3_13B()
    # p1 = PromptEngine(llama_3_13B_obj.llm)
    # p1.chat_conversation()

    # llama_3_8B_obj = Llama3_8B()
    # p1 = PromptEngine(llama_3_8B_obj.llm)
    # p1.chat_conversation()

    # llama_2_7B_obj = Llama2_7B()
    # p1 = PromptEngine(llama_2_7B_obj.llm)
    # p1.chat_conversation()
    
    # p1.chat_history_summary()


    # llm_openai = Prompt_Engineering_OpenAI_Handler()
    # llm_openai.llm_config()

    # llm_llama_7B = Prompt_Engineering_LLAMA_2_7B_Handler()
    # llm_llama_7B.llm_config()

    # bert_llm = Prompt_Engineering_BERT_Handler()
    # bert_llm.llm_config()

    # llama_2_7B_obj = Llama2PromptEngine_7B()
    # llama_2_7B_obj.debug_print()
    # llama_2_7B_obj.hf_login()
    # llama_2_7B_obj.load_model()


    # llama_2_7B_obj.build_prompt_chat_memory()


    # llama_2_7B_obj.build_prompt()
    # llama_2_7B_obj.do_conversation()
    # llama_2_7B_obj.do_conversation()
    # llama_2_7B_obj.do_conversation()

    # mistral_7B_obj = MistralPromptEngine_7B()
    # mistral_7B_obj.debug_print()
    # mistral_7B_obj.hf_login()
    # mistral_7B_obj.load_model()
    # mistral_7B_obj.build_prompt_chat_memory()
    # mistral_7B_obj.build_prompt3()
    # mistral_7B_obj.do_conversation3()

    # vicuna_7B_obj = VicunaPromptEngine_7B()
    # vicuna_7B_obj.debug_print()
    # vicuna_7B_obj.hf_login()
    # vicuna_7B_obj.load_model()
    # vicuna_7B_obj.build_prompt()
    # vicuna_7B_obj.do_conversation()
    # vicuna_7B_obj.do_conversation()
    # vicuna_7B_obj.do_conversation()

if __name__ == "__main__":
    main()

