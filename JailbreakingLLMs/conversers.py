
import common
from language_models import GPT, Claude, PaLM, HuggingFace
import torch
# from transformers import AutoModelForCausalLM, AutoTokenizer
from config import VICUNA_PATH, LLAMA_PATH, ATTACK_TEMP, TARGET_TEMP, ATTACK_TOP_P, TARGET_TOP_P   

def load_attack_and_target_models(args):
    # Load attack model and tokenizer
    attackLM = AttackLM(model_name = args.attack_model, 
                        max_n_tokens = args.attack_max_n_tokens, 
                        max_n_attack_attempts = args.max_n_attack_attempts, 
                        temperature = ATTACK_TEMP, # init to 1
                        top_p = ATTACK_TOP_P, # init to 0.9
                        )
    preloaded_model = None
    if args.attack_model == args.target_model:
        print("Using same attack and target model. Using previously loaded model.")
        preloaded_model = attackLM.model
    targetLM = TargetLM(model_name = args.target_model, 
                        max_n_tokens = args.target_max_n_tokens,
                        temperature = TARGET_TEMP, # init to 0
                        top_p = TARGET_TOP_P, # init to 1
                        preloaded_model = preloaded_model,
                        )
    return attackLM, targetLM

class AttackLM():
    """
        Base class for attacker language models.
        
        Generates attacks for conversations using a language model. The self.model attribute contains the underlying generation model.
    """
    def __init__(self, 
                model_name: str, 
                max_n_tokens: int, 
                max_n_attack_attempts: int, 
                temperature: float,
                top_p: float):
        
        self.model_name = model_name
        self.temperature = temperature
        self.max_n_tokens = max_n_tokens
        self.max_n_attack_attempts = max_n_attack_attempts
        self.top_p = top_p
        self.model, self.template = load_indiv_model(model_name)
        
        if "vicuna" in model_name or "llama" in model_name:
            self.model.extend_eos_tokens()

    async def get_attack(self, convs_list, prompts_list, reply=False):
        """
        Generates responses for a batch of conversations and prompts using a language model. 
        Only valid outputs in proper JSON format are returned. If an output isn't generated 
        successfully after max_n_attack_attempts, it's returned as None.
        
        Parameters:
        - convs_list: List of conversation objects.
        - prompts_list: List of prompts corresponding to each conversation.
        
        Returns:
        - List of generated outputs (dictionaries) or None for failed generations.
        """
        
        assert len(convs_list) == len(prompts_list), "Mismatch between number of conversations and prompts."
        
        batchsize = len(convs_list)
        indices_to_regenerate = list(range(batchsize))
        valid_outputs = [None] * batchsize

        # Initalize the attack model's generated output to match format
        if len(convs_list[0].messages) == 0:
            init_message = """{\"improvement\": \"\",\"pre_prompt\": \"""" if reply else """{\"improvement\": \"\",\"prompt\": \""""
        else:
            init_message = """{\"improvement\": \"""" 

        full_prompts = []
        # Add prompts and initial seeding messages to conversations (only once)
        for conv, prompt in zip(convs_list, prompts_list):
            conv.append_message(conv.roles[0], prompt)
            # Get prompts
            if "gpt" in self.model_name or "o1" in self.model_name:
                full_prompts.append(conv.to_openai_api_messages())
            elif "claude" in self.model_name:
                full_prompts.append(conv.to_openai_api_messages())
            else:
                conv.append_message(conv.roles[1], init_message)
                full_prompts.append(conv.get_prompt()[:-len(conv.sep2)])
            
        for attempt in range(self.max_n_attack_attempts):
            # Subset conversations based on indices to regenerate
            full_prompts_subset = [full_prompts[i] for i in indices_to_regenerate]

            # Generate outputs 
            outputs_list = await self.model.batched_generate(full_prompts_subset,
                                                        max_n_tokens = self.max_n_tokens,  
                                                        temperature = self.temperature,
                                                        top_p = self.top_p
                                                    )
            
            # Check for valid outputs and update the list
            new_indices_to_regenerate = []
            for i, full_output in enumerate(outputs_list):
                orig_index = indices_to_regenerate[i]
                if "gpt" not in self.model_name and "o1" not in self.model_name and "claude" not in self.model_name:
                    full_output = init_message + full_output

                attack_dict, json_str = common.extract_json(full_output, reply=reply)
                
                if attack_dict is not None:
                    valid_outputs[orig_index] = attack_dict
                    convs_list[orig_index].update_last_message(json_str)  # Update the conversation with valid generation
                else:
                    new_indices_to_regenerate.append(orig_index)
            
            # Update indices to regenerate for the next iteration
            indices_to_regenerate = new_indices_to_regenerate
            
            # If all outputs are valid, break
            if not indices_to_regenerate:
                break
        
        if any([output for output in valid_outputs if output is None]):
            print(f"Failed to generate output after {self.max_n_attack_attempts} attempts. Terminating.")
        return valid_outputs

class TargetLM():
    """
        Base class for target language models.
        
        Generates responses for prompts using a language model. The self.model attribute contains the underlying generation model.
    """
    def __init__(self, 
            model_name: str, 
            max_n_tokens: int, 
            temperature: float,
            top_p: float,
            preloaded_model: object = None):
        
        self.model_name = model_name
        self.temperature = temperature
        self.max_n_tokens = max_n_tokens
        self.top_p = top_p
        if preloaded_model is None:
            self.model, self.template = load_indiv_model(model_name)
        else:
            self.model = preloaded_model
            _, self.template = get_model_path_and_template(model_name)

    async def get_response(self, prompts_list, pre_prompt_list = None, pre_response_list = None):
        batchsize = len(prompts_list)
        convs_list = [common.conv_template(self.template) for _ in range(batchsize)]
        if pre_prompt_list is None:
            pre_prompt_list = [None] * batchsize
        if pre_response_list is None:
            pre_response_list = [None] * batchsize
        full_prompts = []
        for conv, prompt, pre_prompt, pre_response in zip(convs_list, prompts_list, pre_prompt_list, pre_response_list):
            if pre_prompt is not None:
                conv.append_message(conv.roles[0], pre_prompt)
            if pre_response is not None:
                conv.append_message(conv.roles[1], pre_response)
            conv.append_message(conv.roles[0], prompt)
            if "gpt" in self.model_name or "o1" in self.model_name:
                # Openai does not have separators
                full_prompts.append(conv.to_openai_api_messages())
            elif "claude" in self.model_name:
                full_prompts.append(conv.to_openai_api_messages())
            elif "palm" in self.model_name:
                full_prompts.append(conv.messages[-1][1])
            else:
                conv.append_message(conv.roles[1], None) 
                full_prompts.append(conv.get_prompt())
        
        outputs_list = await self.model.batched_generate(full_prompts, 
                                                        max_n_tokens = self.max_n_tokens,  
                                                        temperature = self.temperature,
                                                        top_p = self.top_p
                                                    )
        return outputs_list



def load_indiv_model(model_name, device=None):
    model_path, template = get_model_path_and_template(model_name)
    if model_name.startswith("gpt") or model_name.startswith("o1"):
        lm = GPT(model_name)
    elif model_name.startswith("claude"):
        lm = Claude(model_name)
    elif model_name in ["palm-2"]:
        raise NotImplementedError
        lm = PaLM(model_name)
    else:
        raise NotImplementedError
        model = AutoModelForCausalLM.from_pretrained(
                model_path, 
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,device_map="auto").eval()

        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            use_fast=False
        ) 

        if 'llama-2' in model_path.lower():
            tokenizer.pad_token = tokenizer.unk_token
            tokenizer.padding_side = 'left'
        if 'vicuna' in model_path.lower():
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = 'left'
        if not tokenizer.pad_token:
            tokenizer.pad_token = tokenizer.eos_token

        lm = HuggingFace(model_name, model, tokenizer)
    
    return lm, template

def get_model_path_and_template(model_name):
    # full_model_dict={
    #     "gpt-4":{
    #         "path":"gpt-4",
    #         "template":"gpt-4"
    #     },
    #     "gpt-3.5-turbo": {
    #         "path":"gpt-3.5-turbo",
    #         "template":"gpt-3.5-turbo"
    #     },
    #     "vicuna":{
    #         "path":VICUNA_PATH,
    #         "template":"vicuna_v1.1"
    #     },
    #     "llama-2":{
    #         "path":LLAMA_PATH,
    #         "template":"llama-2"
    #     },
    #     "claude-instant-1":{
    #         "path":"claude-instant-1",
    #         "template":"claude-instant-1"
    #     },
    #     "claude-2":{
    #         "path":"claude-2",
    #         "template":"claude-2"
    #     },
    #     "palm-2":{
    #         "path":"palm-2",
    #         "template":"palm-2"
    #     }
    # }
    # path, template = full_model_dict[model_name]["path"], full_model_dict[model_name]["template"]
    # return path, template
    return model_name, model_name

    