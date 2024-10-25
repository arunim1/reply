import asyncio
import openai
import anthropic
import google.generativeai as palm

from openai import AsyncOpenAI
from anthropic import AsyncAnthropic

import os
import time
import torch
import gc
from typing import Dict, List
from dotenv import load_dotenv


load_dotenv()

OPENAI_CONCURRENCY = 10
ANTHROPIC_CONCURRENCY = 10

class LanguageModel():
    def __init__(self, model_name):
        self.model_name = model_name
    
    def batched_generate(self, prompts_list: List, max_n_tokens: int, temperature: float):
        """
        Generates responses for a batch of prompts using a language model.
        """
        raise NotImplementedError
        
class HuggingFace(LanguageModel):
    def __init__(self,model_name, model, tokenizer):
        self.model_name = model_name
        self.model = model 
        self.tokenizer = tokenizer
        self.eos_token_ids = [self.tokenizer.eos_token_id]

    def batched_generate(self, 
                        full_prompts_list,
                        max_n_tokens: int, 
                        temperature: float,
                        top_p: float = 1.0,):
        raise NotImplementedError
        inputs = self.tokenizer(full_prompts_list, return_tensors='pt', padding=True)
        inputs = {k: v.to(self.model.device.index) for k, v in inputs.items()}
    
        # Batch generation
        if temperature > 0:
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_n_tokens, 
                do_sample=True,
                temperature=temperature,
                eos_token_id=self.eos_token_ids,
                top_p=top_p,
            )
        else:
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_n_tokens, 
                do_sample=False,
                eos_token_id=self.eos_token_ids,
                top_p=1,
                temperature=1, # To prevent warning messages
            )
            
        # If the model is not an encoder-decoder type, slice off the input tokens
        if not self.model.config.is_encoder_decoder:
            output_ids = output_ids[:, inputs["input_ids"].shape[1]:]

        # Batch decoding
        outputs_list = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)

        for key in inputs:
            inputs[key].to('cpu')
        output_ids.to('cpu')
        del inputs, output_ids
        gc.collect()
        torch.cuda.empty_cache()

        return outputs_list

    def extend_eos_tokens(self):        
        # Add closing braces for Vicuna/Llama eos when using attacker model
        self.eos_token_ids.extend([
            self.tokenizer.encode("}")[1],
            29913, 
            9092,
            16675])

class GPT(LanguageModel):
    API_RETRY_SLEEP = 10
    API_ERROR_OUTPUT = "$ERROR$"
    API_QUERY_SLEEP = 0.5
    API_MAX_RETRY = 5
    API_TIMEOUT = 20
    
    def __init__(self, model_name):
        super().__init__(model_name)
        self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        # Add semaphore for concurrency control
        self.semaphore = asyncio.Semaphore(OPENAI_CONCURRENCY)

    async def generate(self, conv: List[Dict], 
                max_n_tokens: int, 
                temperature: float,
                top_p: float):
        # ... existing code ...
        if "o1" in self.model_name:
            system_prompt = conv[0]["content"] # assume that the first message is the system prompt
            conv[1]["content"] = f"<system>{system_prompt}</system>\n{conv[1]['content']}"
            conv = conv[1:]
        async with self.semaphore:  # Add semaphore context manager
            output = self.API_ERROR_OUTPUT
            for _ in range(self.API_MAX_RETRY):
                try:
                    if "o1" in self.model_name:
                        response = await self.client.chat.completions.create(
                                    model=self.model_name,
                                    messages=conv,
                                    timeout=self.API_TIMEOUT,
                                    )
                    else:
                        response = await self.client.chat.completions.create(
                                    model=self.model_name,
                                    messages=conv,
                                    max_tokens=max_n_tokens,
                                    temperature=temperature,
                                    top_p=top_p,
                                    timeout=self.API_TIMEOUT,
                                    )
                    output = response.choices[0].message.content
                    break
                except Exception as e:
                    print(type(e), e)
                    await asyncio.sleep(self.API_RETRY_SLEEP)
            
                await asyncio.sleep(self.API_QUERY_SLEEP)
            return output 
    
    async def batched_generate(self, 
                        convs_list: List[List[Dict]],
                        max_n_tokens: int, 
                        temperature: float,
                        top_p: float = 1.0,):
        tasks = [self.generate(conv, max_n_tokens, temperature, top_p) for conv in convs_list]
        return await asyncio.gather(*tasks)

class Claude():
    API_RETRY_SLEEP = 10
    API_ERROR_OUTPUT = "$ERROR$"
    API_QUERY_SLEEP = 1
    API_MAX_RETRY = 5
    API_TIMEOUT = 20
    API_KEY = os.getenv("ANTHROPIC_API_KEY")
   
    def __init__(self, model_name) -> None:
        self.model_name = model_name
        self.client = AsyncAnthropic(api_key=self.API_KEY)
        self.semaphore = asyncio.Semaphore(2)

    async def generate(self, conv: List, 
                max_n_tokens: int, 
                temperature: float,
                top_p: float):
        async with self.semaphore:
            output = self.API_ERROR_OUTPUT
            for _ in range(self.API_MAX_RETRY):
                try:
                    response = await self.client.messages.create(
                        model=self.model_name,
                        max_tokens=max_n_tokens,
                        messages=[
                            {**msg, 'content': msg['content'].strip()} 
                            for msg in conv[1:]
                        ],
                        temperature=temperature,
                        top_p=top_p,
                        system=conv[0]["content"]
                    )
                    output = response.content[0].text
                    break
                except anthropic.APIError as e:
                    print("HERE", type(e), e, f"\n{conv}")
                    await asyncio.sleep(self.API_RETRY_SLEEP)
            
                await asyncio.sleep(self.API_QUERY_SLEEP)
            return output
    
    async def batched_generate(self, 
                        convs_list: List[List[Dict]],
                        max_n_tokens: int, 
                        temperature: float,
                        top_p: float = 1.0,):
        tasks = [self.generate(conv, max_n_tokens, temperature, top_p) for conv in convs_list]
        return await asyncio.gather(*tasks)
        
class PaLM():
    API_RETRY_SLEEP = 10
    API_ERROR_OUTPUT = "$ERROR$"
    API_QUERY_SLEEP = 1
    API_MAX_RETRY = 5
    API_TIMEOUT = 20
    default_output = "I'm sorry, but I cannot assist with that request."
    API_KEY = os.getenv("PALM_API_KEY")

    def __init__(self, model_name) -> None:
        self.model_name = model_name
        palm.configure(api_key=self.API_KEY)

    def generate(self, conv: List, 
                max_n_tokens: int, 
                temperature: float,
                top_p: float):
        '''
        Args:
            conv: List of dictionaries, 
            max_n_tokens: int, max number of tokens to generate
            temperature: float, temperature for sampling
            top_p: float, top p for sampling
        Returns:
            str: generated response
        '''
        raise NotImplementedError
        output = self.API_ERROR_OUTPUT
        for _ in range(self.API_MAX_RETRY):
            try:
                completion = palm.chat(
                    messages=conv,
                    temperature=temperature,
                    top_p=top_p
                )
                output = completion.last
                
                if output is None:
                    # If PaLM refuses to output and returns None, we replace it with a default output
                    output = self.default_output
                else:
                    # Use this approximation since PaLM does not allow
                    # to specify max_tokens. Each token is approximately 4 characters.
                    output = output[:(max_n_tokens*4)]
                break
            except Exception as e:
                print(type(e), e)
                time.sleep(self.API_RETRY_SLEEP)
        
            time.sleep(1)
        return output
    
    def batched_generate(self, 
                        convs_list: List[List[Dict]],
                        max_n_tokens: int, 
                        temperature: float,
                        top_p: float = 1.0,):
        return [self.generate(conv, max_n_tokens, temperature, top_p) for conv in convs_list]
