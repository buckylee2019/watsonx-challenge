import requests
from langchain.llms.base import LLM
from typing import Optional, List, Mapping, Any
import pydantic
import os
from langchain import PromptTemplate, LLMChain
from time import sleep

import requests
import json
class ChatBot():
    def __init__(self) -> None:
        self.url = "http://150.240.64.87:5000/v1/chat/completions"
    def chat(self, prompt, temperature=0.5, streaming=False):
        headers = {
            'accept': 'application/json',
            'Content-Type': 'application/json'
            }   
        payload =  json.dumps({
                "messages": [
                    {
                        "role": "user",
                        "content": prompt

                    }
                ]})
        response = requests.request("POST", self.url,headers=headers, data=payload)
        return response.json()["choices"][0]["message"]["content"]



class MetaChat(LLM):
    
    history_data: Optional[List] = []
    chatbot : Optional[ChatBot] = ChatBot()
    conversation : Optional[str] = ""
    cookiepath : Optional[str]
    #### WARNING : for each api call this library will create a new chat on chat.openai.com
    
    
    @property
    def _llm_type(self) -> str:
        return "custom"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        if stop is not None:
            pass
            #raise ValueError("stop kwargs are not permitted.")
        #token is a must check
        if self.chatbot is None:
            if self.cookiepath is None:
                ValueError("Cookie path is required, pls check the documentation on github")
            else: 
                if self.conversation == "":
                    print('Here')
                    self.chatbot = ChatBot()
                else:
                    raise ValueError("Something went wrong")
            
        
        sleep(2)
        data = self.chatbot.chat(prompt=prompt, temperature=0.5, streaming=False)
        #conversation_list = self.chatbot.get_conversation_list()
        #print(conversation_list)
        
        #add to history
        self.history_data.append({"prompt":prompt,"response":data})    
        
        return data

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {"model": "ClaudeCHAT"}



# llm = ClaudeChat() #for start new chat


# print(llm("Hello, how are you?"))
#print(llm("what is AI?"))
#print(llm("Can you resume your previus answer?")) #now memory work well

