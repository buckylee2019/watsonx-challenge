from langchain.llms.base import LLM
from typing import Optional, List, Mapping, Any
import os
from langchain import PromptTemplate, LLMChain
from time import sleep
import requests
import json
from dotenv import load_dotenv
from genai.credentials import Credentials
from genai.model import Model
from genai.schemas import GenerateParams
from genai.credentials import Credentials
import LLM.watsonx_example.due_diligence as due_diligence 

load_dotenv()
api_key = os.getenv("GENAI_KEY", None) 
api_url = os.getenv("GENAI_API", None)
creds = Credentials(api_key, api_endpoint=api_url)

class ChatBot():
    def __init__(self) -> None:
        return None
    def chat(self, prompt):
        params = GenerateParams(
            decoding_method="sample",
            max_new_tokens=1536,
            min_new_tokens=10,
            temperature=0.7,
        )
        lan_model = Model("bigscience/bloom", params=params, credentials=creds)
        return lan_model.generate([prompt])


class IBMChat(LLM):
    
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
        data = self.chatbot.chat(prompt=prompt)
        #conversation_list = self.chatbot.get_conversation_list()
        #print(conversation_list)
        
        #add to history
        self.history_data.append({"prompt":prompt,"response":data})    
        return data[0].generated_text

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {"model": "IBM watsonx"}


def watsonx_ref_model(name, title, content):
    params = GenerateParams(decoding_method="sample", max_new_tokens=300, temperature=0.1)
    LLM = Model("ibm/mpt-7b-instruct", params=params, credentials=creds)
    applicant_prompt = f"Extract {name}'s below information from below content"
    format_prompt = 'committed crime name, sentence, involved in news title, if the field is not mentioned fill it with N/A Make sure every information you display is found in the given news. Allowed format: {"news_title" : "Title of news", "name": "person name in the news", "committed_crime_name" : "Crime that the person is possibly committed in the news;sentence:Sentence that given by the judge in the news"} Response ONLY in list format and the value should be its original text, DO NOT translate'
    news_prompt = f"""
        Title: {title}
        Content: {content}
    """
    applicant_content = applicant_prompt+format_prompt+news_prompt

    json_data = f"""{due_diligence.template_1}
        Transcript:{applicant_content}
        Summary:
        """

    response = LLM.generate([json_data])
    print(response[0].generated_text)
    
    result = json.loads(response[0].generated_text)
    return result

# llm = IBMChat() #for start new chat

# print(llm("Hello, how are you?"))

# for result in ChatBot().chat('Hello! How are you?':
#     print("\t {}".format(result.generated_text))
