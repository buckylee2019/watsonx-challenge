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
from ibm_watson_machine_learning.foundation_models import Model
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
from ibm_watson_machine_learning.foundation_models.utils.enums import ModelTypes, DecodingMethods

# load_dotenv()

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

class watsonx():
    def __init__(self, api_key ,api_endpoint):
    
        # To display example params enter
        GenParams().get_example_values()

        generate_params = {
            GenParams.MAX_NEW_TOKENS: 300
        }

        self.model = Model(
            model_id=ModelTypes.MPT_7B_INSTRUCT2,
            params=generate_params,
            credentials={
                "apikey": api_key,
                "url": api_endpoint
            },
            project_id="5dec0b1d-c146-4422-830d-4185d36e5223"
            )
    def watsonx_ref_model(self, name, title, content):
        
        
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
        
        response = self.model.generate(json_data)['results'][0]['generated_text']
        
        # fixjson = f"Fix the following json to valid syntax and response JSON Format only: {response}"
        # jsonout = self.model.generate(fixjson)['results'][0]['generated_text']
        # result = jsonout['results'][0]['generated_text']
        return json.loads(response)

# llm = IBMChat() #for start new chat

# print(llm("Hello, how are you?"))

# for result in ChatBot().chat('Hello! How are you?':
#     print("\t {}".format(result.generated_text))
