import tiktoken  # !pip install tiktoken
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tqdm.auto import tqdm
from uuid import uuid4
from dotenv import load_dotenv
import os
from LLM import ChatGPTAPI_unsafe  # FREE GOOGLE BARD API
import nest_asyncio
from requests import post
import re
from langchain.vectorstores import Pinecone
from langchain.prompts import PromptTemplate
from ibm_watson import DiscoveryV2
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
import json
from flask import Flask, request, jsonify
from ibm_watson import DiscoveryV2
import openai
from langchain.embeddings import HuggingFaceHubEmbeddings
from langchain.chains import LLMChain

from neo4j_import import create_node_and_relationship
load_dotenv()




SEARCH_TOP_K = 20
# Set up Watson Discovery credentials
api_key = os.environ.get("WATSON_API_KEY")
url = os.environ.get("WATSON_URL")
project_id = os.environ.get("WATSON_PROJECT_ID")
collection_list = eval(os.environ.get("WD_COLLECTION_IDS"))
DATABASE_BEARER_TOKEN = os.environ.get("DATABASE_BEARER_TOKEN")
authenticator = IAMAuthenticator(api_key)
discovery = DiscoveryV2(version="2021-09-01", authenticator=authenticator)
discovery.set_service_url(url)



def query_watson_discovery(query,topk = 30):
    return discovery.query(
        project_id=project_id,
        collection_ids=collection_list,
        natural_language_query=query,
        passages={
            "enabled":True,
            "find_answers":True,
            "per_document":True,
            "fields":["text"],
            "characters":280,
            "max_per_document":1},
        count = topk
        
    ).get_result()
def tiktoken_len(text):
    
    return len(text)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=20,
    length_function=len,
    separators=["\n\n"]
)

embedding_size = 384 # if you change this you need to change also in Embedding/HuggingFaceEmbedding.py
repo_id = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
hf = HuggingFaceHubEmbeddings(
    repo_id=repo_id,
    huggingfacehub_api_token=os.environ.get("HUGGINGFACEHUB_API_TOKEN"),
)

# index_name = 'knowledgebase-index'

# pinecone.init(
#         api_key= os.environ.get("PINECONE_API_KEY"),  # find api key in console at app.pinecone.io
#         environment=os.environ.get("PINECONE_ENVIRONMENT")  # find next to api key in console
# )

# index = pinecone.Index(index_name)
# index.describe_index_stats()

batch_limit = 5


def update_pinecone(query):
    texts = []
    metadatas = []
    result = query_watson_discovery(query)
    v_query = hf.embed_query(query)
    for i, record in enumerate(result["results"]):
        # first get metadata fields for this record
        find_exist = index.query(vector = v_query,top_k=3,filter=
                       {"doc-id":{"$eq":record['document_id']}},include_metadata=True)

        if(find_exist['matches']):
            print("search Exist")
            print(find_exist['matches'])
            continue
        metadata = {
            'doc-id': str(record['document_id']),
            'source': record['extracted_metadata']['filename'],
            'title': record['title']
        }
        # now we create chunks from the record text
        record_texts = text_splitter.split_text(record['text'][0])
        # create individual metadata dicts for each chunk
        record_metadatas = [{
            "chunk": j, "text": text, **metadata
        } for j, text in enumerate(record_texts)]
        # append these to current batches
        texts.extend(record_texts)
        metadatas.extend(record_metadatas)
        # if we have reached the batch_limit we can add texts

        
        ids = [str(uuid4()) for _ in range(len(texts))]
        # embeds = []
        # for text in texts:
        #     embeds.append(embeddings_model(text))
        embeds = hf.embed_documents(texts)

        index.upsert(vectors=zip(ids, embeds, metadatas),namespace="News")
        texts = []
        metadatas = []

# update_pinecone("吳宗憲")


# text_field = "text"

# # switch back to normal index for langchain
# index = pinecone.Index(index_name)

# vectorstore = Pinecone(
#     index, hf.embed_query, text_field
# )
# vectorstore._namespace = "News"
# query = "陳玉珠"
# query_response = vectorstore.similarity_search(
#     query,  # our search query
#     k=5  # return 3 most relevant docs
# )



model = "gpt-3.5-turbo-0613"
llm= ChatGPTAPI_unsafe.ChatGPT(token=os.environ.get("CHATGPT_TOKEN"), conversation=os.environ.get("CONVERSATION_ID_1"))


llm2json= ChatGPTAPI_unsafe.ChatGPT(token=os.environ.get("CHATGPT_TOKEN"), conversation=os.environ.get("CONVERSATION_ID_2"))
prompt_template =  """{context}
Check if the given article a crime news or not, NOTE: ONLY ANSWER TRUE OR FALSE
"""

# PROMPT = PromptTemplate(
#     template=prompt_template, input_variables=["context", "topic"]
# )

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context"]
)
unique_chain = LLMChain(llm=llm, prompt=PROMPT)

prompt_template =  """{context}
Extract name, gender, age, position,works for, committed crime name, sentence, involved in news title, if the field is not mentioned fill it with N/A

Allowed format:
{{
"news_title": " Title of news",
"suspects": [
{{
"Name": "perosn1 name",
"Gender": "Gender of the perosn",
"Age": "Age of the person",
"Position": "Job title of position of the person",
"Works For": "Company or organization name",
"Committed Crime Name": "Crime that the person is possibly committed",
"Sentence": "Sentence that given by the judge"
}},
{{
"Name": "perosn2 name",
"Gender": "Gender of the perosn",
"Age": "Age of the person",
"Position": "Job title of position of the person",
"Works For": "Company or organization name",
"Committed Crime Name": "Crime that the person is possibly committed",
"Sentence": "Sentence that given by the judge"
}}
]
}}
Response ONLY in JSON format"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context"]
)
extract_chain = LLMChain(llm=llm2json, prompt=PROMPT)
# llm("hello, what is quantun computing?")
def retrieve_extraction(topic):
    docs = query_watson_discovery(topic, topk=10)
    # docs = vectorstore.similarity_search(topic, k=5)
    # context = '\n'.join([doc.page_content for doc in docs])
    # For classifying news (Crime or not)
    contexts = ['Title: ' + doc['title'] +'\n Content: ' + doc['text'][0] for doc in docs["results"] ]
    outputs_classfied = [{"Crime":unique_chain.apply([{"context":context}])[0]['text'],"context":context} for context in contexts]
    with open('output/classified_news.json','w') as f:
        json.dump(outputs_classfied, f,ensure_ascii=False)

    # For generation CSV
    outputs_json = [(extract_chain.apply([{"context":out["context"]}])[0]['text'],out["context"]) for out in outputs_classfied if out['Crime']=="TRUE"]
    
    return outputs_json
    
output_extract_chain = retrieve_extraction("吳建宏的犯罪新聞")
news_json =[]
for output,content in output_extract_chain:
    jsonout = eval(output)
    pattern = r'Content: (.+)'

    # Use re.search to find the first occurrence of the pattern.
    match = re.search(pattern, content)
    jsonout['News'] = match.group(1)
    news_json.append(jsonout)
with open('output/extracted_news.json','w') as f:
    json.dump(news_json, f, ensure_ascii=False)
for data in news_json:
    for row in data['suspects']:
        row["Involved in News Title"] = data['news_title']
        row['News Content'] = data["News"]
        create_node_and_relationship(row)
# prompt_template =  """請依照Context中提供的多篇文章擷取資訊總結並整理出CSV 結果：
# Conext: {text}
# Summarized Output:"""

# PROMPT = PromptTemplate(
#     template=prompt_template, input_variables=["text"]
# )


# summarize_chain = LLMChain(llm=llm, prompt=PROMPT)
# summarize_chain.apply([{"text":news_info}])