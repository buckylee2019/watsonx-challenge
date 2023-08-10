import tiktoken  # !pip install tiktoken
import time
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tqdm.auto import tqdm
from uuid import uuid4
from dotenv import load_dotenv
import os
from LLM import ChatGPTAPI_unsafe  # FREE GOOGLE BARD API
from LLM import MetaChatAPI  # FREE GOOGLE BARD API
import nest_asyncio
from requests import post
from pathlib import Path
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
import numpy as np
from sklearn.cluster import OPTICS
import matplotlib.pyplot as plt
# import hdbscan
from preprocess.neo4j_import import create_node_and_relationship,delete_nodes
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
load_dotenv()
cwd = os.path.dirname(os.path.abspath(__file__))


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
PERSIST_DIRECTORY = cwd+'/chroma_db'


def query_watson_discovery(query,topk = 30, offset = 0 ):
    return discovery.query(
        project_id=project_id,
        collection_ids=collection_list,
        query=query,
        offset=offset,
        passages={
            "enabled":True,
            "find_answers":True,
            "per_document":True,
            "fields":["text"],
            "characters":280,
            "max_per_document":1},
        count = topk
        
    ).get_result()
def find_similar_docs(document_ids,topk=10):
    return discovery.query(
        project_id=project_id,
        collection_ids=collection_list,
        similar={
            "enagbles":True,
            "document_ids":document_ids
            },
        passages={
            "enabled":True,
            "find_answers":True,
            "per_document":True,
            "fields":["text"],
            "characters":280,
            "max_per_document":1},
        count = topk).get_result()
def tiktoken_len(text):
    
    return len(text)




class vector_store():
    def __init__(self,persist_dir = PERSIST_DIRECTORY):
        repo_id = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        
        self.hf = HuggingFaceHubEmbeddings(
            repo_id=repo_id,
            huggingfacehub_api_token=os.environ.get("HUGGINGFACEHUB_API_TOKEN"),
        )

        if Path(persist_dir).exists():
            self.vectorstore = Chroma(persist_directory=persist_dir, embedding_function=self.hf)
            
        else:
            self.vectorstore = Chroma.from_documents([Document(page_content="",metadata={'intial':'true'})], self.hf,persist_directory=persist_dir)
            self.vectorstore.persist()

    def clustering(self, docs, topic):
        
        # texts = ['Title: ' + doc['title'] +'\n Content: ' + doc['text'][0] for doc in docs["results"] ]
        documents = [Document(page_content=doc) for doc,ids in docs]
        ids = [ids for doc,ids in docs]
        embeds = self.hf.embed_documents([doc for doc,ids in docs])
        
        # optics_model = OPTICS(min_samples=2, xi=0.03) # Adjust parameters as needed
        # optics_model.fit(embeds)
        # labels = optics_model.labels_
        hdbscan_model = hdbscan.HDBSCAN(min_cluster_size=2, min_samples=3) # Adjust parameters as needed
        hdbscan_model.fit(embeds)
        metadatas = [{"label":f'{topic}_{label}','maxlabel':str(max(hdbscan_model.labels_))} for label in hdbscan_model.labels_]
        labels = [f'{topic}_{label}' for label in hdbscan_model.labels_]
        for doc, mt in zip(documents,metadatas):
            doc.metadata=mt
        self.vectorstore.add_documents(documents = documents,ids = ids, embeddings = embeds)
        return labels


    def add2cluster(self, docs, topic):
        
        # texts = ['Title: ' + doc['title'] +'\n Content: ' + doc['text'][0] for doc in docs["results"] ]
        documents = [Document(page_content=doc) for doc,ids in docs]
        ids = [ids for doc,ids in docs]

        scores = [{"score":self.vectorstore.similarity_search_with_score(query = doc)} for doc, docid in docs]
        scores = [{"score":sc['score'][0]}  if len(sc['score'])>1 else {"score":()} for sc in scores]
        embeds = self.hf.embed_documents([doc for doc,ids in docs])

        optics_model = OPTICS(min_samples=2, xi=0.03) # Adjust parameters as needed
        optics_model.fit(embeds)
        labels = optics_model.labels_
        # hdbscan_model = hdbscan.HDBSCAN(min_cluster_size=2, min_samples=3) # Adjust parameters as needed
        # hdbscan_model.fit(embeds)
        # labels = hdbscan_model.labels_
        maxlabel = max(labels)
        merge_label = [ ]
        metadatas = []
        for label, score in zip(labels, scores):
            if  len(score['score']) > 1 and  score['score'][1] >= 2:
                merge_label.append(f"{topic}_{label+int(score['score'][0].metadata['maxlabel'])+2}") 
                metadatas.append({"label": f"{topic}_{label+int(score['score'][0].metadata['maxlabel'])+2}","maxlabel":str(maxlabel)})
            elif len(score['score']) > 1 and  score['score'][1] < 2:
                merge_label.append(score['score'][0].metadata['label'])
                metadatas.append({"label": score['score'][0].metadata['label'],"maxlabel":str(maxlabel)})
            else:
                merge_label.append(label)
                metadatas.append({"label": f'{topic}_{str(label)}',"maxlabel":str(maxlabel)})
             
        # min is always -1, need to plus 2 to avoid label duplicated
        # metadatas = [ {"label": f"{topic}_{label+int(score['score'][0]['metadata']['max_label'])+2}","max_label":maxlabel} if 'score' in score and  score['score'][1] > 3 else {"label": score['score'][0]['metadata']['label'],"maxlabel":maxlabel}
        #     for label, score in zip(hdbscan_model.labels_, scores) ]
        for doc, mt, sc, emb, id in zip(documents,metadatas,scores,embeds,ids):
            doc.metadata = mt
            if len(sc['score']) > 1 and sc['score'][1] == 0:
                self.vectorstore.update_document(document_id = id, document = doc)
            else:
                self.vectorstore.add_documents(documents = [doc],ids = [id], embeddings = [emb])

        return merge_label

    
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

def initial_llm(mode):

    if mode == "chatgpt":
        # Use ChatGPT
        model = "gpt-3.5-turbo-0613"
        llm = ChatGPTAPI_unsafe.ChatGPT(token=os.environ.get("CHATGPT_TOKEN"), conversation=os.environ.get("CONVERSATION_ID_1"))
        llm2json = ChatGPTAPI_unsafe.ChatGPT(token=os.environ.get("CHATGPT_TOKEN"), conversation=os.environ.get("CONVERSATION_ID_2"))
    elif mode == "meta":
        #Use Meta LLama2
        llm = MetaChatAPI.MetaChat()
        llm2json = MetaChatAPI.MetaChat()


    prompt_template =  """{context}
    Check if the given news article a crime news or not. And extract the keywords in above news article
    Allowed format:
    {{
    "Keywords":[Keywords list],
    "Crime": "TRUE OR FALSE"
    }}
    Response ONLY in JSON format"""

    # PROMPT = PromptTemplate(
    #     template=prompt_template, input_variables=["context", "topic"]
    # )

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context"]
    )
    classify_chain = LLMChain(llm=llm, prompt=PROMPT)


    prompt_template =  """{context}
    Extract name, gender, age, position,works for, committed crime name, sentence, involved in news title, if the field is not mentioned fill it with N/A
    Make sure every information you display is found in the given news.
    Allowed format:
    {{
    "news_title": " Title of news",
    "suspects": [
    {{
    "Name": "perosn1 name in the news",
    "Gender": "Gender of the perosn in the news",
    "Age": "Age of the person in the news",
    "Position": "Job title of position of the person in the news in the news",
    "Works For": "Company or organization name in the news",
    "Committed Crime Name": "Crime that the person is possibly committed in the news",
    "Sentence": "Sentence that given by the judge in the news"
    }},
    {{
    "Name": "perosn2 name in the news",
    "Gender": "Gender of the perosn in the news",
    "Age": "Age of the person in the news",
    "Position": "Job title of position of the person in the news",
    "Works For": "Company or organization name in the news",
    "Committed Crime Name": "Crime that the person is possibly committed in the news",
    "Sentence": "Sentence that given by the judge in the news"
    }}
    ]
    }}
    Response ONLY in JSON format and the value should be its original text, DO NOT translate"""

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context"]
    )
    extract_chain = LLMChain(llm=llm2json, prompt=PROMPT)

    return extract_chain, classify_chain
# llm("hello, what is quantun computing?")
def classification(topic,topk,offset,llm):
    # labels = clustering(topic, topk=40)
    docs = query_watson_discovery(topic, topk=topk,offset=offset)
    for doc in docs["results"]:
        if 'Title' in doc:
            doc['title'] = doc['Title']

        if 'Content' in doc:
            doc['text'] = [doc['Content']]

    with open(cwd+'/output/classified_news.json') as f:
        preload  = json.load(f)
    # For classifying news (Crime or not) and extract the keywords
    contexts = [('Title: ' + doc['title'] +'\n Content: ' + doc['text'][0], doc['document_id'] )for doc in docs["results"] if 'title' in doc and doc['document_id'] not in preload]
    
    outputs_classfied = { id:{"Crime":llm.apply([{"context":context}])[0]['text'], "context":context} for context,id in contexts}
    outputs_classfied = {**outputs_classfied, **preload}


    with open(cwd+'/output/classified_news.json','w') as f:
        json.dump(outputs_classfied, f,ensure_ascii=False)
    
    # with open(cwd+'/output/classified_news.json') as f:
    #     outputs_classfied = json.load(f)
    # Cluster the crime news
    classified = [{"keywords":','.join(eval(outputs_classfied[out]["Crime"])['Keywords']),"context":outputs_classfied[out]["context"],'id':out} for out in outputs_classfied if outputs_classfied[out]['Crime']!="" and "TRUE" in eval(outputs_classfied[out]['Crime'])['Crime'].upper()]
    return classified
def cluster_news(initial, query, classified,llm):
    """
    initial[bool]: intial the db
    """
    # Cluster the crime news
    vectordb = vector_store()
    # vectordb.vectorstore.__query_collection(query_texts=[])
    with open(cwd+'/output/extracted_news.json') as f:
        extract_json = json.load(f)
    if initial:
        clustered = vectordb.clustering(docs = [(cls["keywords"],cls['id']) for cls in classified if cls['id'] ],topic = query)
    else:
        new_class = [(cls["keywords"],cls['id']) for cls in classified if cls['id'] not in extract_json]
        if new_class:
            clustered = vectordb.add2cluster(docs = new_class, topic = query)
        else:
            clustered = {}
    # not_classified = [','.join(eval(out["Crime"])['Keywords']) for out in outputs_classfied ]
    # clustered_not_cls = clustering(not_classified)

    # For generate JSON
    if clustered:
        outputs_json = {clsf['id']:{"out":llm.apply([{"context":clsf["context"]}])[0]['text'],"content":clsf["context"],"label":str(clst)} for clsf, clst in zip(classified,clustered) if clsf['id'] not in extract_json }
        clustered_json = {**extract_json,**outputs_json}
    else:
        clustered_json = extract_json
    with open(cwd+'/output/extracted_news.json','w') as f:
        json.dump(clustered_json, f, ensure_ascii=False)
    return clustered_json
    
if __name__ == "__main__":
    with open(cwd+'/data/namelist.txt') as f:
        namelist = f.readlines()[4:6]
    for query in namelist:
        print(query)
        extraction_chain, classfication_chain = initial_llm(mode="chatgpt")
        output_classified = classification(query, 30, 0, classfication_chain)
        output_clusterd = cluster_news(initial = False, classified = output_classified, query = query, llm = extraction_chain)
        news_json =[]
        for cls in output_clusterd:
            jsonout = eval(output_clusterd[cls]['out'])
            pattern = r'Content: (.+)'
            content = output_clusterd[cls]['content']
            # Use re.search to find the first occurrence of the pattern.
            match = re.search(pattern, content)
            jsonout['News'] = match.group(1)
            jsonout['Group'] = output_clusterd[cls]['label']
            news_json.append(jsonout)
        time.sleep(180)
    # with open(cwd+'/output/extracted_news.json') as f:
    #     extract_json = json.load(f)
    #     news_json=[]
    # for cls in extract_json:
    #     jsonout = eval(extract_json[cls]['out'])
    #     pattern = r'Content: (.+)'
    #     content = extract_json[cls]['content']
    #     # Use re.search to find the first occurrence of the pattern.
    #     match = re.search(pattern, content)
    #     jsonout['News'] = match.group(1)
    #     jsonout['Group'] = extract_json[cls]['label']
    #     news_json.append(jsonout)
    delete_nodes()
    for data in news_json:
        for row in data['suspects']:
            row["Involved in News Title"] = data['news_title']
            row['News Content'] = data["News"]
            row['Group'] = data['Group']
            
            create_node_and_relationship(row)
# prompt_template =  """請依照Context中提供的多篇文章擷取資訊總結並整理出CSV 結果：
# Conext: {text}
# Summarized Output:"""

# PROMPT = PromptTemplate(
#     template=prompt_template, input_variables=["text"]
# )


# summarize_chain = LLMChain(llm=llm, prompt=PROMPT)
# summarize_chain.apply([{"text":news_info}])