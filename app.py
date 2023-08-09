import os
import json
from flask import Flask, request, jsonify
import requests
import LLM.watsonx_example.due_diligence as due_diligence
import LLM.IBMChatAPI as ibm_ref

app = Flask(__name__)
watsonx_token= os.environ.get("WATSONX_TOKEN") 

def graphdb_search_response(name, job, company):
    neo4j_headers = {
        'Accept': 'application/json;charset=UTF-8',
        'X-Stream': 'true',
        'Authorization': 'Basic bmVvNGo6cGFzc3dvcmQ=',
        'Content-Type': 'application/json',
    }
    query = f"""MATCH (person:Person) - [r:INVOLVED_IN] -> (news:News)
    where person.name = '{name}'
    OPTIONAL MATCH (person) - [rec:IS_RECOGNIZED] -> (id:Identifier)
    OPTIONAL MATCH (person) - [work:WORKS_FOR] -> (org:Organization)
    where id.position =~ '.*{job}' or org.name =~ '{company}.*'
    return distinct news.title, news.content"""
    json_data = {
    'statements': [
        {
            'statement': query,
        },
    ],
    }
    result = requests.post('http://128.168.134.242:3074/db/neo4j/tx', headers=neo4j_headers, json=json_data)
    title = result.json()['results'][0]['data'][0]['row'][0]
    content = result.json()['results'][0]['data'][0]['row'][1]
    response = {"name":name, "title":title, "content":content}
    return response

def watsonx_response(name, title, content):
    watsonx_headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
        'Authorization': 'Bearer {watsonx_token}',
    }
    params = {
        'version': '2023-05-29',
    }
    template = due_diligence.template_1
    applicant_content = f"""
        Extract {name}'s below information from below content, committed crime name, sentence, involved in news title, if the field is not mentioned fill it with N/A Make sure every information you display is found in the given news. Allowed format: [ "news_title", " Title of news",  "Name", "person name in the news", "Committed Crime Name", "Crime that the person is possibly committed in the news", "Sentence", "Sentence that given by the judge in the news"] Response ONLY in list format and the value should be its original text, DO NOT translate
            Title: {title}
            Content: {content}
    """
    json_data = {
        "model_id": "ibm/mpt-7b-instruct2",
        'input': f"""{template}
        Transcript:{applicant_content}
        Summary:
        """,
        "parameters": {
            "decoding_method": "greedy",
            "max_new_tokens": 200,
            "min_new_tokens": 50,
            "stop_sequences": [],
            "repetition_penalty": 1,
        },
        "project_id": "5dec0b1d-c146-4422-830d-4185d36e5223",
    }
    response = requests.post(
        'https://us-south.ml.cloud.ibm.com/ml/v1-beta/generation/text',
        params=params,
        headers=watsonx_headers,
        json=json_data,
    )
    print(response.json())
    return jsonify(response)

@app.route('/due_diligence', methods=['POST'])
def due_diligence_response():
    data = request.get_json()
    name = data['name']
    job = data['job']
    company = data['company']
    graph_response = graphdb_search_response(name, job, company)
    name = graph_response['name']
    title = graph_response['title']
    content = graph_response['content']
    result = ibm_ref.watsonx_ref_model(name, title, content)
    return result


if __name__ == '__main__':
    app.run(host='0.0.0.0')