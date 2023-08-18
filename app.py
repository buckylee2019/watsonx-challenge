#coding=utf-8
import os
import json
from flask import Flask, request, jsonify
import requests
import LLM.watsonx_example.due_diligence as due_diligence
import LLM.IBMChatAPI as ibm_ref
from LLM import ChatGPTAPI_unsafe  # FREE GOOGLE BARD API
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.sql_database import SQLDatabase
import datetime
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
watsonx_token= os.environ.get("WATSONX_TOKEN","aaa") 
llm = ChatGPTAPI_unsafe.ChatGPT(token=os.environ.get("CHATGPT_TOKEN"), conversation=os.environ.get("CONVERSATION_ID_1"))

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
    # 用假資料代替GraphDB搜尋
    # graph_response = graphdb_search_response(name, job, company)
    # name = graph_response['name']
    # title = graph_response['title']
    # content = graph_response['content']
    # watson_llm = ibm_ref.watsonx(os.environ.get('GENAI_KEY'),os.environ.get('GENAI_API'))
    # result = watson_llm.watsonx_ref_model(name, title, content)
    result = jsonify({
        "news_title": "掏空案創單起案件最高賠償",
        "name": "王小明",
        "committed_crime_name": "掏空",
        "sentence": "賠償1000億多元"    
    })
    return result
@app.route('/get_account_status', methods=['POST'])
def account_status():
    db = SQLDatabase.from_uri(os.environ.get("MYSQL_CONN","None"))
    data = request.get_json()
    # data = {"name":"孫道存"}
    name = data['name']
    account = db.run(f"select * from account_status where customer_name = \"{name}\"")
    # print(eval(account))
    if account == "":
        # account = [(5, '孫道存', 'Rejected', 'Pending', '涉及金融犯罪相關新聞, 需進一步審查', '2023-08-10')]
        return jsonify({"error":"搜尋名稱不存在！"})
    id, Name, _, result, reason, dt =  eval(account)[0]
    return jsonify({
        "Name": Name,
        "Status":result,
        "Reason":reason,
        "UpdateDate" : dt
    })

@app.route('/get_abnormal_list', methods=['POST'])
def abnormal_list():
    db = SQLDatabase.from_uri(os.environ.get("MYSQL_CONN","None"))
    data = request.get_json()
    
    dateoftrans = data['date']
    kyc = db.run(f"select * from kyc_info")
    print(kyc)
    if kyc == "":
        kyc = [
    {"Transaction ID": "TX12345", "Date and Time": "2023-08-01 14:30:00", "Account Number": "A123456", "Transaction Type": "Withdrawal", "Transaction Amount": 10000, "Counterparty": "-", "Location": "Paris, France", "Transaction Channel": "ATM", "Purpose or Description": "Large cash withdrawal", "Risk Score": 8, "Flag or Alert": True, "Review Status": "Pending", "Associated Accounts": "-", "Source of Funds": "Savings", "Transaction Patterns": "Unusual amount", "Notes or Comments": "Customer reported travel plans", "Investigation Status": "Ongoing"},
    {"Transaction ID": "TX67890", "Date and Time": "2023-07-15 09:45:00", "Account Number": "B987654", "Transaction Type": "Deposit", "Transaction Amount": 50, "Counterparty": "Friend's Account", "Location": "New York, USA", "Transaction Channel": "Online Banking", "Purpose or Description": "Repaying borrowed money", "Risk Score": 5, "Flag or Alert": False, "Review Status": "Cleared", "Associated Accounts": "-", "Source of Funds": "Personal funds", "Transaction Patterns": "Small deposits", "Notes or Comments": "-", "Investigation Status": "N/A"},
    {"Transaction ID": "TX24680", "Date and Time": "2023-06-10 11:00:00", "Account Number": "C543210", "Transaction Type": "Transfer", "Transaction Amount": 1000, "Counterparty": "Account D123456", "Location": "Los Angeles, USA", "Transaction Channel": "Mobile App", "Purpose or Description": "Regular monthly payment", "Risk Score": 3, "Flag or Alert": False, "Review Status": "Cleared", "Associated Accounts": "Account D123456", "Source of Funds": "Salary", "Transaction Patterns": "Regular payment", "Notes or Comments": "-", "Investigation Status": "N/A"},
    {"Transaction ID": "TX98765", "Date and Time": "2023-08-05 18:15:00", "Account Number": "E135792", "Transaction Type": "Purchase", "Transaction Amount": 2000, "Counterparty": "Online Retailer", "Location": "Tokyo, Japan", "Transaction Channel": "Credit Card", "Purpose or Description": "Electronics purchase", "Risk Score": 7, "Flag or Alert": True, "Review Status": "Pending", "Associated Accounts": "-", "Source of Funds": "Credit", "Transaction Patterns": "Unusual location and amount", "Notes or Comments": "-", "Investigation Status": "Ongoing"},
    {"Transaction ID": "TX54321", "Date and Time": "2023-07-03 23:00:00", "Account Number": "F246813", "Transaction Type": "Transfer", "Transaction Amount": 1500, "Counterparty": "Account G357159", "Location": "Miami, USA", "Transaction Channel": "Online Banking", "Purpose or Description": "Inter-account transfer", "Risk Score": 4, "Flag or Alert": False, "Review Status": "Cleared", "Associated Accounts": "Account G357159", "Source of Funds": "Savings", "Transaction Patterns": "Regular inter-account transfer", "Notes or Comments": "-", "Investigation Status": "N/A"}
]
        return jsonify({
        "Response": llm(f"使用繁體中文總結這份列表的異常交易行為, 並列出異常帳戶及交易異常行為:{str(kyc)}")
    })
    
    # id, Name, result, _, reason, dt =  kyc[0]
    return jsonify({
        "Response": f"以下是 {dateoftrans} 的異常交易行為的總結，以及列出的異常帳戶和交易異常行為：\n\n總結：\n這份列表中包含了幾筆異常交易行為，涵蓋了不同類型的交易，包括提款、存款、轉帳和購物等。這些異常交易行為涉及金額、地點、交易渠道等多方面的不尋常特徵，有些已經觸發警示標誌，有些正在審查中，並且可能涉及潛在的風險。一些交易已經被解除警示，但其他一些交易仍在調查中。\n\n異常帳戶及交易異常行為：\n1. 帳戶 A123456\n   - 交易ID：TX12345\n   - 交易類型：提款\n   - 交易金額：10000\n   - 地點：巴黎，法國\n   - 交易渠道：ATM\n   - 交易特徵：大額現金提款\n   - 風險分數：8\n   - 警示標誌：是\n   - 審查狀態：待審查\n   - 交易模式：金額不尋常\n   - 註解：客戶報告的旅行計劃\n   - 調查狀態：進行中\n\n2. 帳戶 E135792\n   - 交易ID：TX98765\n   - 交易類型：購物\n   - 交易金額：2000\n   - 地點：東京，日本\n   - 交易渠道：信用卡\n   - 交易特徵：不尋常的地點和金額\n   - 風險分數：7\n   - 警示標誌：是\n   - 審查狀態：待審查\n   - 交易模式：金額和地點不尋常\n   - 註解：無\n   - 調查狀態：進行中\n\n以上帳戶的交易行為顯示出金額、地點或其他特徵的不尋常情況，可能涉及風險或潛在的異常交易。這些帳戶的交易正在審查中，以確定是否存在任何不當行為。"
    })

@app.route('/get_risk_list', methods=['GET'])
def risk_list():
    fakelist = """
    風險名單 1：
        姓名：李小明
        國家：中國
        原因：涉嫌進行非法的金融詐騙活動

    風險名單 2：
        姓名：Emily Johnson
        國家：美國
        原因：涉嫌參與跨國洗錢計劃

    風險名單 3：
        姓名：Rajesh Patel
        國家：印度
        原因：涉嫌違反國際制裁，與叙利亞有不當商業交易

    風險名單 4：
        姓名：Anna Petrov
        國家：俄羅斯
        原因：涉嫌參與非法的武器買賣交易

    風險名單 5：
        姓名：Mohammed Ali
        國家：沙特阿拉伯
        原因：涉嫌支持恐怖主義活動，資助極端組織"""
    # id, Name, result, _, reason, dt =  kyc[0]
    return jsonify({
        "Response":fakelist})

if __name__ == '__main__':
   
    app.run(host='0.0.0.0',port=5000)