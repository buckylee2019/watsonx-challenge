
from py2neo import Graph, Node, Relationship
import os
import neo4j
from py2neo.matching import *

# Connect to your Neo4j database
uri = os.environ.get("NEO4J_URL")  # Replace with your Neo4j URI
username = os.environ.get("NEO4J_USER")     # Replace with your Neo4j username
password = os.environ.get("NEO4J_PWD")     # Replace with your Neo4j password

graph = Graph(uri, auth=(username, password))
nodes = NodeMatcher(graph)

def create_node_and_relationship(row):
    person_node = Node("Person", name=row["Name"])

    identifier_node =  Node("Identifier", name=row["Name"]+"-識別資訊", gender=row["Gender"], age=row["Age"], position=row["Position"])
    org_node = Node("Organization", name=row["Works For"])
    news_node = Node("News", title=row["Involved in News Title"], content=row["News Content"])
    graph.merge(person_node, "Person", "name")
    graph.merge(org_node, "Organization", "name")
    graph.merge(news_node, "News", "title")
    person_id_relation = Relationship(person_node, "IS RECOGNIZED", identifier_node) 
    graph.create(person_id_relation)
    if row["Committed Crime Name"]!="" and row["Committed Crime Name"]!="N/A":
        
      crime_node = Node("Crime", name=row["Committed Crime Name"])
      graph.merge(crime_node, "Crime","name")
      committed_crime_relation = Relationship(identifier_node, "COMMITTED_CRIME", crime_node)      
      graph.create(committed_crime_relation)
      
    if row['Sentence'] != "" and row['Sentence'] != "N/A":
       
      sentence_node = Node("Sentence", name=row["Sentence"])
      graph.merge(sentence_node, "sentence", "name")
      sentence_relation = Relationship(identifier_node, "IS_SENTENCED", sentence_node)
      graph.create(sentence_relation)
    


    
    

    works_for_relation = Relationship(person_node, "WORKS_FOR", org_node)
    involved_in_news_relation = Relationship(person_node, "INVOLVED_IN", news_node)
    

    graph.create(works_for_relation)
    graph.create(involved_in_news_relation)
    
    
# # Sample CSV data (replace this with your actual data)
# data ={
#   "news_title": "HTC員工洩漏設計圖",
#   "suspects": [
#     {
#       "Name": "簡志霖",
#       "Gender": "",
#       "Age": "",
#       "Position": "宏達電設計部副總",
#       "Works For": "宏達電",
#       "Committed Crime Name": "違反證交法、背信罪、營業秘密法",
#       "Sentence": "羈押中，交保金額：八百萬元"
#     },
#     {
#       "Name": "吳建宏",
#       "Gender": "",
#       "Age": "",
#       "Position": "宏達電設計部處長",
#       "Works For": "宏達電",
#       "Committed Crime Name": "違反證交法、背信罪、營業秘密法",
#       "Sentence": "羈押中，交保金額：五百萬元"
#     },
#     {
#       "Name": "黃國清",
#       "Gender": "",
#       "Age": "",
#       "Position": "宏達電工業設計部資深經理",
#       "Works For": "宏達電",
#       "Committed Crime Name": "",
#       "Sentence": ""
#     },
#     {
#       "Name": "黃弘毅",
#       "Gender": "",
#       "Age": "",
#       "Position": "宏達電工業設計部資深經理",
#       "Works For": "宏達電",
#       "Committed Crime Name": "",
#       "Sentence": ""
#     },
#     {
#       "Name": "洪琮鎰",
#       "Gender": "",
#       "Age": "",
#       "Position": "宏達電製造設計部經理",
#       "Works For": "宏達電",
#       "Committed Crime Name": "",
#       "Sentence": ""
#     },
#     {
#       "Name": "陳枻佐",
#       "Gender": "",
#       "Age": "",
#       "Position": "宏達電員工",
#       "Works For": "宏達電",
#       "Committed Crime Name": "",
#       "Sentence": ""
#     },
#     {
#       "Name": "張俊宜",
#       "Gender": "",
#       "Age": "",
#       "Position": "供應廠商",
#       "Works For": "",
#       "Committed Crime Name": "",
#       "Sentence": ""
#     },
#     {
#       "Name": "陳榮元",
#       "Gender": "",
#       "Age": "",
#       "Position": "供應廠商",
#       "Works For": "",
#       "Committed Crime Name": "",
#       "Sentence": ""
#     },
#     {
#       "Name": "陳忠貴",
#       "Gender": "",
#       "Age": "",
#       "Position": "供應廠商",
#       "Works For": "",
#       "Committed Crime Name": "",
#       "Sentence": ""
#     }
#   ]
# }
# csv_data = [
#     {
#         "Name": "簡志霖",
#         "Gender": "Male",
#         "Age": "",
#         "Position": "設計部副總",
#         "Works For": "宏達電",
#         "Committed Crime Name": "違反證交法、背信罪、營業秘密法",
#         "Sentence": "羈押中",
#         "Involved in News Title": "HTC員工洩漏設計圖",
#         "News Content":"〔自由時報記者林俊宏、王憶紅／台北報導〕台北地檢署認定宏達電未發表的圖形介面遭竊取，當時的宏達電設計部副總簡志霖、設計部處長吳建宏私下拉攏員工在外另成立曉玉公司，洩漏宏達電尚未發表操作介面的ICON圖形設計，赴中國北京簡報，另多次浮報款項及收受回扣達三三五六萬六千元，昨依違反證交法、背信罪、營業秘密法，將簡志霖等九人起訴。簡志霖、吳建宏重金交保全案昨移審台北地院，法官裁定羈押中的簡志霖和吳建宏分別以八百萬元和五百萬元交保，均限制住居、出境、出海，兩人離去時不發一語。昨被起訴的被告，還包括宏達電工業設計部資深經理黃國清、黃弘毅、製造設計部經理洪琮鎰、員工陳枻佐及供應廠商張俊宜、陳榮元及陳忠貴等人。宏達電表示，注重員工的道德與廉潔操守，若有任何違反情形，必定毋枉毋縱。至於原預計明年二月發表的HTC Sence 6.0，是否因圖形介面遭竊而延後推出？宏達電則不予評論。"
#     },
#     # Add more rows as needed
# ]

# for row in data['suspects']:
#     row["Involved in News Title"] = data['news_title']
#     row['News Content'] = "〔自由時報記者林俊宏、王憶紅／台北報導〕台北地檢署認定宏達電未發表的圖形介面遭竊取，當時的宏達電設計部副總簡志霖、設計部處長吳建宏私下拉攏員工在外另成立曉玉公司，洩漏宏達電尚未發表操作介面的ICON圖形設計，赴中國北京簡報，另多次浮報款項及收受回扣達三三五六萬六千元，昨依違反證交法、背信罪、營業秘密法，將簡志霖等九人起訴。簡志霖、吳建宏重金交保全案昨移審台北地院，法官裁定羈押中的簡志霖和吳建宏分別以八百萬元和五百萬元交保，均限制住居、出境、出海，兩人離去時不發一語。昨被起訴的被告，還包括宏達電工業設計部資深經理黃國清、黃弘毅、製造設計部經理洪琮鎰、員工陳枻佐及供應廠商張俊宜、陳榮元及陳忠貴等人。宏達電表示，注重員工的道德與廉潔操守，若有任何違反情形，必定毋枉毋縱。至於原預計明年二月發表的HTC Sence 6.0，是否因圖形介面遭竊而延後推出？宏達電則不予評論。"
    
#     create_node_and_relationship(row)
