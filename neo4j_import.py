
from py2neo import Graph, Node, Relationship
import os
import neo4j
from py2neo.matching import *
import json
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
    person_id_relation = Relationship(person_node, "IS_RECOGNIZED", identifier_node) 
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
# with open('output/extracted_news.json') as f:
#   news_json = json.load(f)
# for data in news_json:
#     for row in data['suspects']:
#         row["Involved in News Title"] = data['news_title']
#         row['News Content'] = data["News"]
#         create_node_and_relationship(row)
