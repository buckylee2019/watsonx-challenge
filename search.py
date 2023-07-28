from langchain.chat_models import ChatOpenAI
from langchain.chains import GraphCypherQAChain
from langchain.graphs import Neo4jGraph
import os
from dotenv import load_dotenv
from LLM import ChatGPTAPI_unsafe  # FREE GOOGLE BARD API
from langchain.prompts import PromptTemplate
from LLM import MetaChatAPI  # FREE GOOGLE BARD API
# Connect to your Neo4j database
load_dotenv()
graph = Neo4jGraph(
    url=os.environ.get("NEO4J_URL"), 
    username= os.environ.get("NEO4J_USER"), 
    password=os.environ.get("NEO4J_PWD") )
     

CYPHER_RECOMMENDATION_TEMPLATE = """Task:Generate Cypher statement to query a graph database.
Instructions:
Use only the provided relationship types and properties in the schema.
Do not use any other relationship types or properties that are not provided.
Please check if the query is in valid cypher.
Schema:
{schema}
Cypher examples:
# How many streamers are from Norway?
MATCH (s:Stream)-[:`HAS_LANGUAGE`]->(:Language {{name: 'no'}})
RETURN count(s) AS streamers
# Which streamers do you recommend if I like kimdoe?
MATCH (s:Stream)
WHERE s.name = "kimdoe"
WITH collect(s) AS sourceNodes
CALL gds.pageRank.stream("shared-audience", 
  {{sourceNodes:sourceNodes, relationshipTypes:['SHARED_AUDIENCE'], 
    nodeLabels:['Stream']}})
YIELD nodeId, score
WITH gds.util.asNode(nodeId) AS node, score
WHERE NOT node in sourceNodes
RETURN node.name AS streamer, score
ORDER BY score DESC LIMIT 3

Note: Do not include any explanations or apologies in your responses.
Do not respond to any questions that might ask anything else than for you to construct a Cypher statement.
Do not include any text except the generated Cypher statement.

The question is:
{question}"""
CYPHER_RECOMMENDATION_PROMPT = PromptTemplate(
    input_variables=["schema", "question"], template=CYPHER_RECOMMENDATION_TEMPLATE
)
llm = MetaChatAPI.MetaChat()
# llm= ChatGPTAPI_unsafe.ChatGPT(token=os.environ.get("CHATGPT_TOKEN"), conversation=os.environ.get("CONVERSATION_ID_1"))

chain = GraphCypherQAChain.from_llm(
    llm, graph=graph, verbose=True,cypher_prompt=CYPHER_RECOMMENDATION_PROMPT
)



print(chain.run("""
列出吳建宏是否涉及犯罪及相關人員識別資訊如職位年紀性別
"""))
     
