import json
format1 = """Extract name, gender, age, position,works for, committed crime name, sentence, involved in news title, if the field is not mentioned fill it with N/A
    Make sure every information you display is found in the given news.
    Allowed format:
    {
    "news_title": " Title of news",
    "suspects": [
    {
    "Name": "perosn1 name in the news",
    "Gender": "Gender of the perosn in the news",
    "Age": "Age of the person in the news",
    "Position": "Job title of position of the person in the news in the news",
    "Works For": "Company or organization name in the news",
    "Committed Crime Name": "Crime that the person is possibly committed in the news",
    "Sentence": "Sentence that given by the judge in the news"
    },
    {
    "Name": "perosn2 name in the news",
    "Gender": "Gender of the perosn in the news",
    "Age": "Age of the person in the news",
    "Position": "Job title of position of the person in the news",
    "Works For": "Company or organization name in the news",
    "Committed Crime Name": "Crime that the person is possibly committed in the news",
    "Sentence": "Sentence that given by the judge in the news"
    }
    ]
    }
    Response ONLY in JSON format and the value should be its original text, DO NOT translate"""

format2 = """
    Check if the given news article a crime news or not. And extract the keywords in above news article
    Allowed format:
    {
    "Keywords":[Keywords list],
    "Crime": "TRUE OR FALSE"
    }
    Response ONLY in JSON format"""
with open("/workspace/answerfromKB/WatsonxChallenge/output/extracted_news.json") as f:
    extract = json.load(f)
with open("/workspace/answerfromKB/WatsonxChallenge/output/classified_news.json") as f:
    clssify = json.load(f)
with open("/workspace/answerfromKB/WatsonxChallenge/output/train2.txt",'w') as f:

    # for i in extract:
    #     f.write(f"\"<s>Human:{extract[i]['content']}\n{format1}\n</s><s>Assistant:{extract[i]['out']}</s>\"")
    #     f.write("\n")


    for i in clssify:
        f.write(f"\"<s>Human:{clssify[i]['context']}\n{format2}\n</s><s>Assistant:{clssify[i]['Crime']}</s>\"")
        f.write("\n")
    
