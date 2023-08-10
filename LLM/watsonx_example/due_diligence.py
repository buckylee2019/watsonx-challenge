template_1 = """
Write a short summary for the news.
Transcript: Extract 呂財益's below information from below content, committed crime name, sentence, involved in news title, if the field is not mentioned fill it with N/A Make sure every information you display is found in the given news. Allowed format: ["news_title", " Title of news",  "Name", "person name in the news", "Committed Crime Name", "Crime that the person is possibly committed in the news", "Sentence", "Sentence that given by the judge in the news"] Response ONLY in LIST format and the value should be its original text, DO NOT translate
    Title: 關稅總局前副總局長呂財益涉貪污一審判刑7年半 
    Content: 台北地院審理7年前關稅總局集體貪污案，認定前副總局長呂財益接受業者關說人事升遷關員收賄及上阿公店飲宴，再放行違禁品並幫助業者逃漏稅，今依貪污罪判呂財益7年6月，另有詐欺、洩密罪併判7月可易科罰金，第一線關員林東瑩等人各處3年10月至14年不等有期徒刑。可上訴。判決指出業者全昱報關行負責人詹美華、靠行業者賴欽聰、立委李復興前助理張勝泰等人，明知從中國進口的石材、磁磚或生鮮蓮藕等貨物，均屬禁止輸入物品，卻從2009年12月起密集行賄海關，先按櫃向進口商收取1萬6000元到20萬元不等的公關費，再以每櫃1000元到5萬元不等價碼行賄關員。檢方駐守自主管理貨櫃站的關員陳玉珠不但配合業者放水，還插股50萬元。此外，關員鄭張達協助業者不實申報貨物進口價格以逃稅，還提供其他業者的報關單據。檢方說，林東瑩原將調職，業者擔心影響業務，透過賴欽聰找上張勝泰運作，請呂財益幫忙將林留在原單位，並以30萬元行賄呂，呂收錢後又退款，有賣官之虞。
Summary:
    {
    "news_title": "關稅總局前副總長呂財益涉貪污一審判刑7年"
    "Name": "呂財益",
    "Committed Crime Name": "收取鉅額回扣、無法理底為主張的不當營運及非法付款名口有量、侵害中小企業資產組別，陷入 financially unstable condition 的醜聞",
    "Sentence": "7年半徒刑"
    }
"""