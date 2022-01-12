from tqdm import tqdm
from confs import col, dev_col

tags = []
labels = ['B', "I", "E", "O", 'S']
categories = {}
texts = []
text = []
with open("data/CCKS2021中文NLP地址要素解析/final_test.conll") as f:
    data = f.read()
    lines = data.split("\n\n")
    for line in tqdm(lines):
        entities = []
        tags = []
        ws = []
        ts = []
        cs = line.split('\n')
        for c in cs:
            if " " in c:
                w = c.split(" ")[0]
                ws.append(w)
                t = c.split(" ")[1]
                ts.append(t)
        entity = ""
        for i in range(len(ws)):
            if ts[i].startswith("E") or ts[i].startswith("S"):
                entity += ws[i]
                entities.append(entity)
                entity = ""
                tags.append(ts[i].split("-")[1])
            elif ts[i].startswith("B"):
                if entity != "":
                    entities.append(entity)
                    entity = ""
                entity += ws[i]
                # tags.append(ts[i].split("-")[1])
            elif ts[i].startswith("O"):
                entity += ws[i]
                # entities.append(entity)
                # entity = ""
                if len(tags) == len(entities):
                    tags.append(ts[i])
            else:
                entity += ws[i]
        if entity != "":
            entities.append(entity)
        dev_col.insert({
            "entity": entities,
            'tag': tags
        })
        # print(entities, tags)