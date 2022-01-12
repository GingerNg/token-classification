

def parse_line(line):
    ws = []
    ts = []
    cs = line.split('\n')
    for c in cs:
        print(c)
        if " " in c:
            w = c.split(" ")[0]
            ws.append(w)
            t = c.split(" ")[1]
            ts.append(t)
    return ws, ts


def BIEO2Word(ws, ts):
    entities = []
    tags = []
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
    return entities, tags
