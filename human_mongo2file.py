from confs import col, mock_col

Ls = []

def mongo2file(col, f):
    for item in col.find():
        entities = item['entity']
        tags = item['tag']
        Ls.append(len("".join(entities)))
        for i in range(len(entities)):
            es = list(entities[i])
            if tags[i] != "O":
                if len(entities[i]) == 1:
                    ts = ["S-" + tags[i]]
                elif len(entities[i]) == 2:
                    ts = ["B-" + tags[i], "E-" + tags[i]]
                else:
                    ts = ["B-" + tags[i]] + (["I-" + tags[i]] * (len(entities[i]) - 2)) + ["E-" + tags[i]]
            else:
                ts = ["O"] * len(entities[i])
            # print(es, ts)
            for j in range(len(es)):
                f.write(es[j])
                f.write(" ")
                f.write(ts[j])
                f.write("\n")
        f.write('\n')


with open("data/CCKS2021中文NLP地址要素解析/train_human.conll", 'w') as f:
    mongo2file(col, f)
    mongo2file(mock_col, f)
# print(set(Ls))

