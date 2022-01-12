from tqdm import tqdm
from confs import test_col

tags = []
labels = ['B', "I", "E", "O", 'S']
categories = {}
texts = []
text = []
col_delimter = ""  # 分隔符为\u0001
with open("data/CCKS2021中文NLP地址要素解析/final_test.txt") as f:
    data = f.read()
    lines = data.split("\n")
    for line in lines:
        number = line.split(col_delimter)[0]
        text = line.split(col_delimter)[1]
        test_col.insert(
            {
                "text": text,
                'No': number
            }
        )
