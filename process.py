from pymongo import MongoClient
from bson import ObjectId

client = MongoClient('localhost', 27017)

col_name = 'train'

db = client["CCKS2021中文NLP地址要素解析"]
col = db[col_name]

train_bak_col = db["train_bak"]
dev_col = db["dev"]

dev_lines = []

# for item in dev_col.find():
#     dev_lines.append("".join(item['entity']))

for item in col.find({'_id': {'$lt': ObjectId("60dc00000000000000000000")}}):
    line = "".join(item['entity'])
    if line not in dev_lines:
        train_bak_col.insert(item)
