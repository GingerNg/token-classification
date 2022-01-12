from pymongo import MongoClient

client = MongoClient('localhost', 27017)

col_name = 'train'

db = client["CCKS2021中文NLP地址要素解析"]
col = db[col_name]

mock_col = db["train_mock"]

dev_col = db['dev']

test_col = db['test']

noises = ['（', '）', '(', ')', "A", '000000', '00000000000',
          'AAAAAA', 'AAAAAAAAAA', 'AAAA', 'AAAAA',
          '，'
          ]

# category
cate1s = ['O'],
cate2s = ['prov', 'city', 'district', 'town']
cate3s = ['community', 'poi', 'road', 'devzone', 'village_group', 'subpoi']
cate4s = ['roadno', 'houseno', 'intersection',
          'assist', 'cellno', 'floorno', 'distance']


# 05 商铺 、803 房间 、304 寝室、房、东户、西户、中户、门市
# detail=12-3-1001
