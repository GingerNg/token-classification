from confs import col, mock_col, noises
import random

# category
cate1s = ['O'],
cate2s = ['prov', 'city', 'district', 'town']
cate3s = ['community', 'poi', 'road', 'devzone', 'village_group', 'subpoi']
cate4s = ['roadno', 'houseno', 'intersection', 'assist', 'cellno', 'floorno', 'distance']

cates = ['O', 'prov', 'city', 'district', 'town', 'community',
         'poi', 'road', 'roadno', 'subpoi', 'devzone', 'houseno',
         'intersection', 'assist', 'cellno', 'floorno', 'distance',
         'village_group']
cate_words = {}
for cate in cates:
    cate_words[cate] = []
for item in col.find({}):
    entities = item['entity']
    tags = item['tag']
    for i in range(len(entities)):
        entity = entities[i]
        cate = tags[i]
        cate_words[cate].append(entity)


def generate_data1():
    # 直接随机挑选ent，生成数据
    entities = []
    tags = []
    for _ in range(random.randint(1, 10)):
        ent, tag = randon_ent()
        entities.append(ent)
        tags.append(tag)
    mock_col.insert({
        "entity": entities,
        'tag': tags
    })


def randon_ent():
    # 随机选择一个实体
    ind = random.randint(0, len(cates)-1)
    tag = cates[ind]
    words = cate_words[tag]
    ent = words[random.randint(0, len(words)-1)]
    return ent, tag


def insert_sth(entities, tags, ent, tag):
    ind = random.randint(0, len(entities)-1)
    return entities[0:ind] + [ent] + entities[ind:], tags[0:ind] + [tag] + tags[ind:]


def generate_data2():
    # 对已有train-data，随机插入ent
    for item in col.find():
        entities = item['entity']
        tags = item['tag']
        if random.randint(0, 1) == 1:
            ent = noises[random.randint(0, len(noises)-1)]
            tag = "O"
        else:
            ent, tag = randon_ent()
        entities, tags = insert_sth(entities, tags, ent, tag)
        mock_col.insert({
            "entity": entities,
            'tag': tags
        })

# def generate_data3():

for _ in range(500):
    generate_data1()
generate_data2()