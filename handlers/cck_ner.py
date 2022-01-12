import ner_model

import cck_process_data_bert as process_data
import numpy as np
from utils import tag_utils

dataset_name = 'ccks2021-api'

model, (vocab, chunk_tags) = ner_model.create_model(train=False, dataset_name=dataset_name)
model.load_weights('{}/crf_{}.h5'.format(ner_model.model_dir, dataset_name))
# model.load_weights('model/bilstm_softmax.h5')

def infer(predict_text):
    # predict_text = '中华人民共和国国务院总理周恩来在外交部长陈毅和王东的陪同下，连续访问了埃塞俄比亚等非洲10国以及阿尔巴尼亚'
    str, length = process_data.process_data(predict_text, vocab)
    # print(str, length)

    # raw = model.predict(str)[0][-length:]
    raw = model.predict(str)[0][1:length+1]
    # print(raw)
    # result = [np.argmax(row) for row in raw]
    result = raw
    # print(result)

    # print(result_tags)
    result_tags = [chunk_tags[i] for i in result]
    if len(predict_text) > len(result_tags):
        result_tags += ["O"] * (len(predict_text) - len(result_tags))

    ws = list(predict_text)
    ts = result_tags
    res = tag_utils.BIEO2Word(ws, ts)

    return res
