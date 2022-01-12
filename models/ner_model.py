# 通过import实现不同的行为
# import models.hotel_process_data as process_data  # 根据模型输入和数据集处理data
import models.cck_process_data as process_data
# import models.cck_process_data_bert as process_data

import pickle
# import tensorflow_addons as tfa

import models.bilstm as resp_model  # 两层BiLSTM+CRF
# import transformer as resp_model
# import bert as resp_model

EMBED_DIM = 100
BiRNN_UNITS = 100
model_dir = "saved_model"


class NerModel(object):
  def __init__(self) -> None:
      super().__init__()



def create_model(train=True, dataset_name=None, model_name=None, use_crf=True):
    if train:
        (train_x, train_y), (val_x, val_y), (test_x, test_y), (vocab,
                                               chunk_tags) = process_data.load_data()
    else:
        with open('{}/{}_config.pkl'.format(model_dir, dataset_name), 'rb') as inp:
            (vocab, chunk_tags) = pickle.load(inp)
    n_tags = len(chunk_tags)
    model = resp_model.create_model(EMBED_DIM, vocab, n_tags, use_crf=use_crf)
    if train:
        # model.summary()
        return model, (train_x, train_y), (val_x, val_y), (test_x, test_y)
    else:
        return model, (vocab, chunk_tags)
