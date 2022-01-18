import sys
import pytest
import logging

def test_parse_line():
    from utils import tag_utils
    line = '当 O\n希 O\n望 O\n工 O\n程 O'
    ws, ts = tag_utils.parse_line(line)
    assert ws == ['当', '希', '望', '工', '程']


def test_BIEO2Word():
    from utils import tag_utils
    ws = ['当', '希', '望', '工', '程']
    ts = ['O', 'B-Pro', 'I-Pro', 'I-Pro', 'E-Pro']
    res = tag_utils.BIEO2Word(ws, ts)
    print(res)
    assert 0


def test_a():  # test开头的测试函数
    print("------->test_a")
    assert 1  # 断言成功


def test_parse():
    import cck_process_data_bert
    train = cck_process_data_bert._parse_data(open('data/CCKS2021中文NLP地址要素解析/train_human.conll', 'r'))
    print(train)


if __name__ == "__main__":
    # print(__file__)
    sys.exit(pytest.main([__file__]))
