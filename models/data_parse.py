import platform
class Parser(object):
    def __init__(self):
        pass

    def parse_data(self, s):
        return s

class ConllParser(Parser):
    def __init__(self):
        super().__init__()

    def parse_data(self, pth):
        fh = open(pth, 'r')
        #  in windows the new line is '\r\n\r\n' the space is '\r\n' . so if you use windows system,
        #  you have to use recorsponding instructions
        if platform.system() == 'Windows':
            split_text = '\r\n'
        else:
            split_text = '\n'

        string = fh.read()
        data = [[row.split() for row in sample.split(split_text)] for
                sample in
                string.strip().split(split_text + split_text)]
        for s in data:
            for w in s:
                if len(w) < 2:
                    print(w)
        fh.close()
        return data