import os


def load():
    file_reader = open(os.path.join('data', 'MRPC', 'dev.tsv'), 'r')
    _file_content = file_reader.read()
    _data = list()

    for line in _file_content.split('\n')[1:]:
        if line.strip() == '':
            continue

        line = line.split('\t')
        _data.append(line)
    file_reader.close()

    return _data
