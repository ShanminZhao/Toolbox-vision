import os

def mkdir(path):
    if os.path.exists(path):
        print('file is exit.')
    else:
        os.mkdir(path)