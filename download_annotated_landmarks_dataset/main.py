
import os
import sys
import urllib
import requests

lines = []
with open('annotations_landmarks/annotation_full_train.txt') as f:
    lines += f.readlines()
with open('annotations_landmarks/annotation_full_val.txt') as f:
    lines += f.readlines()

image_path = './images/'
try:
    os.mkdir(image_path)
except:
    print 'hmm'

i = 0
for line in lines:
    if i % 1000 == 0:
        print '=============', i
    line = line.split()
    try:
            r = requests.get(line[0], allow_redirects=False, timeout=10.0)
            if r.status_code == 200:
                open(image_path + str(i) + '.jpg', 'wb').write(r.content)
            else:
                print i, r.status_code
    except requests.exceptions.Timeout as ex:
        print i, 'time out'
    except Exception as e:
        if e is KeyboardInterrupt:
            raise e
        print i, 'something is wrong with', line[0], e
    i += 1

