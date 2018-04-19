
import os
import sys
import urllib
import requests

# read txt
lines = []
with open('annotations_landmarks/annotation_full_train.txt') as f:
    lines += f.readlines()
with open('annotations_landmarks/annotation_full_val.txt') as f:
    lines += f.readlines()

# create folder
image_path = './images/'
try:
    os.mkdir(image_path)
except:
    print 'hmm'
    
# download function
def download_image(i):
    if i % 1000 == 0:
        print '=============', i
    line = lines[i].split()
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
    
# par download
import multiprocessing  
pool = multiprocessing.Pool(processes=10)
pool.map(download_image, range(len(lines)))
pool.close()
pool.join()   
print('done')

# check image read
import os
import sys
import imageio
rm_count = 0
for f in os.listdir(image_path):
    f = image_path + f
    try:
        img = imageio.imread(f)
    except:
        os.remove(f)
        rm_count += 1
print rm_count
