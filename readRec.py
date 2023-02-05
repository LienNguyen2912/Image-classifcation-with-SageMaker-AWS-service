# if you use python 3.6,
# in order to use matplotlib you may need to install
# pip install matplotlib==3.0.3
# pip install pyparsing==2.4.7
# due to version error and dll loaded error
# refer https://stackoverflow.com/questions/69964466/unknown-version-in-python-library-pyparsing
# https://stackoverflow.com/questions/24251102/from-matplotlib-import-ft2font-importerror-dll-load-failed-the-specified-pro

# below is  how to read and plot images from .rec file
# dataset_dogs_cats
# ├── trainTestRec // cd to here to generate .rec , .lst and .idx files(*)
# │   ├── train
# │       ├── cats
# │           ├── 1.jpg
# │           └── 2.jpg
# │       ├── dogs
# │           ├── 1.jpg
# │           └── 2.jpg
# └── dataset_rec_val.rec
# └── dataset_rec_val.lst
# └── dataset_rec_val.idx

# (*) is
# python tools/im2rec.py ./dataset_rec_val ./train/ --recursive --list --num-thread 8
# python im2rec.py ./dataset_rec_val ./train --recursive --pass-through --pack-label --resize 254 --quality 90 --num-thread 8

import mxnet as mx
import numpy as np
from matplotlib import pyplot
import os
path = os.getcwd()
print(path)

sample_train_number = 0
data_iter = mx.image.ImageIter(batch_size=4, data_shape=(3, 224, 224),
                              path_imgrec='dataset_dogs_cats/dataset_rec_val.rec',
                              path_imgidx='dataset_dogs_cats/dataset_rec_val.idx')
data_iter.reset()
for j in range(4):
    batch = data_iter.next()
    sample_train_number += 1
    data = batch.data[0]
    #print(batch)
    label = batch.label[0].asnumpy()
    for i in range(4):
        ax = pyplot.subplot(1,4,i+1)
        pyplot.imshow(data[i].asnumpy().astype(np.uint8).transpose((1,2,0)))
        
        ax.set_title('class: ' + str(label[i]))
        pyplot.axis('off')
    pyplot.show()

