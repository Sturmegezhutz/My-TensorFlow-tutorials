###################################################################################################
"""
Copyright(C), 2017 - 2027, ivvi Scientific (NanChang) Co.,Ltd
File name: input_data.py
Description:input_data.py is used for build the model architecture
Others:data:

Department:  AI Innovation department
Author:      AI Innovation department Software team
Version:     V1.00.01
Date:        2017.10.18

Function List:
1....

History:
1.Author:     guanxuejin
Date:         2017.10.18
Modification: Create file
"""
###################################################################################################

import tensorflow as tf
import numpy as np
import os
import config

#%%
img_width = 288
img_height = 288

#%%
def get_files(file_dir):
    """
    Args:
        file_dir: file directory
    Returns:
        list of images and labels
    """
    cats = []
    label_cats = []
    dogs = []
    label_dogs = []
    for file in os.listdir(file_dir):
        name = file.split(sep=".")
        if name[0] == "cat":
            cats.append(file_dir + file)
            label_cats.append(0)
        else:
            dogs.append(file_dir + file)
            label_dogs.append(1)
    print("There are %d cats\nThere are %d dog"%(len(cats), len(dogs)))

    #stack the file on top of each other
    image_list = np.hstack((cats, dogs))
    # stack the label on top of each other
    label_list = np.hstack((label_cats, label_dogs))

    #disrupt the order, so it wont be done again when generating the batch
    temp = np.array([image_list, label_list])
    temp = temp.transpose()
    np.random.shuffle(temp)

    image_list = list(temp[:,0])
    label_list = list(temp[:,1])
    label_list = [int(i) for i in  label_list]

    return image_list, label_list

#%%
def get_batch(image, label, image_W, image_H, batch_size, capacity):
    """
    Args:
        :param image: list type
        :param label: list type
        :param image_W: image width
        :param image_H: image height, if the size of image is not suitable, it should be modified in this function
        :param batch_size: batch size
        :param capacity: the maximum elements in queue
    Returns:
        image_batch: 4D tensor [batch_size, width, height, 3], dtype=tf.float32
        label_batch: 1D tensor [batch_size], dtype=tf.int32
    """

    # transform the python list to tf format
    image = tf.cast(image, tf.string)
    label = tf.cast(label, tf.int32)

    # make an input queue
    input_queue = tf.train.slice_input_producer([image, label])

    label = input_queue[1]
    image_contents = tf.read_file(input_queue[0])
    image = tf.image.decode_jpeg(image_contents, channels=3)
    ########################################
    # data argumentation  should go to here
    ########################################
    image = tf.image.resize_image_with_crop_or_pad(image, image_W, image_H)
    #TODO:replace of tf.image.resize_image_with_crop_or_pad()
    #image = tf.image.resize_images(image, [image_H, image_W], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    #TODO: delete of tf.image.per_image_standardization()
    image = tf.image.per_image_standardization(image)
    image_batch, label_batch = tf.train.batch([image,label],batch_size=batch_size, num_threads=64, capacity=capacity)

    #image_batch, label_batch = tf.train.shuffle_batch([image,label],batch_size=BATCH_SIZE, num_threads=64, capacity=CAPACITY, min_after_dequeue=CAPACITY-1)

    #TODO: delete of tf.reshape()
    label_batch = tf.reshape(label_batch, [batch_size])
    image_batch = tf.cast(image_batch, tf.float32)
    
    return image_batch, label_batch


 
#%% TEST
# To test the generated batches of images
# When training the model, DO comment the following codes

"""
import matplotlib.pyplot as plt

BATCH_SIZE = 2
CAPACITY = 256
IMG_W = 208
IMG_H = 208

train_dir = config.train_dir

image_list, label_list = get_files(config.train_dir)
image_batch, label_batch = get_batch(image_list, label_list, IMG_W, IMG_H, BATCH_SIZE, CAPACITY)

with tf.Session() as sess:
    i = 0
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    
    try:
        while not coord.should_stop() and i<1:
            
            img, label = sess.run([image_batch, label_batch])
            
            # just test one batch
            for j in np.arange(BATCH_SIZE):
                print('label: %d' %label[j])
                plt.imshow(img[j,:,:,:])
                plt.show()
            i+=1
            
    except tf.errors.OutOfRangeError:
        print('done!')
    finally:
        coord.request_stop()
    coord.join(threads)
"""

#%%





    
