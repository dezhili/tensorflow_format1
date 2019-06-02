# -*- coding: utf-8 -*-
"""
Created on Sun Jul 23 19:58:52 2017

@author: cg
"""

import tensorflow as tf
import sys
import time
import numpy as np
import cv2
from datetime import datetime
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import matplotlib as plt
import pdb
from PIL import Image
from vgg16net import VGG16
from datagenerator import ImageDataGenerator

def correct_rate(predictions, labels):
    correct_pred = tf.equal(tf.argmax(predictions,1),tf.argmax(labels,1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred,"float"))
    return accuracy
    
       #logits = self.fc8
       #return logits
def correct_rate_v(predictions, labels):
    correct_pred = tf.equal(predictions,labels)
    accuracy = tf.reduce_mean(tf.cast(correct_pred,"float"))
    return accuracy


def get_restpartdata(batch_size, val_generator, n_classes, mean = np.array([102, 115, 147])):
    '''rest part test data'''
    rest_s = val_generator.data_size%batch_size
    rest_images_path = val_generator.images[-rest_s:]
    rest_labels_vec = val_generator.labels[-rest_s:]
    rest_batch_images_path=[]
    rest_batch_label=[]
    for ti in range(batch_size):
        if ti < rest_s:
            rest_batch_images_path.append(rest_images_path[ti])
            rest_batch_label.append(rest_labels_vec[ti])
        else:
            rest_batch_images_path.append(rest_images_path[rest_s-1])
            rest_batch_label.append(rest_labels_vec[rest_s-1])

    rest_images_data = np.ndarray([batch_size, 224, 224, 3])
    for di in range(batch_size):
        img = Image.open(rest_batch_images_path[di])

        img = img.resize((224,224), Image.ANTIALIAS)
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img = img.astype(np.float32)
        #subtract mean
        img -= mean

        rest_images_data[di] = img

    one_hot_labels = np.zeros((batch_size, n_classes))
    for ii in range(batch_size):
        one_hot_labels[ii][rest_batch_label[ii]] = 1

    return rest_images_data, one_hot_labels, rest_s
#=========================================================     
config = tf.ConfigProto(allow_soft_placement=True) 
config.gpu_options.allow_growth = True       


# Dataset path
train_file = 'basic_trainlabel.txt'
val_file = 'basic_testlabel.txt'


# Learning params
learning_rate = 0.001
num_epochs = 100
batch_size = 32
display_step = 20


train_layers = ['fc8', 'fc7','fc6','conv1_1','conv1_2','conv2_1',
                'conv2_2','conv3_1','conv3_2','conv3_3','conv4_1',
                'conv4_2','conv4_3','conv5_1','conv5_2','conv5_3']

n_classes = 7
keep_rate = 0.5
keep_var = tf.placeholder(tf.float32)


# Graph input
x = tf.placeholder(tf.float32, shape=(batch_size, 224, 224, 3))
y = tf.placeholder(tf.float32, shape=(batch_size, n_classes))

# Load dataset
train_generator = ImageDataGenerator(train_file, 
                                     horizontal_flip = True, shuffle = True)
val_generator = ImageDataGenerator(val_file, shuffle = False) 

# Get the number of training/validation steps per epoch
train_batches_per_epoch = np.floor(train_generator.data_size / batch_size).astype(np.int16)
val_batches_per_epoch = np.floor(val_generator.data_size / batch_size).astype(np.int16)

# Model
model = VGG16(x, keep_var, n_classes, train_layers)
score = model.fc8


var_list = [v for v in tf.trainable_variables() if v.name.split('/')[0] in train_layers]
# List of trainable variables of the layers we want to train
with tf.name_scope("cross_ent"):
    # Loss and optimizer
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=score, labels=y))

tf.summary.scalar('cross_entropy', loss)

with tf.name_scope("train"):

    gradients = tf.gradients(loss, var_list)
    gradients = list(zip(gradients, var_list))

    #optimizer =  tf.train.AdamOptimizer(0.01).minimize(loss)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate) #1attention:optimizer type
    train_op = optimizer.apply_gradients(grads_and_vars=gradients)


tf.summary.scalar('cross_entropy', loss)
# Evaluationn
# Init
init_op = tf.global_variables_initializer()

#Model 
model_path = '/hdd/###/RAF_DB/vgg_basic_fc8/model_'
saver = tf.train.Saver()



with tf.Session(config=config) as sess:
   
    sess.run(init_op)
    #pdb.set_trace()
    #model.load_initial_weights(sess)
    
    for epoch in range(num_epochs):

        print("{} Epoch number: {}".format(datetime.now(), epoch+1))
        step = 1

        while step < train_batches_per_epoch:
        # Get a batch of images and labels    
            batch_xs, batch_ys = train_generator.next_batch(batch_size)

            sess.run(train_op, feed_dict={x: batch_xs, y: batch_ys, keep_var: keep_rate})

            # Display training status
            if step%display_step == 0:
                #acc, batch_loss,lr = sess.run([correct_rate(tf.nn.softmax(model.fc8),batch_ys), loss, learning_rate], feed_dict={x: batch_xs, y: batch_ys, keep_var: 1.})
                batch_loss = sess.run(loss, feed_dict={x: batch_xs, y: batch_ys, keep_var: 1.})
                print("{} Iter {}: Training Loss = {:.4f}".format(datetime.now(), step, batch_loss)) 
            step += 1

        # Validate the model on the entire validation set
        print("{} Start validation".format(datetime.now()))

        test_pre_label=[]
        test_true_label=[]
        #pdb.set_trace()
        rest_test_pre_label=[]
        if val_generator.data_size%batch_size != 0:
            
            rest_images_data, rest_one_hot_labels,rest_s = get_restpartdata(batch_size , val_generator, n_classes)

            rest_pre_labels = sess.run(tf.argmax(tf.nn.softmax(model.fc8),1), feed_dict={x: rest_images_data, y: rest_one_hot_labels, keep_var: 1.})
            rest_test_pre_label1=rest_pre_labels.tolist()
            rest_test_pre_label=rest_test_pre_label1[:rest_s]

        for _ in range(int(val_generator.data_size/batch_size)):
            batch_tx, batch_ty = val_generator.next_batch(batch_size)
            #print("test_count = %d"%(test_count))
            pre_labels = sess.run(tf.argmax(tf.nn.softmax(model.fc8),1), feed_dict={x: batch_tx, y: batch_ty, keep_var: 1.})
                #print pre_labels
            test_pre_label += pre_labels.tolist()
                #test_true_label += tf.argmax(batch_ty,1).tolist()
        #pdb.set_trace()
        test_pre_label +=rest_test_pre_label
        test_acc = accuracy_score(val_generator.labels,test_pre_label)
        print("{} Iter {}: Testing Accuracy = {:.4f}".format(datetime.now(), step, test_acc)) 
        #all_test_acc.append(test_acc)

        print("F1 score = {:.4f}".format(f1_score(val_generator.labels, test_pre_label, average='macro')))
      
        print("Confusionmatrix =" )
        print (" {} ".format(confusion_matrix(val_generator.labels, test_pre_label)))

        # Reset the file pointer of the image data generator
        val_generator.reset_pointer()  #2attention
        train_generator.reset_pointer()

        #save model
        temp = '%05d' %step
           
        model_path_1 = model_path + temp + '.ckpt'
        #save_path = saver.save(sess, model_path_1)  
        #print("Model saved in file: {}".format(save_path))
                    
   
    
      
