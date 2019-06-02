import numpy as np
import cv2
import pdb
from PIL import Image
"""
This code is highly influenced by the implementation of:
https://github.com/joelthchao/tensorflow-finetune-flickr-style/dataset.py
But changed abit to allow dataaugmentation (yet only horizontal flip) and 
shuffling of the data. 
The other source of inspiration is the ImageDataGenerator by @fchollet in the 
Keras library. But as I needed BGR color format for fine-tuneing AlexNet I 
wrote my own little generator.
"""

class ImageDataGenerator:
    def __init__(self, class_list, horizontal_flip=False, shuffle=False, 
                 mean = np.array([102, 115, 147]), scale_size=(224, 224),
                 nb_classes = 7):
        
                
        # Init params
        self.horizontal_flip = horizontal_flip
        self.n_classes = nb_classes
        self.shuffle = shuffle
        self.mean = mean
        self.scale_size = scale_size
        self.pointer = 0
        
        self.read_class_list(class_list)
        
        if self.shuffle:
            self.shuffle_data()

    def read_class_list(self,class_list):
        """
        Scan the image file and get the image paths and labels
        """
        with open(class_list) as f:
            lines = f.readlines()
            self.images = []
            self.labels = []
            for l in lines:
                items = l.split()
                self.images.append(items[0])
                self.labels.append(int(items[1]))
            
            #store total number of data
            self.data_size = len(self.labels)
        
    def shuffle_data(self):
        """
        Random shuffle the images and labels
        """
        images = self.images.copy()
        labels = self.labels.copy()
        self.images = []
        self.labels = []
        
        #create list of permutated index and shuffle data accoding to list
        idx = np.random.permutation(len(labels))
        for i in idx:
            self.images.append(images[i])
            self.labels.append(labels[i])
                
    def reset_pointer(self):
        """
        reset pointer to begin of the list
        """
        self.pointer = 0
        
        if self.shuffle:
            self.shuffle_data()
        
    
    def next_batch(self, batch_size):
        """
        This function gets the next n ( = batch_size) images from the path list
        and labels and loads the images into them into memory 
        """
        # Get next batch of image (path) and labels
        paths = self.images[self.pointer:self.pointer + batch_size]
        labels = self.labels[self.pointer:self.pointer + batch_size]
        
        #update pointer
        self.pointer += batch_size
        
        # Read images
        images = np.ndarray([batch_size, self.scale_size[0], self.scale_size[1], 3])
        for i in range(len(paths)):
            img = Image.open(paths[i])
            
            #img = np.asarray(img)[:,:,::-1].copy()
            #flip image at random if flag is selected
            #pdb.set_trace()
            #rescale image
            img = img.resize((224,224), Image.ANTIALIAS)
            img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            if self.horizontal_flip and np.random.random() < 0.5: # validation data dose not need flip!!!
                img = cv2.flip(img, 1)
            img = img.astype(np.float32)
            
            #subtract mean
            img -= self.mean
                                                                 
            images[i] = img
        # Expand labels to one hot encoding
        one_hot_labels = np.zeros((batch_size, self.n_classes))
        for i in range(len(labels)):
            one_hot_labels[i][labels[i]] = 1

        #return array of images and labels
        return images, one_hot_labels


