import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import json
from collections import namedtuple
Rectangle = namedtuple('Rectangle', 'xmin ymin xmax ymax')


img_height, img_width = 256, 256
def decode_img(img):
  # convert the compressed string to a 3D uint8 tensor
  img = tf.image.decode_jpeg(img)
  # resize the image to the desired size
  return tf.image.resize(img, [img_height, img_width]), img.shape

def normalize(input_image):
    input_image = tf.cast(input_image, tf.float32) / 255.0
    return input_image

def get_image_tensor(file):
    decoded_img, shape = decode_img(tf.io.read_file(file))
    return normalize(decoded_img), shape

def get_image_tensor_from_image_array(image):
    decoded_img = tf.image.resize(tf.keras.preprocessing.image.img_to_array(image), [img_height, img_width])
    return normalize(decoded_img), image.shape

def create_mask(pred_mask1, pred_mask2):
    pred_mask1 = tf.argmax(pred_mask1, axis=-1)
    pred_mask1 = pred_mask1[..., tf.newaxis]
    pred_mask2 = tf.argmax(pred_mask2, axis=-1)
    pred_mask2 = pred_mask2[..., tf.newaxis]
    return pred_mask1[0], pred_mask2[0]

def get_masks(model, image):
    pred_mask1, pred_mask2 = model.predict(image)
    table_mask, column_mask = create_mask(pred_mask1, pred_mask2)
    return table_mask, column_mask


def plot_predictions(model, image):
    table_mask, column_mask = get_masks(model,image)
    display([image[0], table_mask, column_mask])

def display(display_list):
    plt.figure(figsize=(15, 15))

    title = ['Input Image', 'Table Mask', 'Column Mask']

    for i in list(range(len(display_list)))[1:]:
       # plt.subplot(1, len(display_list), i+1)
        plt.subplot(1, 2, i)
        #plt.title(title[i])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[0]))
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]),alpha=0.5)
        plt.axis('off')
    plt.show()
    

def get_absolute_coordinates(rect1,rect2):
    x11, y11, w1, h1 = rect1
    x21, y21, w2, h2 = rect2
    
    x1 = x11 + x21
    x2 = x11 + x21 + w2
    y1 = y11 + y21
    y2 = y11 + y21 + h2
    
    return (x1,y1, x2-x1, y2-y1)




def convert_to_dict(string):
    #Convert dict string to dict object from json file
    return json.loads(string.replace("\'", "\"").replace('None','null'))


def circumscribed_rectangle(bounding_box):
    '''
    get circumscribed rectangle for this kind of bounding box
     ----------------
    |1-------------2|
    | \           / | 
    |   4--------3  |
    -----------------
    '''
    pt1 = bounding_box[0]
    pt2 = bounding_box[1]
    pt3 = bounding_box[2]
    pt4 = bounding_box[3]
    
    #Get furthest point left
    x1 = min(pt1['x'],pt4['x'])
    y1 = min(pt1['y'],pt2['y'])
    
    #Get furthest point right
    x2 = max(pt2['x'],pt3['x'])
    y2 = max(pt3['y'],pt4['y'])
    
    #Return rectangle as two points
    return Rectangle(x1,y1,x2,y2)

def dist(x1,x2):
    return abs(x1-x2)

def area(rect):
    l = rect.xmax-rect.xmin
    h = rect.ymax-rect.ymin
    return l*h


def intersection_area(rect1,rect2):
    #Compute intersaction of two straight rectangles
    dx = min(rect1.xmax, rect2.xmax) - max(rect1.xmin, rect2.xmin)
    dy = min(rect1.ymax, rect2.ymax) - max(rect1.ymin, rect2.ymin)
    if (dx>=0) and (dy>=0):
        return dx*dy
    else:
        return 0


def rotate_point(point,center,angle):
    #apply rotation and translation a point to find new coordinates
    coordinates = np.array([point['x'],point['y']])
    new_x = np.cos(angle) * (coordinates[0] - center[0]) + np.sin(angle) * (coordinates[1] - center[1]) + center[0]
    new_y = -np.sin(angle) * (coordinates[0] - center[0]) + np.cos(angle) * (coordinates[1] - center[1]) + center[1];
    new_point = {'x':round(new_x),'y':round(new_y)}
    return new_point