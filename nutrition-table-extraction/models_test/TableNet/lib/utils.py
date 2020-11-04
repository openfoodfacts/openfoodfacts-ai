import matplotlib.pyplot as plt
import tensorflow as tf


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