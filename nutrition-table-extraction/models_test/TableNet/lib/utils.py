import matplotlib.pyplot as plt
import tensorflow as tf

def create_mask(pred_mask1, pred_mask2):
    pred_mask1 = tf.argmax(pred_mask1, axis=-1)
    pred_mask1 = pred_mask1[..., tf.newaxis]
    pred_mask2 = tf.argmax(pred_mask2, axis=-1)
    pred_mask2 = pred_mask2[..., tf.newaxis]
    return pred_mask1[0], pred_mask2[0]

def get_masks(model, image):
    pred_mask1, pred_mask2 = model.predict(image, verbose=1)
    table_mask, column_mask = create_mask(pred_mask1, pred_mask2)
    return table_mask, column_mask


def plot_predictions(model, image):
    table_mask, column_mask = get_masks(model,image)
    display([image[0], table_mask, column_mask])

def display(display_list):
    plt.figure(figsize=(15, 15))

    title = ['Input Image', 'Table Mask', 'Column Mask']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        plt.axis('off')
    plt.show()