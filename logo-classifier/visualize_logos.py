from utils import read_json, get_labels, get_str_labels
import settings
import numpy as np
from PIL import Image
import tqdm
import torch.utils.data as t_data
import requests
from requests.exceptions import ConnectionError as RequestConnectionError
from requests.exceptions import SSLError, Timeout
import PIL
from io import BytesIO
import os 

class ImagePreloader(t_data.Dataset): 
    '''Class for main Dataset Classes''' 
    def __init__(self, classes: list, json_dataset: str, missed_logos: dict):
        self.ids = []
        self.source = []
        self.predicted = []
        self.true = []
        self.bounding_box = []
        self.score = []
        set_ids = set(missed_logos.keys())
        data_gen = read_json(json_dataset)
        for dict in tqdm.tqdm(data_gen):
            id = str(dict['logo_id'])
            if id in set_ids and (classes == [] or missed_logos[id]['prediction'] in classes):
                self.ids.append(id)
                self.source.append(dict['source_img'])
                self.predicted.append(missed_logos[id]['prediction']) 
                self.true.append(missed_logos[id]['truth'])
                self.bounding_box.append(dict['bounding_box'])
                self.score.append(missed_logos[id]['score']) 

    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, idx):

        return (
            self.ids[idx], 
            self.predicted[idx], 
            self.true[idx], 
            generate_images(
                self.source[idx], 
                self.bounding_box[idx],
            ),
            self.score[idx]
        )

def get_classes(input_json_file):
    classes_str, classes_ids = get_labels(settings.labels_path, [])
    classes_str = np.array(classes_str)
    classes_ids = np.array(classes_ids)

    missed_logos = [dict for dict in read_json(input_json_file)][0]

    res = {}

    for logos_id in missed_logos.keys():
        truth, prediction, score = missed_logos[logos_id]
        truth, prediction = get_str_labels([truth, prediction], classes_ids, classes_str)

        res[logos_id] = {'truth': truth, 'prediction': prediction, 'score': score}
        
    return res

def custom_collate_fn(batch):
    return batch
                
def generate_images(source_img: str, bounding_box: list):
    base_url = "https://images.openfoodfacts.org/images/products"
    image_url = base_url + source_img
    try: 
        r = requests.get(image_url)
    except (RequestConnectionError, SSLError, Timeout):
        return None
    if r.status_code == 404: 
        return None
    try:
        image = Image.open(BytesIO(r.content))
    except (Image.DecompressionBombError, PIL.UnidentifiedImageError):
        print(f"This one has a problem : source_img")
        return None
    try:
        assert np.shape(image)[0]>0
    except (IndexError, AssertionError):
        print(f"This one has a shape problem : source_img and shape = {np.shape(image)}")
        return None
    y_min, x_min, y_max, x_max = bounding_box
    height, width = np.shape(image)[0], np.shape(image)[1]
    cropped = image.crop((width*x_min, height*y_min, width*x_max, height*y_max))
    image = cropped.resize((224,224))
    if image.mode == 'CMYK':
        image = image.convert('RGB')
    return image

def save_image(id, predicted, true, image, score):
    base_dict = "missed_logos/"
    class_predicted_dict = predicted + "/"
    true_class_dict = true + "/"
    image_title = str(score) + "_" + str(id) + ".png"
    if not os.path.isdir(base_dict + class_predicted_dict):
        os.mkdir(base_dict + class_predicted_dict)
    if not os.path.isdir(base_dict + class_predicted_dict + true_class_dict):
        os.mkdir(base_dict + class_predicted_dict + true_class_dict)
        os.mkdir(base_dict + class_predicted_dict + true_class_dict + "true/")
        os.mkdir(base_dict + class_predicted_dict + true_class_dict + "false/")
    if not os.path.isfile(base_dict + class_predicted_dict + true_class_dict + image_title):
        try:
            image.save(base_dict + class_predicted_dict + true_class_dict + image_title)
        except:
            breakpoint()

if __name__ == '__main__':
    '''
    Once the training is run, if you want to check the logos where the model was wrong, run this script.
    *missed_logos_file should contain the path of the file created at the same time as the onnx 
    model saving during training.
    *logos_infos_file should contain the path of a jsonl file where each line is a dictionnary with
    as this one : {"class": "no_class", "id": 0, ....}.
    '''

    missed_logos_file = "missed_logos.json"
    logos_infos_file = "datasets/jsonl_dataset.jsonl"
    missed_logos = get_classes(missed_logos_file)
    dataset = ImagePreloader([], logos_infos_file, missed_logos)
    dataloader = t_data.DataLoader(dataset, batch_size=32, num_workers=4, collate_fn=custom_collate_fn)
    for batch in tqdm.tqdm(dataloader):
        for id, predicted, true, image, score in batch:
            save_image(id, predicted, true, image, score)

