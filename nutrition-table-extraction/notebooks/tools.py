"""
Tools and functions used in the notebooks 
"""
from requests import get, post
import filetype
import json
import cv2
from bounding_box import bounding_box as bb
import matplotlib.pyplot as plt
import time
from sys import exit
from tenacity import retry, wait_fixed
from constants import suffix_images, suffix_cropped_images, suffix_json, suffix_cropped_json
import matplotlib.image as mpimg

def explore_image(img_id, base_path):
    data_path = base_path + "nutrition-lc-fr-country-fr-last-edit-date-2019-08"
    cropped_imgs_path = base_path + "cropped_images/"
    imgs_path = base_path + "image_files/"
    json_path = base_path + "json_files/"
    cropped_json_path = base_path + "cropped_json_files/"

    img_name = img_id + suffix_images
    cropped_image_name = img_id + suffix_cropped_images
    json_name =  img_id + suffix_json
    cropped_json_name =  img_id + suffix_cropped_json
    
    img = mpimg.imread(imgs_path + img_name)
    cropped_img = mpimg.imread(cropped_imgs_path + cropped_image_name )
    json_file_path = json_path + json_name
    cropped_json_file_path = cropped_json_path + cropped_json_name
    
    with open(json_file_path) as f:
        json_file = json.load(f)
        
    with open(cropped_json_file_path) as cf:
        cropped_json_file = json.load(cf)

    print("image id: %s" %img_id)
    
    imgs = [img, cropped_img]
    fig = plt.figure(figsize=(16, 16))
    for (i, img) in enumerate(imgs):
        fig.add_subplot(1, 2, i+1)
        plt.imshow(img)
    plt.show()
    
    print(json_file)
    print("-"*50)
    #print(cropped_json_file)
    print("*"*100)

####

@retry(wait=wait_fixed(40))
def post_api_request(endpoint, apim_keys, post_url, source, headers):
    with open(source, "rb") as f:
        data_bytes = f.read()
    resp = post(url = post_url, data = data_bytes, headers = headers)
    get_url = resp.headers["operation-location"]
    return get_url

@retry(wait=wait_fixed(40))
def get_api_resp(endpoint, apim_keys, post_url, source, headers):
    img_name = source.split("/")[-1]
    get_url = post_api_request(endpoint, apim_keys, post_url, source, headers)
    resp = get(url = get_url, headers = headers)
    if resp.status_code != 200:
        raise Exception("GET analyze failed:\n%s" % resp.text)

    json_data = json.loads(resp.text)
    while json_data['status'] != 'succeeded':
        #print("sleeping 2s..")
        time.sleep(2)
        resp = get(url = get_url, headers = headers)
        json_data = json.loads(resp.text)
    print("Done...%s" %img_name)
    return json_data

def show_img(image, img_id, dim=20):
    plt.figure(figsize=(dim, dim))
    plt.title(img_id)
    plt.imshow(image)

def display_bb(img_path, lines):
    image = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img_id = img_path.split("/")[-1].split(".")[0]
    for line in lines:
        bb_coords, text = line['boundingBox'], line['text']
        a,b,c,d = bb_coords[0], bb_coords[1], bb_coords[2], bb_coords[5]
        bb.add(image, a, b, c, d, text)
    show_img(image, img_id)

    