"""
Tools and functions used in the notebooks 
"""
import json
import time
from sys import exit

from requests import get, post

import cv2
import filetype
import matplotlib
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pandas as pd
from bounding_box import bounding_box as bb
from constants import (suffix_cropped_images, suffix_cropped_json,
                       suffix_images, suffix_json)
from PIL import Image, ImageDraw, ImageFont
from tenacity import retry, wait_fixed

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
    print("\n" + "=-"*50)
    print("Done...%s" %img_name)
    return json_data

def show_img(image, img_id, dim=10):
    plt.figure(figsize=(dim, dim))
    plt.title(img_id)
    plt.imshow(image)

def display_bb(img_path, lines, base_path, show_text=True):
    img_id = img_path.split("/")[-1].split(".")[0]
    vmin = 1
    vmax = len(lines)

    cmap = matplotlib.cm.tab20
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)

    with Image.open(img_path) as image:
        draw = ImageDraw.Draw(image)
        for i, line in enumerate(lines):
            color = cmap(norm(i))
            color = tuple(int(255*c) for c in color)
            bb_coords, text = line['boundingBox'], line['text']
            draw.line(bb_coords[:8], width=5, fill=color)
            draw.line((bb_coords[0], bb_coords[1], bb_coords[-2], bb_coords[-1]), width=5, fill=color)

            if show_text:
                x, y = bb_coords[0], bb_coords[1]
                a, b = bb_coords[-2], bb_coords[-1]
                font_size = abs(y-b) // 2
                fnt = ImageFont.truetype(base_path + 'arial.ttf', font_size)
                w, h = fnt.getsize(text)
                draw.rectangle((x, y-h, x + w, y), fill=(0,0,0,0))
                draw.text((x, y-h), text, font=fnt, fill=(209, 239, 8) )
            #bb.add(image, a, b, c, d, text)
        show_img(image, img_id)
        
def get_table_df(json_data):
    api_table = json_data['analyzeResult']['pageResults'][0]['tables'][0]
    #print("got api_table!")
    n_rows, n_cols, cells = api_table["rows"], api_table["columns"], api_table["cells"]
    table_res = {r : {c : {"text":"-", "boundingBox": []} for c in range(n_cols)} for r in range(n_rows)}
    for cell in cells:
        table_res[cell["rowIndex"]][cell["columnIndex"]] = {"text": cell["text"], "boundingBox": cell["boundingBox"]}
    df = pd.DataFrame.from_dict(table_res).T.applymap(lambda x: x["text"])
    return df

def test_single_img(img_id, base_path, endpoint, apim_key, post_url, headers, cropped_img=True):
    source = base_path + "image_files/%s.nutrition.jpg" %img_id
    if cropped_img:
        source = base_path + "cropped_images/%s.nutrition.cropped.jpg" %img_id
    json_data = get_api_resp(endpoint, apim_key, post_url, source, headers)
    lines = json_data['analyzeResult']['readResults'][0]['lines']
    display_bb(source, lines, base_path)
    plt.show()
    df = get_table_df(json_data)
    display(df)
    return df, json_data





