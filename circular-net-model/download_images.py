import generate_barcode
import requests
import shutil
import os


DOWNLOAD_IMAGE_URL = "https://images.openfoodfacts.org/images/products/{}"    

def download_images_dataset():
    barcodes = []
    with open("barcodes.jsonl", "r") as barcode_file:
        for line in barcode_file:
            barcodes.append(line)
    
    for x in barcodes:
        image_file_name = generate_barcode.generate_image_path(x)
        url = DOWNLOAD_IMAGE_URL.format(image_file_name)

        res = requests.get(url, stream = True)

        if res.status_code == 200:
            with open('./images/'+x.strip()+".jpg",'wb') as f:
                shutil.copyfileobj(res.raw, f)
            print('Image sucessfully Downloaded: ',image_file_name)
        else:
            print('Image Couldn\'t be retrieved {}, {}'.format(DOWNLOAD_IMAGE_URL, image_file_name))

download_images_dataset()

