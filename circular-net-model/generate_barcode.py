import re
from typing import Optional, List
import requests

FETCH_IMAGE_ID_URL = "https://world.openfoodfacts.org/api/v0/product/{}.json?fields=images"

BARCODE_PATH_REGEX = re.compile(r"^(...)(...)(...)(.*)$") 

def get_image_id(barcode: str) -> Optional[str]:
    url = FETCH_IMAGE_ID_URL.format(barcode)
    result = requests.get(url).json()
    image_data = result["product"]["images"]
    front_image_key = "front_fr"

    for key, val in image_data.items():
        if key.startswith("front_"):
            front_image_key = key

    image_id = image_data[front_image_key]["imgid"]
    return image_id

def generate_image_path(barcode: str) -> Optional[str]:
    image_id = get_image_id(barcode)
    splitted_barcode = split_barcode(barcode.strip())
    if splitted_barcode is None:
        return None
    return "{}/{}.jpg".format("/".join(splitted_barcode), image_id)
    
def split_barcode(barcode: str) -> Optional[List[str]]:
    if not barcode.isdigit():
        print("unknown barcode format: {}".format(barcode))
        # raise ValueError("unknown barcode format: {}".format(barcode))
        return None
        
    match = BARCODE_PATH_REGEX.fullmatch(barcode)

    if match:
        return [x for x in match.groups() if x]
    
    return [barcode]
