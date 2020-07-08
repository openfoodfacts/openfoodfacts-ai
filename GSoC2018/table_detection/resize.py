import os
import glob
from PIL import Image

def get_filename(file):
    name_list = filename.split("/")
    name = name_list[-1].split(".")
    return name[0]

def resize(filename, nx, ny):
    imagename = get_filename(filename)
    img = im.resize((int(nx), int(ny)), Image.ANTIALIAS)
    img.save("test_images/{}.jpg".format(imagename), optimize=True, quality=95)

count = 0
path = 'test_images/'
for filename in glob.glob(os.path.join(path, '*.jpg')):
    im = Image.open(filename)
    nx, ny = im.size
    if(nx >= ny):
        new_nx = 1000
        ratio = new_nx / nx
        new_ny = ratio * ny 
        resize(filename, new_nx, new_ny)

    else:
        new_ny = 1000
        ratio = new_ny / ny
        new_nx = ratio * nx 
        resize(filename, new_nx, new_ny)
    print(filename)
    count+=1



print (count)