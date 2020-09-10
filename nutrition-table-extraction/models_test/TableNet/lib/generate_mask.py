'''
Generate Comumn and Table mask from Marmot Data
'''

import xml.etree.ElementTree as ET
import os
import numpy as np
from PIL import Image

# Returns if columns belong to same table or not
def sameTable(ymin_1, ymin_2, ymax_1, ymax_2):
    min_diff = abs(ymin_1 - ymin_2)
    max_diff = abs(ymax_1 - ymax_2)

    if min_diff <= 5 and max_diff <=5:
        return True
    elif min_diff <= 4 and max_diff <=7:
        return True
    elif min_diff <= 7 and max_diff <=4:
        return True
    return False


if __name__ == "__main__":
    # directory = '../data/Marmot_data'
    directory = '../data/off_data'
    final_col_directory = '../data/column_mask/'
    if not os.path.exists(final_col_directory):
        os.makedirs(final_col_directory)
    final_table_directory = '../data/table_mask/'
    if not os.path.exists(final_table_directory):
        os.makedirs(final_table_directory)

    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        # Find all the xml files
        if filename.endswith(".xml"):
            filename = filename[:-4]

            # Parse xml file
            tree = ET.parse(os.path.join(directory , filename + '.xml'))
            root = tree.getroot()
            size = root.find('size')

            # Parse width
            width = int(size.find('width').text)
            height = int(size.find('height').text)

            # Create grayscale image array
            col_mask = np.zeros((height, width), dtype=np.int32)
            table_mask = np.zeros((height, width), dtype = np.int32)

            got_first_column = False
            i=0
            table_xmin = 10000
            table_xmax = 0

            table_ymin = 10000
            table_ymax = 0
            for obj in root.findall('object'):
                category = obj.find('name').text

                if category == 'column':
                    bndbox = obj.find('bndbox')
                    xmin = int(bndbox.find('xmin').text)
                    ymin = int(bndbox.find('ymin').text)
                    xmax = int(bndbox.find('xmax').text)
                    ymax = int(bndbox.find('ymax').text)
                
                    col_mask[ymin:ymax, xmin:xmax] = 255

                elif category == 'table':

                    bndbox = obj.find('bndbox')
                    xmin = int(bndbox.find('xmin').text)
                    ymin = int(bndbox.find('ymin').text)
                    xmax = int(bndbox.find('xmax').text)
                    ymax = int(bndbox.find('ymax').text)

                    table_mask[ymin:ymax, xmin:xmax] = 255

            # for column in root.findall('object'):
            #     bndbox = column.find('bndbox')
            #     xmin = int(bndbox.find('xmin').text)
            #     ymin = int(bndbox.find('ymin').text)
            #     xmax = int(bndbox.find('xmax').text)
            #     ymax = int(bndbox.find('ymax').text)

            #     col_mask[ymin:ymax, xmin:xmax] = 255
                                
            #     if got_first_column:
            #         if sameTable(prev_ymin, ymin, prev_ymax, ymax) == False:
            #             i+=1
            #             got_first_column = False
            #             table_mask[table_ymin:table_ymax, table_xmin:table_xmax] = 255
                        
            #             table_xmin = 10000
            #             table_xmax = 0

            #             table_ymin = 10000
            #             table_ymax = 0
                        
            #     if got_first_column == False:
            #         got_first_column = True
            #         first_xmin = xmin
                    
            #     prev_ymin = ymin
            #     prev_ymax = ymax
                
            #     table_xmin = min(xmin, table_xmin)
            #     table_xmax = max(xmax, table_xmax)
                
            #     table_ymin = min(ymin, table_ymin)
            #     table_ymax = max(ymax, table_ymax)

            # table_mask[table_ymin:table_ymax, table_xmin:table_xmax] = 255

            im = Image.fromarray(col_mask.astype(np.uint8),'L')
            im.save(final_col_directory + filename + ".jpeg")

            im = Image.fromarray(table_mask.astype(np.uint8),'L')
            im.save(final_table_directory + filename + ".jpeg")

            



