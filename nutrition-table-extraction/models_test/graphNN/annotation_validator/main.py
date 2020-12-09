import os
import pandas as pd
import json
import numpy as np
import cv2
from urllib.request import urlopen
from functools import partial
from bokeh.plotting import figure, show, curdoc
from bokeh.models.widgets import DataTable, TableColumn, Button, Paragraph
from bokeh.models import ColumnDataSource

root_path = "C:/Users/yichen.zhu/OneDrive - Accenture/Projects/OpenFoodFact/Implementation/annotation_validator/"
os.listdir(root_path)
os.chdir(root_path)

def url_to_image(url):
    # download the image, convert it to a NumPy array, and then read
    # it into OpenCV format
    resp = urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    image = np.array(image, dtype=np.int64)
    return image

def rectify_coord(x):
    return max(x,0)

def compute_image(df_row):
	data = json.loads(df_row['data'].replace("\'", "\"").replace("None","null"))['annotation']

	# Read original image from URL 
	url_prefix = 'https://static.openfoodfacts.org/images/products'
	source_image = df_row['source_image']
	img = url_to_image(url_prefix+source_image)

	# Crop image
	crop_rect = [data['crop']['start']['x'],data['crop']['start']['y'],data['crop']['end']['x'],data['crop']['end']['y']]
	crop_rect = [rectify_coord(x) for x in crop_rect]
	img_crop = img[crop_rect[1]:crop_rect[3],crop_rect[0]:crop_rect[2]]

	#Convert to Bokeh format image
	M, N, _ = img_crop.shape
	img_bk = np.empty((M, N), dtype=np.uint32)
	view = img_bk.view(dtype=np.uint8).reshape((M, N, 4))
	view[:,:,0] = img_crop[:,:,0]
	view[:,:,1] = img_crop[:,:,1]
	view[:,:,2] = img_crop[:,:,2]
	view[:,:,3] = 255
	img_bk = img_bk[::-1]

	return img_bk

def compute_table(df_row):
	# Load annotation data
	data = json.loads(df_row['data'].replace("\'", "\"").replace("None","null"))['annotation']

	# Filter bounding boxes
	bbox_raw = data['textAnnotations']
	count = 0
	bboxes_obj = {}

	for i in bbox_raw:
		if bbox_raw[i]['cell_id'] != None:
			bboxes_obj[count] = bbox_raw[i]
			count+=1
		else:
			None

	# Get table size
	max_row = 0
	max_col = 0

	for i in range(len(bboxes_obj)):
		row_index = [int(i) for i in bboxes_obj[i]['row_index']]
		col_index = [int(i) for i in bboxes_obj[i]['column_index']]
		bboxes_obj[i]['index'] = [[x, y] for x in row_index for y in col_index]
		max_row = max(max_row, max(row_index))
		max_col = max(max_col, max(col_index))

	# Generate table
	table = np.ndarray((max_row+1,max_col+1),object)

	for i in range(len(bboxes_obj)):
	    if len(bboxes_obj[i]['index']) > 1:
	        for idx in bboxes_obj[i]['index']:
	            if table[idx[0], idx[1]] != None:
	                table[idx[0], idx[1]] = table[idx[0], idx[1]] + bboxes_obj[i]['description']
	            else:
	                table[idx[0], idx[1]] = bboxes_obj[i]['description']
	    else:
	        if table[bboxes_obj[i]['index'][0][0],bboxes_obj[i]['index'][0][1]] != None:
	            table[bboxes_obj[i]['index'][0][0],bboxes_obj[i]['index'][0][1]] = table[bboxes_obj[i]['index'][0][0],bboxes_obj[i]['index'][0][1]]+" "+bboxes_obj[i]['description']
	        else:
	            table[bboxes_obj[i]['index'][0][0],bboxes_obj[i]['index'][0][1]] = bboxes_obj[i]['description']

	# Convert to datatable
	df_table = pd.DataFrame(table)
	df_table.columns = [str(x) for x in range(max_col+1)]

	return df_table

def val_handler(df):
	try:
		# Write validation result
		df.at[int(source_index.text),'validation'] = 1
		df.to_csv('dump_post_val.csv',index=False)

		#print(df['validation'])
		source_index.text = str(int(source_index.text)+1)
		print(source_index.text)
		df_row = df.iloc[int(source_index.text),:]
		# Update image
		img_bk_new = compute_image(df_row)
		img_source.data = {'image': [img_bk_new]}

		# Update table
		df_table_new = compute_table(df_row)
		source.data = df_table_new
		data_table.columns = [TableColumn(field=str(x),title=str(x)) for x in range(df_table_new.shape[1])]
	except:
		source_index.text = str(int(source_index.text)+1)
		print("Cannot load data")

def f_handler(df):
	try:
		# Write validation result
		df.at[int(source_index.text),'validation'] = 0
		df.to_csv('dump_post_val.csv',index=False)

		#print(df['validation'])
		source_index.text = str(int(source_index.text)+1)
		print(source_index.text)
		df_row = df.iloc[int(source_index.text),:]
		# Update image
		img_bk_new = compute_image(df_row)
		img_source.data = {'image': [img_bk_new]}

		# Update table
		df_table_new = compute_table(df_row)
		source.data = df_table_new
		data_table.columns = [TableColumn(field=str(x),title=str(x)) for x in range(df_table_new.shape[1])]
	except:
		source_index.text = str(int(source_index.text)+1)
		print("Cannot load data")
	
########################################################################################################
df = pd.read_csv('dump')
df['validation'] = ""
# Initial
source_index = Paragraph(text='1', name="source_index")
image_amount = Paragraph(text=str(len(df)), name="image_amount")
df_row = df.iloc[int(source_index.text),:]

# Display image
img_bk = compute_image(df_row)
img_source = ColumnDataSource({'image': [img_bk]})
p = figure(name="p")
p.image_rgba(image='image',x=0,y=0,dw=10,dh=10,source=img_source)

# Display table
df_table = compute_table(df_row)
source = ColumnDataSource(df_table)
columns = [TableColumn(field=str(x),title=str(x)) for x in range(df_table.shape[1])]
data_table = DataTable(source=source,columns=columns,width=350,height=300,name="data_table",index_position=None)

val = Button(label="Validate", name="val",button_type="success")
f = Button(label="False Annotation", name="f",button_type="danger")

val.on_click(partial(val_handler,df=df))
f.on_click(partial(f_handler,df=df))

curdoc().add_root(source_index)
curdoc().add_root(image_amount)
curdoc().add_root(p)
curdoc().add_root(data_table)
curdoc().add_root(val)
curdoc().add_root(f)