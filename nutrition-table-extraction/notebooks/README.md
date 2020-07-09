# First steps to run the notebooks:

## Dowload the Dataset
```
bash download_dataset.sh <YOUR_BASE_PATH>
```
You should get a working tree similar to this:
```
+-<YOUR_BASE_PATH>/
  |
  +-arial.ttf
  |
  +-image_files/
  | |
  | +-3596710308583.nutrition.jpg
  | |
  | +-3256224160069.nutrition.jpg
  | |
  | +-... <additional files>
  |
  +-json_files/
  | |
  | +-3770011826018.nutriments.json
  | |
  | +-... <additional files>
  |
  +-cropped_json_files/
  | |
  | +-3660992120123.nutrition.cropped.json
  | |
  | +-... <additional files>
  |
  +-cropped_images/
    |
    +-20112349.nutrition.cropped.jpg
    |
    +-... <additional files>
```
## Config
- Copy the config_example.yml file into config.yml 
- Update config.yml with the required information:
    - **base_path:** path to the dataset
    - **endpoint:** formRecognizer api endpoint
    - **apim_key:** API key
# Next steps
- FormRecongnizer API description in order to get **endpoint** and **apim_key**
- Train the custom FormRecognizer API
- Develop data processing pipeline. (cf description in the data_exploration notebook)
- Develop line detection algorithm to handle cases where the table structure is not detected
    - Idea: take the middle point of the bounding box (x_middle, y_middle) and perform a clustering technique based on y_middle.    
        - Bounding boxes on the same line tend to have very close y_middle values => same cluster.
        - The API returns the document rotation angle, it should be taken into account while projecting the middle points on the y_axis of the image.