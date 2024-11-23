#%%
from ultralytics import YOLO
#%%
data_path = "~/data_dump/openfoodfacts/yolo"

# Load a model
model = YOLO(f"{data_path}/yolo11n.pt")

# Train the model
train_results = model.train(
    data=f"{data_path}/data.yaml",  # path to dataset YAML
    epochs=1000,  # number of training epochs
    patience=20,  # early stopping patience
    dropout=0.1,  # dropout probability
    imgsz=640,  # training image size
    plots=True,  # create plots
    device="cpu",  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
    project=f"{data_path}/model",  # save training results to project/name
)

# Evaluate model performance on the validation set
metrics = model.val()
#%%
# Perform object detection on an image

image_test_list = [f"{data_path}/test/4xXuBb8u3E.jpg", f"{data_path}/test/test_image.jpg"]

results = model.predict(image_test_list, conf=0.1, save=True, project=f"{data_path}/test")  # or with a list of image paths
# results[0].show()
#%%
# Export model
path = model.export(format="onnx") 
#%%
