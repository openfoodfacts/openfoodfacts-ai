import cv2
import numpy as np
from urllib.request import urlopen
import sys
sys.path.append('../lib')
from utils import convert_to_dict, circumscribed_rectangle, area, intersection_area, rotate_point


class OFFImage:
    
    def __init__(self,image_url,table_bbx=None,json_ocr=None):
        self.image_url = image_url
        self.table_bbx = table_bbx
        self.json_ocr = json_ocr
        self.ocr = None
        self.image = None
        self.cropped_image = None
        self.corrected_image = None
        self.corrected_cropped_image = None
        self.corrected_ocr = None
        self.center = None
        self.angle = None
    
    @staticmethod
    def url_to_image(url):
        # download the image, convert it to a NumPy array, and then read
        # it into OpenCV format
        resp = urlopen(url)
        image = np.asarray(bytearray(resp.read()), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        image = np.array(image, dtype=np.uint8)
        return cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    
    @staticmethod
    def url_to_json(url):
        #Will use it to retreive ocr json file from image url, replace .jpg with .json
        pass
    
    def get_ocr(self):
        #Build Word object for each ocr detected word and filter out words where bounding box is missing a component (x or y for any point)
        self.ocr = [Word(description=word['description'],
                         bounding_box=word['boundingPoly']['vertices']) for word in self.json_ocr
                   if all('x' in coordinate.keys() and 'y' in coordinate.keys() for coordinate in word['boundingPoly']['vertices'])]
    
    
    def get_image(self):
        #Get original image and cropped image
        self.image = self.url_to_image(self.image_url)
        
        #Make sure there is no negative coordinates
        pt1 = (max(self.table_bbx.xmin,0),max(self.table_bbx.ymin,0))
        pt2 = (max(self.table_bbx.xmax,0),max(self.table_bbx.ymax,0))
        
        #Crop rectangle points might be inversed so need to correct it
        if pt1[0]>pt2[0]:
            a=pt1
            pt1=pt2
            pt2=a
        self.cropped_image = self.image[pt1[1]:pt2[1],pt1[0]:pt2[0]]
        
    def find_words_included_in_nutrition_table(self):
        for word in self.ocr:
            word.included_in_nutrition_table = self.is_included_in_nutrition_table(word)
            
    def find_words_relative_coordinates(self):
        #compute new coordinates of points inside cropped area
        for word in self.ocr:
            word.find_relative_bounding_box_coordinates(self.table_bbx)
    
    def is_included_in_nutrition_table(self,word):
        
        #Get word bounding_box vetices
        word_vertices = word.bounding_box
        
        #Find straight circumscribed rectangle
        bounding_box = circumscribed_rectangle(word_vertices)
        
        #Get inclusion score un nutrition table = area of intersection / area of word bounding box
        inclusion_score = intersection_area(bounding_box,self.table_bbx)/area(bounding_box)
         
        return inclusion_score>0.5

    def get_center_of_image(self):
        (h, w) = self.cropped_image.shape[:2]
        self.center = (w // 2, h // 2)
        
    def get_angle_of_image(self):
        #compute all angles
        all_angles = [x.get_angle() for x in self.ocr if x.included_in_nutrition_table]
        
        #Generel rotation is 
        self.angle = np.mean(all_angles)
        
        
    def rotate(self):
        #Method to correct image rotation
        (h, w) = self.cropped_image.shape[:2]
        #Compute center of rotation as center of image
        self.get_center_of_image()
        
        #General rotation angle 
        self.get_angle_of_image()
        
        #Get rotation matrix
        M = cv2.getRotationMatrix2D(self.center, self.angle*180/np.pi, 1.0)
        
        #Apply rotation on image
        rotated = cv2.warpAffine(self.cropped_image, M, (w, h), 
                                 flags=cv2.INTER_CUBIC,
                                 borderMode=cv2.BORDER_REPLICATE)
        self.corrected_image=rotated
        
        #To compute absolute rotation of bounding boxes, we need to compute the relative rotation around the center of the bounding box (cent_1) 
        #and apply a translation of the vector Vect(Vect(0,cent1)-Vect(0,cent1))
        for word in self.ocr:
            #YICHEN, CAN WE REUSE DETERMINE ROTATION MATRIX FROM IMAGE ROTATION MATRIX ?
            word.rotate(self.center,self.angle)
            
            

class Word:
    def __init__(self,description,bounding_box):
        self.description = description
        self.bounding_box = bounding_box
        self.relative_bounding_box=None
        self.included_in_nutrition_table = False
        
        
    def __repr__(self):
        if self.included_in_nutrition_table:
            return "Word: {}, Bounding Box: {}, included".format(self.description, self.bounding_box)
        else:
            return "Word: {}, Bounding Box: {}, not included".format(self.description, self.bounding_box)
    
    
    @staticmethod
    def find_relative_coordinates_in_crop_rectangle(point,crop_rectangle):
        #Compute new coordinates inside a crop.
        relative_x = point['x']-crop_rectangle.xmin
        relative_y =  point['y'] - crop_rectangle.ymin
        return {'x':relative_x,'y':relative_y}
    
    def find_relative_bounding_box_coordinates(self,nutrition_table_bounding_box):
        #Compute new coordinates inside a crop.
        self.relative_bounding_box = [self.find_relative_coordinates_in_crop_rectangle(x,nutrition_table_bounding_box)
                                      for x in self.bounding_box]
        
    

    def get_angle(self):
        #Get word's relative bounding box
        vertices = self.bounding_box
        #get the two bottom points of the rectangle
        pt3 = vertices[2]
        pt4 = vertices[3]
        x3 = pt3['x']
        x4 = pt4['x']
        y3 = pt3['y']
        y4 = pt4['y']
        #Compute adjacent segment
        adj = abs(x3-x4)
        #compute hypotenuse
        hyp = np.sqrt((x3-x4)**2 + (y3-y4)**2)
        #rotation direction
        rot_direc = np.sign(y3-y4)
        #angle in radian
        angle = np.arccos(adj/hyp)
        return rot_direc*angle
    
    
    def rotate(self,center,angle):
        #self.bounding_box = [rotate_point(x,center,angle) for x in self.bounding_box]
        self.relative_bounding_box = [rotate_point(x,center,angle) for x in self.relative_bounding_box]