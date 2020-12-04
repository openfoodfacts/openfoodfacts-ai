import cv2
import numpy as np
from urllib.request import urlopen
import sys
import json
sys.path.append('../lib')
from utils import convert_to_dict, circumscribed_rectangle, area, intersection_area, rotate_point


class OFFImage:
    
    """The summary line for a class docstring should fit on one line.

    If the class has public attributes, they may be documented here
    in an ``Attributes`` section and follow the same formatting as a
    function's ``Args`` section. Alternatively, attributes may be documented
    inline with the attribute's declaration (see __init__ method below).

    Properties created with the ``@property`` decorator should be documented
    in the property's getter method.

    Attributes:
        attr1 (str): Description of `attr1`.
        attr2 (:obj:`int`, optional): Description of `attr2`.

    """

    
    def __init__(self,image_url,table_bbx=None):
        self.image_url = image_url
        self.table_bbx = table_bbx
        self.ocr = None
        self.image = None
        self.cropped_image = None
        self.corrected_image = None
        self.center = None
        self.angle = None
    
    @staticmethod
    def url_to_image(url):
        """Get image from url in uint8 type

        Keyword arguments:
        url -- String
        """
        # download the image, convert it to a NumPy array, and then read
        # it into OpenCV format
        resp = urlopen(url)
        image = np.asarray(bytearray(resp.read()), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        image = np.array(image, dtype=np.uint8)
        return cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    
    @staticmethod
    def url_to_json(url):
        """Get json from url

        Keyword arguments:
        url -- String
        """
        #Will use it to retreive ocr json file from image url, replace .jpg with .json
        response = urlopen(url)
        result = response.read().decode('utf8')
        return json.loads(result)
        
        
    def get_ocr(self):
        """Build Word object for each ocr detected word and filter out words where bounding box is missing a component (x or y for any point)
        """
        
        #replace .jpg to .json to retreive json file
        json_url = self.image_url.replace('.jpg','.json')
        
        #get json
        json_response = self.url_to_json(json_url)
        
        #get text annotation from json file
        ocr_json = json_response['responses'][0]['textAnnotations']

        #build word objects list from result
        self.ocr = [Word(description=word['description'],
                         bounding_box=word['boundingPoly']['vertices']) for word in ocr_json
                   if all('x' in coordinate.keys() and 'y' in coordinate.keys() for coordinate in word['boundingPoly']['vertices']) and 'locale' not in word.keys()]
    
    
    def get_image(self):
        """Get original image from image url and and cropped image using the object crop bounding box
        """
        #Get original image
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
        """For each words in ocr attribute, set attribute is_included_in_nutrition_table to True if word is inside crop boundaries
        """
        for word in self.ocr:
            word.included_in_nutrition_table = self.is_included_in_nutrition_table(word)
            
    def find_words_relative_coordinates(self):
        """For each words in ocr attribute, set attribute is_included_in_nutrition_table to True if word is inside crop boundaries
        """
        #compute new coordinates of points inside cropped area
        for word in self.ocr:
            if word.included_in_nutrition_table:
                word.find_relative_bounding_box_coordinates(self.table_bbx)
    
    def is_included_in_nutrition_table(self,word):
        """Compute intersection area between a word area and nutrition table area and return a boolean to check if
        percentage of the word's area included in nutritional table is greater than 50%
        Keyword arguments:
        word -- Word
        """
        #Get word bounding_box vetices
        word_vertices = word.bounding_box
        
        #Find straight circumscribed rectangle
        bounding_box = circumscribed_rectangle(word_vertices)
        
        #Get inclusion score un nutrition table = area of intersection / area of word bounding box
        inclusion_score = intersection_area(bounding_box,self.table_bbx)/area(bounding_box)
         
        return inclusion_score>0.5

    def get_center_of_image(self):
        """Compute center point of the image
        """
        (h, w) = self.cropped_image.shape[:2]
        self.center = (w // 2, h // 2)
        
    def get_angle_of_image(self):
        """Compute rotation angle of the image
        """
        #compute all angles
        all_angles = [x.get_angle() for x in self.ocr if x.included_in_nutrition_table]
        
        #Generel rotation is 
        self.angle = np.mean(all_angles)
        
        
    def rotate(self):
        """Compute image center and angle for rotation, rotate image and update bounding boxes coordinates
        """
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
            if word.included_in_nutrition_table:
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
        """Returns relative coordinates of points inside a crop area if a dict format. 
        Example output: {'x':150,'y':100}
        
        Keyword arguments:
        point -- dict
        crop_rectangle -- named tuple Rectangle('Rectangle', 'xmin ymin xmax ymax')
        """
        #Compute new coordinates inside a crop.
        relative_x = point['x']-crop_rectangle.xmin
        relative_y =  point['y'] - crop_rectangle.ymin
        return {'x':relative_x,'y':relative_y}
    
    def find_relative_bounding_box_coordinates(self,nutrition_table_bounding_box):
        """Compute relative coordinates for each point of word's bounding box inside the nutritional table. 
        
        Keyword arguments:
        nutrition_table_bounding_box -- named tuple Rectangle('Rectangle', 'xmin ymin xmax ymax')
        """
        #Compute new coordinates inside a crop.
        self.relative_bounding_box = [self.find_relative_coordinates_in_crop_rectangle(x,nutrition_table_bounding_box)
                                      for x in self.bounding_box]
        
    

    def get_angle(self):
        
        
        
        """Compute rotation angle α of the word's bounding box
        

        
               ^                            
               |         ▓                  
               |        ▓ ▓                 
               |       ▓ ▓ ▓                
               |      ▓ ▓   ▓               
               |     ▓ ▓     ▓              
               |    ▓ ▓       ▓             
               |   ▓ ▓         ▓            
               |  ▓ ▓           ▓           
               | ▓ ▓             ▓          
               |  ▓               ▓         
               |   ▓               ▓        
               |    ▓               ▓       
               |     ▓               ▓      
               |      ▓               ▓     
               |       ▓               ▓    
               |        ▓               ▓   
               |         ▓               ▓  
               |          ▓               ▓ 
               |           ▓             ▓ ▓
               |            ▓           ▓ ▓ 
               |             ▓         ▓ ▓  
               |            _ ▓       ▓ ▓   
               |          /    ▓     ▓ ▓    
               |         /      ▓   ▓ ▓     
               |        /        ▓ ▓ ▓      
               |       /   α      ▓ ▓       
               |       |           ▓        
               |----------------------------------------------------->
        
        """
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
        """Compute new word's bounding box coordinates after rotation. 
        """
        #self.bounding_box = [rotate_point(x,center,angle) for x in self.bounding_box]
        self.relative_bounding_box = [rotate_point(x,center,angle) for x in self.relative_bounding_box]