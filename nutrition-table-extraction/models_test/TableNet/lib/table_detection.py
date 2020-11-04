import tensorflow as tf

import cv2
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
import imutils
from utils import *
from scipy.misc import electrocardiogram
from scipy.signal import find_peaks
import itertools
from sklearn.preprocessing import Normalizer
from collections import namedtuple
Rectangle = namedtuple('Rectangle', 'xmin ymin xmax ymax')

def get_mask_from_bounding_box(bounding_box_coordinates,shape):
    """Get a mask of detected table from detected table bounding box coordinates.

    Keyword arguments:
    bounding_box_coordinates -- Rectangle coordinates with first point coordinates, width, height (x,y,w,h)
    shape -- image shape
    """
    #unwrap bouding box coordinates
    x,y,w,h = bounding_box_coordinates
    #create blank image with corresponding shape
    blank_image = np.zeros(shape, np.uint8)
    #create corrected mask
    corrected_mask = cv2.rectangle(corrected_mask,(x,y),(x+w,y+h),(255,255,255),-1)
    return corrected_mask

def get_biggest_gap_index(elements_list):
    """Compute difference between element and next element in a list and return elements where difference is maximum.
    For instance if we take the list [1,2,3,4,10,12,15], we will compute the list [1,1,1,6,2,3] 
    and the function will return (4,10) representing the maximum gap.

    Keyword arguments:
    elements_list -- list of integers
    """
    #compute list of difference between element and next element
    steps = [x-y for y,x in zip(elements_list,elements_list[1:])]
    #Get index where element has biggest gap with next element
    index_where_biggest_gap = np.where(steps==max(steps))[0][0]
    #return element and next element
    return elements_list[index_where_biggest_gap], elements_list[index_where_biggest_gap+1]


def get_sub_mask_by_removing_overfilled_borders(mask,axis):
    """ Compute sum of a matrix (mask) following an axis, get indexes where sum is higher than 80% of the maximum 
    then find biggest submatrix within detected borders. If axis=1, columns will be removed (else, lines will be removed)

    Keyword arguments:
    mask -- binary image
    axis -- integer (0 or 1)
    """
    #Compute sum over the axis
    summed_on_axis = mask.sum(axis=axis)
    #Get maximum value
    maximum_value = summed_on_axis.max()
    #Find lines or columns where sum is over 80% of maximum sum.
    indexes = np.where(summed_on_axis>maximum_value*0.8)[0]
    #Use get_biggest_gap_index to get biggest submatrix within matrix by setting excluded elements to 0
    #
    #               ______________ ________
    #               _______ ____ __________
    #               _______________________
    #           --> 
    # Detected |
    # Submatrix|
    #           --> ______ ______ _________
    #               __ _______________ ____
    #
    #
    start, end = get_biggest_gap_index(indexes)
    if axis == 1:
        mask[:start]=0
        mask[end:] = 0
    elif axis == 0:
        mask[:, :start]=0
        mask[:, end:] = 0
    return mask


def process_column_mask(mask):
    
    #close horizontal thin lines
    kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT,(5,2))
    close = cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernel1)
    
    #get the inverse of the 
    result = (255*(close<128)).astype(np.uint8)
    
    #dilate then erode to connect "broken lines"
    kernel = np.ones((20,1), np.uint8)  # note this is a horizontal kernel
    d_im = cv2.dilate(result, kernel, iterations=1)
    e_im = cv2.erode(d_im, kernel, iterations=1)
    
    #overfilled lines borders are removed
    processed_mask = get_sub_mask_by_removing_overfilled_borders(e_im,axis=1)
    
    return processed_mask

def process_line_mask(mask):
    
    #close vertical thin lines
    kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT,(2,5))
    close = cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernel1)
    
    #get the inverse of the 
    result = (255*(close<128)).astype(np.uint8)
    
    #dilate then erode to connect "broken lines"
    kernel = np.ones((1,20), np.uint8)  # note this is a horizontal kernel
    d_im = cv2.dilate(result, kernel, iterations=1)
    e_im = cv2.erode(d_im, kernel, iterations=1)
    
    #Get smoother lines
    kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT,(20,5))
    close = cv2.morphologyEx(e_im,cv2.MORPH_OPEN,kernel1)
    
    #Get inverse of the mask, connect lines (which correspond to connect empty space in masks within a line) and apply in on mask to get smoother lines
    inverse_mask = (255-close)
    kernel = np.ones((1,100), np.uint8)  # note this is a horizontal kernel
    d_im = cv2.dilate(inverse_mask, kernel, iterations=1)
    e_im = cv2.erode(d_im, kernel, iterations=1)
    
    #apply it on mask
    close[e_im==255]=0
    
    #overfilled lines borders are removed
    processed_mask = get_sub_mask_by_removing_overfilled_borders(close,axis=0)
    
    return close

class TableMask:
    
    def __init__(self,image,predicted_mask):
        self.image = image
        self.predicted_mask = predicted_mask
        self.original_image_shape = (image.shape[1],image.shape[0])
        self.corrected_mask = None
        self.resized_bounding_box = None
        self.original_bounding_box = None
        self.image_with_bounding_box = None
        self.cropped_table = None
    
    @staticmethod
    def get_coef(myimage):
        shape = myimage.shape
        return myimage.sum()/(shape[0]*shape[1])
    
    @staticmethod
    def process_number(x):
        n=round(x)
        if n%2==1:
            return n
        else:
            return n+1

    def get_predicted_area_ratio(self):
        shape = self.corrected_mask.shape
        return (self.corrected_mask==255).sum()/(shape[0]*shape[1]*3)
    
    def get_bounding_box_coordinates(self):
        se1 = cv2.getStructuringElement(cv2.MORPH_RECT, (30,30))
        mask1 = cv2.morphologyEx(self.predicted_mask.copy(), cv2.MORPH_OPEN, se1)
        se2 = np.ones((10,10),np.uint8)
        mask2  = cv2.dilate(mask1, se2)
        mask = (1*(mask2==255)).astype('uint8')
        out = np.logical_and(self.predicted_mask.copy(),mask).astype('uint8')
        se3 = cv2.getStructuringElement(cv2.MORPH_RECT, (10,10))
        cleaned_out = cv2.morphologyEx(out, cv2.MORPH_OPEN, se1)
        cnts = cv2.findContours(cleaned_out.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        try:
            c = max(cnts, key=cv2.contourArea)
            rect = cv2.boundingRect(c)
            x,y,w,h = rect
            self.resized_bounding_box = rect
        except:
            self.resized_bounding_box = (0,0,0,0)

    def generate_corrected_mask(self):
        self.corrected_mask = cv2.resize(get_mask_from_bounding_box(self.resized_bounding_box,(256,256,3)),self.original_image_shape)
    
    def get_bounding_box_coordinates_on_original_image(self):
        point1_org = (np.where(self.corrected_mask==255)[1][0] , np.where(self.corrected_mask==255)[0][0])
        point2_org = (np.where(self.corrected_mask==255)[1][-1] , np.where(self.corrected_mask==255)[0][-1])
        self.original_bounding_box = (point1_org[0],point1_org[1],point2_org[0]-point1_org[0],point2_org[1]-point1_org[1])
    
    def draw_bounding_box(self):
        x,y,w,h = self.original_bounding_box
        self.image_with_bounding_box = cv2.rectangle(self.image.copy(),(x,y),(x+w,y+h),(0,255,0),10)
        
        
    def get_cropped_image(self):
        x,y,w,h = self.original_bounding_box
        return self.image[y:y+h,x:x+w]
    
    
class Pipeline:
    def __init__(self,model,image):
        self.model = model
        self.images = [image]
        self.tm_list = []
        self.predicted_table_masks = []
        self.predicted_column_masks = []
        self.predicted_table_mask = None
        self.predicted_column_mask = None
        self.predicted_line_mask = None
    
    @staticmethod
    def iou(gt_mask,pred_mask):
        intersection = np.logical_and(gt_mask,pred_mask).sum()
        union = np.logical_or(gt_mask,pred_mask).sum()
        return intersection/union
    
    def predict(self):
        img_tensor, shape = get_image_tensor_from_image_array(tf.convert_to_tensor(self.images[-1]))
        image = tf.expand_dims(img_tensor,axis=0)
        predicted_table_mask, predicted_column_mask = get_masks(self.model,image)
        self.predicted_table_masks.append(255*predicted_table_mask.numpy().reshape(256,256))
        self.predicted_column_masks.append(255*predicted_column_mask.numpy().reshape(256,256))
    
    def process_prediction(self):
        bw_mask = self.predicted_table_masks[-1].reshape(256,256)
        self.tm_list.append(TableMask(np.uint8(self.images[-1]),np.uint8(bw_mask)))
        self.tm_list[-1].get_bounding_box_coordinates()
        self.tm_list[-1].generate_corrected_mask()
        self.tm_list[-1].get_bounding_box_coordinates_on_original_image()
        
    def get_predicted_area_ratio(self):
        image = cv2.cvtColor(self.tm_list[-1].corrected_mask,cv2.COLOR_RGB2GRAY)
        shape = image.shape
        return (image==255).sum()/(shape[0]*shape[1])
    
    def predicted_area(self):
        return self.tm_list[-1].get_predicted_area_ratio()
    
    def run(self, pred_area_threshold, loop_until_converge=True):
        while True:
            self.predict()
            self.process_prediction()
            if self.predicted_area()>pred_area_threshold or loop_until_converge==False:
                #Get Line prediction on rotated image
                img_tensor, shape = get_image_tensor_from_image_array(cv2.rotate(self.images[-1].copy(),cv2.ROTATE_90_COUNTERCLOCKWISE))
                image = tf.expand_dims(img_tensor,axis=0)
                _ , predicted_rotated_column_mask = get_masks(self.model,image)
                self.predicted_line_mask = 255*cv2.rotate(predicted_rotated_column_mask.numpy().reshape(256,256),cv2.ROTATE_90_CLOCKWISE)
                self.predicted_column_mask = self.predicted_column_masks[-1]
                break
            else:
                self.images.append(self.tm_list[-1].get_cropped_image())
        
                
    def get_final_prediction_bbx(self):
        rect_list = [tm.original_bounding_box for tm in self.tm_list]
        bbx = rect_list[0]
        x,y,w,h = bbx
        for i in range(1,len(self.tm_list)):
            bbx = get_absolute_coordinates(bbx,rect_list[i])
        return bbx
    
    def get_predicted_table_mask(self):
        bbx = self.get_final_prediction_bbx()
        self.predicted_table_mask = get_mask_from_bounding_box(bbx,self.images[0].shape)
                

class Signal:
    def __init__(self, signal):
        self.signal=signal
        
    @staticmethod   
    def normalize(s):
        normalizer = Normalizer()
        normalizer.fit_transform([s])[0]
        
        return 
    
    def process_signal(self):
        normalized_signal = self.normalize(self.signal)
        return normalized_signal
    
    def find_peaks(self,method='std',distance=None):
        processed_signal = self.process_signal()
        processed_signal = np.insert(self.signal,0,0) #add a 0 to consider first peak
        
        if method=='std':
            prominence = self.signal.std()
        elif method=='variance':
            prominence = self.signal.std()**2
            
        peaks, _ = find_peaks(self.signal, prominence=prominence, distance = distance)
        peaks = peaks[:-1] #Remove last peak as it is not significant
        
        return peaks
              
class LineDetection:
    
    def __init__(self,mask,original_image_shape):
        self.original_image_shape = original_image_shape
        self.mask=mask
        self.signal = None
        self.peaks = None
        self.signal_peaks = None
    
    @staticmethod
    def moving_average(a, n) :
        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1:] / n
    
    @staticmethod
    def runningMean(x, N):
        y = np.zeros((len(x),))
        for ctr in range(len(x)):
             y[ctr] = np.sum(x[ctr:(ctr+N)])
        return y/N
    
    
    def find_peaks(self, window_size, distance,axis, method='std'):
        print(method)
        vertical_sum = self.mask.sum(axis=axis)
        signal = self.runningMean(vertical_sum,window_size)
        self.signal = Signal(signal)
        self.signal_peaks = self.signal.find_peaks(method=method,distance=distance)
        # = peaks[:-1] #Remove last peak as it is not significant
        corrected_peaks = [peak+(window_size//2) for peak in self.signal_peaks]
        self.peaks = [x*self.original_image_shape[1-axis]//256 for x in corrected_peaks]
        

class Table:
    
    def __init__(self, column_mask, line_mask, original_image_shape):
        self.column_mask = column_mask
        self.line_mask = line_mask
        self.original_image_shape = original_image_shape
        self.column_detect = None
        self.line_detect = None
        self.columns = None
        self.lines = None
        self.table = []
        
        
    def find_columns(self):
        preprocessed_column_mask = process_column_mask(self.column_mask.astype(np.uint8))
        line_detect = LineDetection(preprocessed_column_mask, self.original_image_shape)
        line_detect.find_peaks(window_size = 25, distance = 10, axis = 0)
        self.columns = line_detect.peaks
        self.columns.insert(0,0)
        self.columns.append(self.original_image_shape[1])
        
    def find_lines(self):
        preprocessed_line_mask = process_line_mask(self.line_mask.astype(np.uint8))
        line_detect = LineDetection(preprocessed_line_mask, self.original_image_shape)
        line_detect.find_peaks(window_size = 10, distance = 5,method='std', axis = 1)
        self.lines = line_detect.peaks
        self.lines.insert(0,0)
        self.lines.append(self.original_image_shape[0])
        
    
    def find_table(self):
        
        
        self.find_columns()
        self.find_lines()
        points = np.array(list(itertools.product(self.columns,self.lines)))
        points = points.reshape(len(self.columns),len(self.lines),2)
        
        for i in range(1,len(self.columns)):
            line = []
            for j in range(1,len(self.lines)):
                point1 = points[i-1][j-1]
                point2 = points[i][j]
                line.append(Rectangle(point1[0],point1[1],point2[0],point2[1]))
            self.table.append(line)
    