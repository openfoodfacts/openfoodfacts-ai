import cv2
import pytesseract
import numpy as np
import copy

class ImageUtils:
    """ Class for image preprocessing tools"""

    def getTextLen(self, processImage, angle):
        """Takes an input image and rotation angle
        an integral multiple of 90 and returns
        the length of text obtained from rotated image"""
        image = copy.deepcopy(processImage)
        image = np.rot90(image, angle/90)
        text = pytesseract.image_to_string(image)
        text.replace(" ", "")
        return len(text)
    
    def rotationCorrection(self, image):
        """Corrects the orientation of an image,
        takes input an np array containing image 
        and returns the best out of any of the 
        four orientations i.e, 0,90,180,270 degrees"""
        processImage = copy.deepcopy(image)
        processImage = cv2.cvtColor(processImage, cv2.COLOR_BGR2GRAY)
        processImage = cv2.adaptiveThreshold(processImage, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 101, 25)
        rotlist = []
        rotlist.append(self.getTextLen(processImage, 0))
        rotlist.append(self.getTextLen(processImage, 90))
        rotlist.append(self.getTextLen(processImage, 180))
        rotlist.append(self.getTextLen(processImage, 270))
        text = pytesseract.image_to_string(processImage)
        rotlist = np.array(rotlist)
        angle = rotlist.argmax()*90
        if angle == 90:
            image = np.rot90(image)
        elif angle == 180:
            image = np.rot90(image, 2)
        elif angle == 270:
            image = np.rot90(image, 3)
        return image
        
