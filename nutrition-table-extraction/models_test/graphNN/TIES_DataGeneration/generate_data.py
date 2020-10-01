from TFGeneration.GenerateTFRecord import *
import argparse


parser=argparse.ArgumentParser()
parser.add_argument('--filesize',type=int,default=1)                #number of images in a single tfrecord file
parser.add_argument('--threads',type=int,default=1)                 #one thread will work on one tfrecord
parser.add_argument('--outpath',default='tfrecords/')               #directory to store tfrecords

#imagespath,
parser.add_argument('--imagespath',default='../Table_Detection_Dataset/unlv/train/images')
parser.add_argument('--ocrpath',default='../Table_Detection_Dataset/unlv/unlv_xml_ocr')
parser.add_argument('--tablepath',default='../Table_Detection_Dataset/unlv/unlv _xml_gt')

parser.add_argument('--visualizeimgs',type=int,default=0)              #if 1, will store the images along with tfrecords
parser.add_argument('--visualizebboxes',type=int,default=0)			#if 1, will store the bbox visualizations in visualizations folder
args=parser.parse_args()

filesize=max(int(args.filesize),4)
visualizeimgs=False
if(args.visualizeimgs==1):
    visualizeimgs=True

visualizebboxes=False
if(args.visualizebboxes==1):
	visualizebboxes=True

distributionfile='unlv_distribution'

t = GenerateTFRecord(args.outpath,filesize,args.imagespath,
                     args.ocrpath,args.tablepath,visualizeimgs,visualizebboxes,distributionfile)
t.write_to_tf(args.threads)






