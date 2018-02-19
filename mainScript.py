import numpy as np
import cv2
import time
import os
import caffe
import time
import io

camera_port = 0
 
#Number of frames to throw away while the camera adjusts to light levels
ramp_frames = 30
 
# Now we can initialize the camera capture object with the cv2.VideoCapture class.
# All it needs is the index to a camera port.
camera = cv2.VideoCapture(camera_port)

def get_image():
  # read is the easiest way to get a full image out of a VideoCapture object.
  retval, im = camera.read()
  return im

def captureAndLabel(): 
	# Ramp the camera - these frames will be discarded and are only used to allow v4l2
	# to adjust light levels, if necessary
	for i in xrange(ramp_frames):
	 temp = get_image()
	print("Taking image...")
	# Take the actual image we want to keep
	camera_capture = get_image()
	file = "./data/test.png"
	# A nice feature of the imwrite method is that it will automatically choose the
	# correct format based on the file extension you provide. Convenient!
	cv2.imwrite(file, camera_capture)
	 
	# You'll want to release the camera, otherwise you won't be able to create a new
	# capture object until your script exit
	camera.release()
	cv2.destroyAllWindows()

	#Labelling all the images
	MODEL_FILE = '/media/rajat/Miscellaneous/Mtech/MTP/CNN/caffe/models/bvlc_reference_caffenet/deploy.prototxt'
	PRETRAINED = '/media/rajat/Miscellaneous/Mtech/MTP/CNN/caffe/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'

	net = caffe.Classifier(MODEL_FILE, PRETRAINED,
		               mean=np.load('/media/rajat/Miscellaneous/Mtech/MTP/CNN/caffe/python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1),
		               channel_swap=(2,1,0),
		               raw_scale=255,
		               image_dims=(256, 256))
	print "successfully loaded classifier"
	IMAGE_FILE = './data/'

	path = IMAGE_FILE + 'test.png'
	print path
	input_image = caffe.io.load_image(path)
	pred = net.predict([input_image])
	labels = np.loadtxt("/media/rajat/Miscellaneous/Mtech/MTP/CNN/caffe/data/ilsvrc12/synset_words.txt", str, delimiter='\t')
	labels = np.loadtxt("/media/rajat/Miscellaneous/Mtech/MTP/CNN/caffe/data/ilsvrc12/synset_words.txt", str, delimiter='\t')
	#/home/ubuntu/caffe/models/bvlc_reference_caffenet
	top_k = pred.argsort()[:, -1:-6:-1]
	#print top_k 

	#print len(labels[top_k][0]);
	final_labels  = ''
	#print labels[top_k][0][0];
	for i in range(0, len(labels[top_k][0])):
	  raw = labels[top_k][0][i].split(' ')
	  if(i != len(labels[top_k][0]) - 1):
	    final_labels = final_labels + labels[top_k][0][i].replace(raw[0], '') + ','
	  else:
	    final_labels = final_labels + labels[top_k][0][i].replace(raw[0], '') + '.'
	final_string =  "The visual may contain "+final_labels;    
	with io.open('./labels.txt', 'w', encoding='utf-8') as text:
	       text.write(unicode(final_string))
	os.system("trans :en file://labels.txt -p -no-auto")
	os.system("trans :hi file://labels.txt -p -no-auto")


captureAndLabel()
time.sleep(5)
captureAndLabel()
time.sleep(5)
captureAndLabel()
time.sleep(5)
captureAndLabel()
captureAndLabel()
