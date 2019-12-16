from __future__ import print_function
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import numpy as np
import cv2	
import tensorflow as tf	 
import time
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--test_image', default='images/test/test.jpeg', help='Name of the model')
    parser.add_argument('-m', '--model_name', default='model/graph.pb', help='Path to the model file')
    parser.set_defaults(is_debug=False)
    return parser.parse_args()

def main(args):
	img = cv2.imread(args.test_image)

	with tf.Graph().as_default():
		output_graph_path = args.model_name
		with tf.gfile.FastGFile(output_graph_path, 'rb') as f:
			print(output_graph_path)
			graph_def = tf.GraphDef() 
			graph_def.ParseFromString(f.read())
			_ = tf.import_graph_def(graph_def, name='') 
	
		with tf.Session() as sess:
			tf.initialize_all_variables().run()
			input_name = graph_def.node[0].name
			print('Input node name: ', input_name)
			output_name = graph_def.node[-1].name
			print('Output node name: ', output_name)
			input_x = sess.graph.get_tensor_by_name(str(input_name)+":0")
			print(input_x)
			output = sess.graph.get_tensor_by_name(str(output_name)+":0")
			print(output)
			generated = output

			start_time = time.time()
			shape_org = img.shape[:2]
			print(shape_org)
			
			img_ = cv2.resize(img,(512,512))
			X = np.zeros((1,512,512,3),dtype=np.float32)
			X[0] = img_
			 
			print('Input shape: ', X.shape)
			image_transfer = sess.run(generated, feed_dict={input_x: X})

			image_transfer = (image_transfer - image_transfer.min())/(image_transfer.max()-image_transfer.min())
			image_transfer = (image_transfer * 255).astype('uint8')
			image_transfer = cv2.resize(image_transfer, shape_org[::-1])
			
			print('Output shape: ', image_transfer.shape)
			cv2.imwrite('transfer_img.png', image_transfer)
			cv2.imshow('origin img', img)
			cv2.imshow('transfer img', image_transfer)
			
			end_time = time.time()
			time_spend = end_time - start_time
			print('Time cost: ', time_spend)
			key = cv2.waitKey(0)  
			if key == 'q':  
				cv2.destroyAllWindows()

if __name__ == '__main__':
    args = parse_args()
    print(args)
    main(args)
				 

