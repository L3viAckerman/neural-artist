import tensorflow as tf
import numpy as np
import scipy.io

MEAN_PIXEL = np.array([123.68, 116.779, 103.939])

WEIGHTS_INIT_STDEV = .1

# Transform net
def transform(image):
    conv1 = _conv_layer_transform(image, 32, 9, 1)
    conv2 = _conv_layer_transform(conv1, 64, 3, 2)
    conv3 = _conv_layer_transform(conv2, 128, 3, 2)
    resid1 = _residual_block(conv3, 3)
    resid2 = _residual_block(resid1, 3)
    resid3 = _residual_block(resid2, 3)
    resid4 = _residual_block(resid3, 3)
    resid5 = _residual_block(resid4, 3)
    conv_t1 = _conv_tranpose_layer(resid5, 64, 3, 2)
    conv_t2 = _conv_tranpose_layer(conv_t1, 32, 3, 2)
    conv_t3 = _conv_layer_transform(conv_t2, 3, 9, 1, relu=False)
    preds = tf.nn.tanh(conv_t3) * 150 + 255. / 2
    return preds

# Light version of transform
def light_transform(image):
    conv1 = _conv_layer_transform(image, 32, 9, 1)
    conv2 = _conv_layer_transform(conv1, 64, 3, 2)
    conv3 = _conv_layer_transform(conv2, 128, 3, 2)
    resid1 = _residual_block(conv3, 3)
    resid2 = _residual_block(resid1, 3)
    resid3 = _residual_block(resid2, 3)
    conv_t1 = _conv_tranpose_layer(resid3, 32, 3, 2)
    conv_t2 = _conv_tranpose_layer(conv_t1, 32, 3, 2)
    conv_t3 = _conv_layer_transform(conv_t2, 3, 9, 1, relu=False)
    preds = tf.nn.tanh(conv_t3) * 150 + 255. / 2
    return preds

def _conv_tranpose_layer(net, num_filters, filter_size, strides):
    weights_init = _conv_init_vars(
        net, num_filters, filter_size, transpose=True)

    batch_size, rows, cols, in_channels = [i.value for i in net.get_shape()]
    new_rows, new_cols = int(rows * strides), int(cols * strides)

    new_shape = [batch_size, new_rows, new_cols, num_filters]
    tf_shape = tf.stack(new_shape)
    strides_shape = [1, strides, strides, 1]

    net = tf.nn.conv2d_transpose(
        net, weights_init, tf_shape, strides_shape, padding='SAME')
    net = _instance_norm(net)
    return tf.nn.relu(net)


def _conv_layer_transform(net, num_filters, filter_size, strides, relu=True):
    weights_init = _conv_init_vars(net, num_filters, filter_size)
    strides_shape = [1, strides, strides, 1]
    net = tf.nn.conv2d(net, weights_init, strides_shape, padding='SAME')
    net = _instance_norm(net)
    if relu:
        net = tf.nn.relu(net)
    return net

def _residual_block_light(net, filter_size=3):
    tmp = _conv_layer_transform(net, 64, filter_size, 1)
    return net + _conv_layer_transform(tmp, 64, filter_size, 1, relu=False)

def _residual_block(net, filter_size=3):
    tmp = _conv_layer_transform(net, 128, filter_size, 1)
    return net + _conv_layer_transform(tmp, 128, filter_size, 1, relu=False)


def _instance_norm(net, train=True):
    batch, rows, cols, channels = [i.value for i in net.get_shape()]
    var_shape = [channels]
    mu, sigma_sq = tf.nn.moments(net, [1, 2], keep_dims=True)
    shift = tf.Variable(tf.zeros(var_shape))
    scale = tf.Variable(tf.ones(var_shape))
    epsilon = 1e-3
    normalized = (net - mu) / (sigma_sq + epsilon)**(.5)
    return scale * normalized + shift


def _conv_init_vars(net, out_channels, filter_size, transpose=False):
    _, rows, cols, in_channels = [i.value for i in net.get_shape()]
    if not transpose:
        weights_shape = [filter_size, filter_size, in_channels, out_channels]
    else:
        weights_shape = [filter_size, filter_size, out_channels, in_channels]

    weights_init = tf.Variable(
        tf.truncated_normal(weights_shape, stddev=WEIGHTS_INIT_STDEV, seed=1),
        dtype=tf.float32)
    return weights_init


# VGG19 network
def VGGnet(data_path, input_image):
	layers = ('conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1', 'conv2_1',
	      'relu2_1', 'conv2_2', 'relu2_2', 'pool2', 'conv3_1', 'relu3_1',
	      'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'conv3_4', 'relu3_4',
	      'pool3', 'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
	      'relu4_3', 'conv4_4', 'relu4_4', 'pool4', 'conv5_1', 'relu5_1',
	      'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3', 'conv5_4', 'relu5_4')

	data = scipy.io.loadmat(data_path)

	weights = data['layers'][0]

	net = {}
	current = input_image
	for i, name in enumerate(layers):
		kind = name[:4]
		if kind == 'conv':
		    if isinstance(weights[i][0][0][0][0],np.ndarray):
			    kernels, bias = weights[i][0][0][0][0]	
		    else:
			    kernels, bias = weights[i][0][0][2][0]

		    kernels = np.transpose(kernels, (1, 0, 2, 3))
		    bias = bias.reshape(-1)
		    current = _conv_layer(current, kernels, bias)
		elif kind == 'relu':
		    current = tf.nn.relu(current)
		elif kind == 'pool':
		    current = _pool_layer(current)
		net[name] = current

	assert len(net) == len(layers)
	return net


def _conv_layer(input, weights, bias):
    conv = tf.nn.conv2d(
        input, tf.constant(weights), strides=(1, 1, 1, 1), padding='SAME')
    return tf.nn.bias_add(conv, bias)

def _pool_layer(input):
    return tf.nn.max_pool(
        input, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME')

def unprocess(image):
    return image + MEAN_PIXEL

def preprocess(image):
    return image - MEAN_PIXEL



