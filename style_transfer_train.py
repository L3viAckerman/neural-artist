import os

os.environ["CUDA_VISIBLE_DEVICES"]="0"
import sys
import time
import argparse
import functools

import numpy as np
import tensorflow as tf
from tensorflow.python.tools.freeze_graph import freeze_graph

from utils import *

from model import *

STYLE_LAYERS = ('relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1')
CONTENT_LAYER = 'relu4_2'

# Parameters
learning_rate = 0.001
epochs = 1
batch_size = 2
show = 200  
save_every_num_iter = 400  

# weight for loss
content_weight , style_weight, tv_weight = 5e1, 1e2, 2e2

def build_parser():
    parser = argparse.ArgumentParser()


    parser.add_argument(
        '--checkpoint_dir',
        type=str,
        default='checkpoint',
        help='Checkpoint directory')

    parser.add_argument(
        '--vgg-path',
        type=str,
        default='data/imagenet-vgg-verydeep-19.mat',
        help='Path to VGG19 network (default %(default)s)')

    parser.add_argument(
        '--checkpoint_path',
        type=str,
        default='checkpoint/ckpt_i0',
        help='Checkpoint directory')

    parser.add_argument(
        '--style',
        type=str,
        default='./images/styles',
        help='Style image path')

    parser.add_argument(
        '--train-path',
        type=str,
        default='data/train2014',
        help='Path to training images folder')

    parser.add_argument(
        '--test-dir',
        type=str,
        default='images/results',  
        help='Test image save dir')

    parser.add_argument(
        '--test',
        type=str,
        default='images/test/test.jpeg', 
        help='Test image path')

    parser.add_argument('--light_version', type=bool, default=False, help='Optional for light version of transform_net network')

    parser.add_argument('--model_dir', type=str, default='model', help='Model directory')   
    parser.add_argument('--model_name', type=str, default='style_graph', help='Model name')


    return parser


def _tensor_size(tensor):
    from operator import mul
    return functools.reduce(mul, (d.value for d in tensor.get_shape()[1:]), 1)

def evaluate(img_in, img_path, ckpt, name, light_version):
    img_shape = (512, 512, 3)
    batch_shape = (1, 512, 512, 3)

    
    if light_version == True:
        transform_net = light_transform
        img_shape = (256, 256, 3)
        batch_shape = (1,256, 256, 3)

    else:
        transform_net = transform

    soft_config = tf.ConfigProto(allow_soft_placement=True)
    soft_config.gpu_options.allow_growth = True
    print('a')
    with tf.Graph().as_default(), tf.Session(config=soft_config) as sess:

        X_inputs = tf.placeholder(
            tf.float32, shape=batch_shape, name='input')

        preds = transform_net(X_inputs)  
        tf.identity(preds[0], name='output')
        print('aa')


        saver = tf.train.Saver()
        saver.restore(sess, ckpt)

        X = np.zeros(batch_shape, dtype=np.float32)  

        img = get_img(img_in, img_shape)
        X[0] = img

        _preds = sess.run(preds, feed_dict={X_inputs: X})
        save_img(img_path, _preds[0])

        start_time = time.time()
        tf.train.write_graph(
            sess.graph.as_graph_def(),
            FLAGS.model_dir,
            FLAGS.model_name + '_' + name +'.pb',
            as_text=False)
        tf.train.write_graph(
            sess.graph.as_graph_def(),
            FLAGS.model_dir,
            FLAGS.model_name + '_' + name + '.pb.txt',
            as_text=True)
        end_time = time.time()
        ex_time = end_time - start_time
        print('Save file graph pb and pb.txt done!, time:', ex_time)


def optimize(content_targets, style_target, content_weight, style_weight,
             tv_weight, vgg_path, name, light_version):

    mod = len(content_targets) % batch_size
    if (mod > 0):
        content_targets = content_targets[:-mod]

    if light_version == True:
        transform_net = light_transform
    else:
        transform_net = transform

    batch_shape = (batch_size, 256, 256, 3)
    style_shape = (1, *style_target.shape)
    print('batch shape:', batch_shape)
    print('style shape:', style_shape)

    with tf.Graph().as_default(), tf.Session() as sess:
        style_image = tf.placeholder(
            tf.float32, shape=style_shape, name='style_image')
        X_content = tf.placeholder(
            tf.float32, shape=batch_shape, name='X_content')

        start_time = time.time()
        content_features = {}

        X_content_pre = preprocess(X_content)
        content_net = VGGnet(vgg_path, X_content_pre)
        content_features[CONTENT_LAYER] = content_net[CONTENT_LAYER]

        end_time = time.time()
        ex_time = end_time - start_time
        print('ex time for precomputing content features is:', ex_time)

        start_time = time.time()
        style_features = {}
        style_pre = np.array([style_target])  

        style_image_pre = preprocess(style_image)
        style_net = VGGnet(vgg_path, style_image_pre)
        for layer in STYLE_LAYERS:
            features = style_net[layer].eval(
                feed_dict={style_image: style_pre})
            features = np.reshape(features, (-1, features.shape[3]))
            gram = np.matmul(features.T, features) / features.size
            style_features[layer] = gram

        end_time = time.time()
        ex_time = end_time - start_time
        print('ex time for precompute style features is:', ex_time)


        preds = transform_net(X_content / 255.0)
        preds_pre = preprocess(preds)
        preds_net = VGGnet(vgg_path, preds_pre)


        start_time = time.time()
        content_size = _tensor_size(
            content_features[CONTENT_LAYER]) * batch_size
        assert _tensor_size(content_features[CONTENT_LAYER]) == _tensor_size(
            preds_net[CONTENT_LAYER])
        content_loss = content_weight * (2 * tf.nn.l2_loss(
            preds_net[CONTENT_LAYER] - content_features[CONTENT_LAYER]
        ) / content_size)
        end_time = time.time()
        ex_time = end_time - start_time
        print('ex time for computing content loss is:', ex_time)


        start_time = time.time()
        style_losses = []
        for style_layer in STYLE_LAYERS:
            layer = preds_net[style_layer]
            bs, height, width, filters = map(lambda i: i.value,
                                             layer.get_shape())
            size = height * width * filters
            feats = tf.reshape(layer, (bs, height * width, filters))
            feats_T = tf.transpose(feats, perm=[0, 2, 1])
            grams = tf.matmul(feats_T, feats) / size
            style_gram = style_features[style_layer]
            style_losses.append(
                2 * tf.nn.l2_loss(grams - style_gram) / style_gram.size)
        style_loss = style_weight * functools.reduce(tf.add,
                                                     style_losses) / batch_size
        end_time = time.time()
        ex_time = end_time - start_time
        print('ex time for compute style loss is:', ex_time)

        start_time = time.time()
        tv_y_size = _tensor_size(preds[:, 1:, :, :])
        tv_x_size = _tensor_size(preds[:, :, 1:, :])
        y_tv = tf.nn.l2_loss(preds[:, 1:, :, :] -
                             preds[:, :batch_shape[1] - 1, :, :])
        x_tv = tf.nn.l2_loss(preds[:, :, 1:, :] -
                             preds[:, :, :batch_shape[2] - 1, :])
        tv_loss = tv_weight * 2 * (
            x_tv / tv_x_size + y_tv / tv_y_size) / batch_size
        end_time = time.time()
        ex_time = end_time - start_time
        print('total variation denoising time is:', ex_time)


        start_time = time.time()
        all_loss = content_loss + style_loss + tv_loss
        end_time = time.time()
        ex_time = end_time - start_time
        print('time for computing overall loss is:', ex_time)


        train = tf.train.AdamOptimizer(learning_rate).minimize(all_loss)

        sess.run(tf.global_variables_initializer())

        print('Training begin!')
        start_time = time.time()

        num_examples = len(content_targets)
        n_batches = num_examples // batch_size
        iterations = n_batches * epochs


        saver = tf.train.Saver()

        for epoch in range(epochs):
            for batch in range(n_batches):
                iteration = epoch * n_batches + batch + 1

                curr = batch * batch_size
                step = curr + batch_size

                X_batch = np.zeros(batch_shape, dtype=np.float32) 
                for i, img_p in enumerate(content_targets[curr:step]):
                    X_batch[i] = get_img(img_p, (256, 256,
                                                 3)).astype(np.float32)

                assert X_batch.shape[0] == batch_size
                sess.run(train, feed_dict={X_content: X_batch})

                to_get = [style_loss, content_loss, tv_loss, all_loss, preds]

                if (iteration % show == 0):
                    tup = sess.run(to_get, feed_dict={X_content: X_batch})
                    _style_loss, _content_loss, _tv_loss, _all_loss, _preds = tup
                    losses = (_style_loss, _content_loss, _tv_loss, _all_loss)
                    print(
                        'Iteration {}/{} - style loss: {:.3f}, content loss: {:.3f}, tv loss: {:.3f}, all loss: {:.3f}'.
                        format(iteration, iterations, *losses))
                if (iteration % save_every_num_iter == 0) or (
                        iteration == iterations):
                    _all_loss = sess.run(
                        all_loss, feed_dict={X_content: X_batch})
                    ckpt = saver.save(
                        sess,
                        os.path.join(FLAGS.checkpoint_dir,
                                     name + '_' + "ckpt_i{}".format(iteration)))
                                     
                    print('Epoch {}/{}, Iteration: {}/{}, loss: {}'.format(
                        epoch, epochs, iteration, iterations, _all_loss))
                    yield (epoch, iteration, ckpt)

        end_time = time.time()
        ex_time = end_time - start_time
        print('Train total time:', ex_time)


def do_all(_):
    if tf.gfile.Exists(FLAGS.checkpoint_dir):
        tf.gfile.DeleteRecursively(FLAGS.checkpoint_dir)
    tf.gfile.MakeDirs(FLAGS.checkpoint_dir)
    if tf.gfile.Exists(FLAGS.model_dir):
        tf.gfile.DeleteRecursively(FLAGS.model_dir)
    tf.gfile.MakeDirs(FLAGS.model_dir)

    content_targets = get_files(FLAGS.train_path)


    print(FLAGS.style)
    print(list_files(FLAGS.style))
    for name in get_files(FLAGS.style):
        print(name)

        style_target = get_img(name)
        name = name.split('/')[len(name.split('/'))-1]
        name = name[:len(name)-4]
        print(name)

        for epoch, iteration, ckpt in optimize(
                content_targets, style_target, content_weight,
                style_weight, tv_weight, FLAGS.vgg_path, name, FLAGS.light_version):
            if (FLAGS.test):
                assert FLAGS.test_dir is not False
                preds_img_name = "{}_{}.png".format(epoch, iteration)
                preds_img_path = os.path.join(FLAGS.test_dir, preds_img_name)
                evaluate(FLAGS.test, preds_img_path, ckpt, name, FLAGS.light_version)

        start_time = time.time()
        freeze_graph(
            input_graph=os.path.join(FLAGS.model_dir,
                                    FLAGS.model_name + '_' + name + '.pb.txt'),
            input_saver='',
            input_binary=False,
            input_checkpoint=ckpt,
            output_node_names='output',
            restore_op_name='save/restore_all',
            filename_tensor_name='save/Const:0',
            output_graph=os.path.join(FLAGS.model_dir,
                                    '%s_frozen.pb' % (FLAGS.model_name + '_' + name)),
            clear_devices=False,
            initializer_nodes='')
        end_time = time.time()
        ex_time = end_time - start_time
        print('Save file graph frozen pb done!, time:', ex_time)


def freeze_model(_):
    evaluate(FLAGS.test,
                 os.path.join(FLAGS.test_dir, 'res_2.jpg'),
                 FLAGS.checkpoint_path, name, FLAGS.light_version)

    start_time = time.time()
    freeze_graph(
        input_graph=os.path.join(FLAGS.model_dir,
                                 FLAGS.model_name + '.pb.txt'),
        input_saver='',
        input_binary=False,
        input_checkpoint=FLAGS.checkpoint_path,
        output_node_names='output',
        restore_op_name='save/restore_all',
        filename_tensor_name='save/Const:0',
        output_graph=os.path.join(FLAGS.model_dir,
                                  '%s_frozen.pb' % FLAGS.model_name),
        clear_devices=False,
        initializer_nodes='')
    end_time = time.time()
    ex_time = end_time - start_time
    print('Save file graph frozen pb done!, time:', ex_time)


if __name__ == '__main__':
    parser = build_parser()
    FLAGS, unparsed = parser.parse_known_args()

    tf.app.run(main=do_all, argv=[sys.argv[0]] + unparsed)
