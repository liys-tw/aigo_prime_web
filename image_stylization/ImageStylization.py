# Copyright 2019 The Magenta Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Generates stylized images with different strengths of a stylization.

For each pair of the content and style images this script computes stylized
images with different strengths of stylization (interpolates between the
identity transform parameters and the style parameters for the style image) and
saves them to the given output_dir.
See run_interpolation_with_identity.sh for example usage.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import ast
import os

import image_stylization.arbitrary_image_stylization_build_model as build_model
from image_stylization import image_utils
import numpy as np
import tensorflow as tf

slim = tf.contrib.slim

""" class to load model and run inference, image stylize """
class ImageStylization(object):
    def __init__(self, image_stylization_options):
        self.image_stylization_options = image_stylization_options
        tf.logging.set_verbosity(tf.logging.INFO)
        if not tf.gfile.Exists(image_stylization_options.output_dir):
            tf.gfile.MkDir(image_stylization_options.output_dir)
        # Load image stylization model
        self.sess = tf.Session()
        # Defines place holder for the style image.
        self.style_img_ph = tf.placeholder(tf.float32, shape=[None, None, 3])
        if image_stylization_options.style_square_crop:
            self.style_img_preprocessed = image_utils.center_crop_resize_image(
              self.style_img_ph, image_stylization_options.style_image_size)
        else:
            self.style_img_preprocessed = image_utils.resize_image(self.style_img_ph,
                                                            image_stylization_options.style_image_size)

        # Defines place holder for the content image.
        self.content_img_ph = tf.placeholder(tf.float32, shape=[None, None, 3])
        if image_stylization_options.content_square_crop:
            self.content_img_preprocessed = image_utils.center_crop_resize_image(
              self.content_img_ph, image_stylization_options.image_size)
        else:
            self.content_img_preprocessed = image_utils.resize_image(
              self.content_img_ph, image_stylization_options.image_size)

        # Defines the model.
        self.stylized_images, _, _, self.bottleneck_feat = build_model.build_model(
            self.content_img_preprocessed,
            self.style_img_preprocessed,
            trainable=False,
            is_training=False,
            inception_end_point='Mixed_6e',
            style_prediction_bottleneck=100,
            adds_losses=False)

        if tf.gfile.IsDirectory(image_stylization_options.checkpoint):
            checkpoint = tf.train.latest_checkpoint(image_stylization_options.checkpoint)
        else:
            checkpoint = image_stylization_options.checkpoint
            tf.logging.info('loading latest checkpoint file: {}'.format(checkpoint))

        init_fn = slim.assign_from_checkpoint_fn(checkpoint,
                                                 slim.get_variables_to_restore())
        self.sess.run([tf.local_variables_initializer()])
        init_fn(self.sess)
        print('ImageStylization init done.')


    def stylize_image(self, style_image, content_image, interpolation_weight=1.0):
        """
        input: content image and style image with shape [height, width, 3] RGB
        output: stylized image
        """
        print('stylize_image')
        image_stylization_options = self.image_stylization_options

        inp_img_croped_resized_np = self.sess.run(self.content_img_preprocessed
            , feed_dict={self.content_img_ph: content_image})
        image_utils.save_np_image(inp_img_croped_resized_np,
            os.path.join(image_stylization_options.output_dir, 'content_image.jpg'))
        # Computes bottleneck features of the style prediction network for the identity transform.
        identity_params = self.sess.run(self.bottleneck_feat, feed_dict={self.style_img_ph: content_image})

        # Saves preprocessed style image.
        style_img_croped_resized_np = self.sess.run(self.style_img_preprocessed
            , feed_dict={self.style_img_ph: style_image})
        image_utils.save_np_image(style_img_croped_resized_np, os.path.join(image_stylization_options.output_dir, 'style_image.jpg'))

        # Computes bottleneck features of the style prediction network for the given style image.
        style_params = self.sess.run(self.bottleneck_feat, feed_dict={self.style_img_ph: style_image})


        # Interpolates between the parameters of the identity transform and style parameters of the given style image.
        stylized_image_res = self.sess.run(self.stylized_images,
            feed_dict={
            self.bottleneck_feat: identity_params * (1 - interpolation_weight) + style_params * interpolation_weight,
            self.content_img_ph: content_image})

        # Saves stylized image.
        stylized_image_path = os.path.join(image_stylization_options.output_dir, 'stylized_image_%.1f.jpg' % interpolation_weight)
        image_utils.save_np_image(stylized_image_res, stylized_image_path)

        return np.uint8(stylized_image_res * 255.0), stylized_image_path

    def stylize_images_by_path(self, style_images_paths, content_images_paths, interpolation_weight=1.0):
        """
        style_images_paths: Paths to the style images for evaluation.
        content_images_paths: Paths to the content images for evaluation.
        return: stylized images path
        """
        print('stylize_images_by_path')
        image_stylization_options = self.image_stylization_options
        stylized_images_path = []
        # Gets the list of the input style images.
        style_img_list = tf.gfile.Glob(style_images_paths)
        if len(style_img_list) > image_stylization_options.maximum_styles_to_evaluate:
            np.random.seed(1234)
            style_img_list = np.random.permutation(style_img_list)
            style_img_list = style_img_list[:image_stylization_options.maximum_styles_to_evaluate]

        # Gets list of input content images.
        content_img_list = tf.gfile.Glob(content_images_paths)

        for content_i, content_img_path in enumerate(content_img_list):
            content_img_np = image_utils.load_np_image_uint8(content_img_path)[:, :, :3]
            content_img_name = os.path.basename(content_img_path)[:-4]

            # Saves preprocessed content image.
            inp_img_croped_resized_np = self.sess.run(self.content_img_preprocessed
                , feed_dict={self.content_img_ph: content_img_np})
            image_utils.save_np_image(inp_img_croped_resized_np,
                                    os.path.join(image_stylization_options.output_dir,
                                                 '%s.jpg' % (content_img_name)))

            # Computes bottleneck features of the style prediction network for the
            # identity transform.
            identity_params = self.sess.run(self.bottleneck_feat, feed_dict={self.style_img_ph: content_img_np})

            for style_i, style_img_path in enumerate(style_img_list):
                if style_i > image_stylization_options.maximum_styles_to_evaluate:
                    break
                style_img_name = os.path.basename(style_img_path)[:-4]
                style_image_np = image_utils.load_np_image_uint8(style_img_path)[:, :, :3]

                if style_i % 10 == 0:
                    tf.logging.info('Stylizing (%d) %s with (%d) %s' %
                              (content_i, content_img_name, style_i,
                               style_img_name))

                # Saves preprocessed style image.
                style_img_croped_resized_np = self.sess.run(self.style_img_preprocessed
                    , feed_dict={self.style_img_ph: style_image_np})
                image_utils.save_np_image(style_img_croped_resized_np,
                                      os.path.join(image_stylization_options.output_dir, '%s.jpg' % (style_img_name)))

                # Computes bottleneck features of the style prediction network for the
                # given style image.
                style_params = self.sess.run(self.bottleneck_feat
                    , feed_dict={self.style_img_ph: style_image_np})

                # Interpolates between the parameters of the identity transform and
                # style parameters of the given style image.
                stylized_image_res = self.sess.run(self.stylized_images,
                    feed_dict={
                    self.bottleneck_feat: identity_params * (1 - interpolation_weight) + style_params * interpolation_weight,
                    self.content_img_ph: content_img_np
                })

                # Saves stylized image.
                stylized_image_path = os.path.join(image_stylization_options.output_dir, '%s_stylized_%s_%.1f.jpg' % (content_img_name, style_img_name, interpolation_weight))
                image_utils.save_np_image(stylized_image_res, stylized_image_path)
                stylized_images_path.append(stylized_image_path)
        

        return stylized_images_path

""" ImageStylization options """
class ImageStylizationOptions:
    def __init__(self, checkpoint, output_dir , image_size=256, content_square_crop=False
        , style_image_size=256, style_square_crop=False, maximum_styles_to_evaluate=1024):
        # Path to the model checkpoint.
        self.checkpoint = checkpoint
        # Output directory.
        self.output_dir = output_dir
        # Image size.
        self.image_size = image_size
        # Wheather to center crop the content image to be a square or not.
        self.content_square_crop = content_square_crop
        # Style image size.
        self.style_image_size = style_image_size
        # Wheather to center crop the style image to be a square or not.
        self.style_square_crop = style_square_crop
        # Maximum number of styles to evaluate.
        self.maximum_styles_to_evaluate = maximum_styles_to_evaluate
    def to_string(self):
        return 'checkpoint: {:}\noutput_dir: {:}\nimage_size: {:}\ncontent_square_crop: {:}\nstyle_image_size: {:}\nstyle_square_crop: {:}\nmaximum_styles_to_evaluate: {:}'\
        .format(self.checkpoint, self.output_dir, self.image_size, self.content_square_crop, self.style_image_size, self.style_square_crop, self.maximum_styles_to_evaluate)
