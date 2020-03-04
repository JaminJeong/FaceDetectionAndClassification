#!/usr/bin/python
# -*- coding: utf-8 -*-
# pylint: disable=C0103
# pylint: disable=E1101

import sys, os
import time
import numpy as np
import tensorflow as tf
import cv2

sys.path.append("..")
sys.path.append("../../")
# sys.path.append("../../../")

from utils import label_map_util
from utils import visualization_utils_color as vis_util

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = './model/frozen_inference_graph_face.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = './protos/face_label_map.pbtxt'

NUM_CLASSES = 2

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)


import face_recognitions.facenet.src.facenet as facenet

def prewhiten(x):
  """

  :param x: The numpy array representing an image
  :return: A whitened image
  """
  mean = np.mean(x)
  std = np.std(x)
  std_adj = np.maximum(std, 1.0/np.sqrt(x.size))
  y = np.multiply(np.subtract(x, mean), 1/std_adj)
  return y

def load_tf_facenet_graph(FACENET_MODEL_PATH):
    '''
    Loads the Facenet Tensorflow Graph into memory.

    :param FACENET_MODEL_PATH:
    :return:
    '''

    assert os.path.isfile(FACENET_MODEL_PATH)

    facenet.load_model(FACENET_MODEL_PATH)
    images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
    embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
    phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

    return images_placeholder,embeddings,phase_train_placeholder


def get_face_embeddings(sess,embeddings,images_placeholder,phase_train_placeholder,
                        nrof_images,nrof_batches_per_epoch,FACENET_PREDICTION_BATCH_SIZE,images_array):
  """

  :param sess: Current Tensorflow session variable
  :param embeddings: A tensor variable that holds the embeddings of the result
  :param images_placeholder: A tensor variable that holds the images
  :param phase_train_placeholder: A tensor variable
  :param nrof_images: Number of detected faces
  :param nrof_batches_per_epoch: Number of images to run per epoch
  :param FACENET_PREDICTION_BATCH_SIZE: Number of maximum faces per facenet detection
  :param images_array: Numpy representation of an image.
  :return:
  """
  embedding_size = embeddings.get_shape()[1]
  emb_array = np.zeros((nrof_images, embedding_size))

  for i in range(nrof_batches_per_epoch):
    start_index = i * FACENET_PREDICTION_BATCH_SIZE
    end_index = min((i + 1) * FACENET_PREDICTION_BATCH_SIZE, nrof_images)
    images_batch = images_array[start_index:end_index]  # Pass in several different paths
    images_batch = np.array(images_batch)
    feed_dict = {images_placeholder: images_batch, phase_train_placeholder: False}
    function_timer_start = time.time()
    emb_array[start_index:end_index, :] = sess.run(embeddings, feed_dict=feed_dict)
    function_timer = time.time() - function_timer_start
    print('Calculating image embedding cost: {}'.format(function_timer))

  return emb_array

cap = cv2.VideoCapture("./media/videoplayback.mp4")
out = None

face_image_dic = {}
face_file_path = './media/face_image'
assert os.path.isdir(face_file_path)
face_file_list = os.listdir(face_file_path)
for file_path in face_file_list:
    face_image_dic[os.path.splitext(file_path)[0]] = {}
    face_image_dic[os.path.splitext(file_path)[0]]['image'] = cv2.imread(os.path.join(face_file_path, file_path))

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

with detection_graph.as_default():
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True

  images_placeholder, embeddings, phase_train_placeholder = load_tf_facenet_graph(
      '../../face_recognitions/facenet/model/20180402-114759/20180402-114759.pb')

  with tf.Session(graph=detection_graph, config=config) as sess:
    frame_num = 1490;

    print(f"face_image_dic : {face_image_dic}")

    for key in face_image_dic:
        facenet_input_image = face_image_dic[key]['image']
        facenet_input_image = cv2.resize(facenet_input_image, (160, 160))
        facenet_input_image = facenet_input_image.astype(dtype=np.float)
        facenet_input_image = prewhiten(facenet_input_image)
        facenet_input_image = np.expand_dims(facenet_input_image, 0)
        feed_dict = {images_placeholder: facenet_input_image, phase_train_placeholder: False}
        function_timer_start = time.time()
        emb_array = sess.run(embeddings, feed_dict=feed_dict)
        emb_array = np.squeeze(emb_array)
        face_image_dic[key]['embedding'] = emb_array
        # print(f"emb_array_{frame_num}_{idx} : {emb_array}")
        function_timer = time.time() - function_timer_start

    category_index = {1: {'id': 1, 'name': 'unknown'},}
    # category_index = {}
    for idx, key in enumerate(face_image_dic):
        idx = idx + 2
        category_index[idx] = {'id': idx}
        category_index[idx] = {'name': key}

    print(f"category_index : {category_index}")

    while frame_num:
      frame_num -= 1
      ret, image = cap.read()
      if ret == 0:
          break

      if out is None:
          [h, w] = image.shape[:2]
          out = cv2.VideoWriter("./media/test_out.avi", 0, 25.0, (w, h))

      image_np = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

      # the array based representation of the image will be used later in order to prepare the
      # result image with boxes and labels on it.
      # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
      image_np_expanded = np.expand_dims(image_np, axis=0)
      image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
      # Each box represents a part of the image where a particular object was detected.
      boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
      # Each score represent how level of confidence for each of the objects.
      # Score is shown on the result image, together with the class label.
      scores = detection_graph.get_tensor_by_name('detection_scores:0')
      classes = detection_graph.get_tensor_by_name('detection_classes:0')
      num_detections = detection_graph.get_tensor_by_name('num_detections:0')
      # Actual detection.
      start_time = time.time()
      (boxes, scores, classes, num_detections) = sess.run(
          [boxes, scores, classes, num_detections],
          feed_dict={image_tensor: image_np_expanded})
      elapsed_time = time.time() - start_time
      print('inference time cost: {}'.format(elapsed_time))
      # print(boxes.shape, boxes)
      # print(scores.shape,scores)
      # print(classes.shape,classes)
      # print(num_detections)

      boxes = np.squeeze(boxes)
      classes = np.squeeze(classes).astype(np.int32)
      scores = np.squeeze(scores)

      image_shape = image.shape
      # print(f"image_shape : {image_shape}")
      for idx, score in enumerate(scores):
          if score > 0.5 and classes[idx] == 1:
              # print(f"boxes : {boxes[idx]}")
              y1 = int(boxes[idx][0] * image_shape[0])
              x1 = int(boxes[idx][1] * image_shape[1])
              y2 = int(boxes[idx][2] * image_shape[0])
              x2 = int(boxes[idx][3] * image_shape[1])
              facenet_input_image = image[y1:y2, x1:x2, :]
              # print(f"y1 x1 y2 x2 : {y1}, {x1}, {y2}, {x2}")
              # print(f"facenet_input_image.shape : {facenet_input_image.shape}")
              # cv2.imwrite("./media/outputfile/facenet_image_{}_{}.jpg".format(frame_num, idx), facenet_input_image)
              facenet_input_image = cv2.resize(facenet_input_image, (160, 160))
              facenet_input_image = facenet_input_image.astype(dtype=np.float)
              facenet_input_image = prewhiten(facenet_input_image)
              facenet_input_image = np.expand_dims(facenet_input_image, 0)
              feed_dict = {images_placeholder: facenet_input_image, phase_train_placeholder: False}
              function_timer_start = time.time()
              emb_array = sess.run(embeddings, feed_dict=feed_dict)
              emb_array = np.squeeze(emb_array)
              # print(f"emb_array_{frame_num}_{idx} : {emb_array}")
              function_timer = time.time() - function_timer_start
              print('Calculating image embedding cost: {}'.format(function_timer))

              dist_list = []
              for key in face_image_dic:
                  # print(f"face_image_dic_key_embedding shape : {face_image_dic[key]['embedding'].shape}")
                  # print(f"emb_array shape : {emb_array.shape}")
                  dist_list.append(np.linalg.norm(face_image_dic[key]["embedding"] - emb_array))

              dist_list = np.array(dist_list)
              classes[idx] = np.argmin(dist_list) + 2 # offset
          else:
              classes[idx] = 1 # unknown

      # print(f"category_index : {category_index}")

      # Visualization of the results of a detection.
      vis_util.visualize_boxes_and_labels_on_image_array(
#          image_np,
          image,
          boxes,
          classes,
          scores,
          category_index,
          use_normalized_coordinates=True,
          line_thickness=4)
      out.write(image)


    cap.release()
    out.release()
