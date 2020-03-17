import tensorflow as tf

# graph_def = graph_pb2.GraphDef()

#with open(FLAGS.graph, "rb") as f:
#  if FLAGS.input_binary:
#    graph_def.ParseFromString(f.read())
#  else:
#    text_format.Merge(f.read(), graph_def)

with tf.gfile.GFile('./frozen_inference_graph_face.pb', 'rb') as f:   
  graph_def = tf.GraphDef()
  graph_def.ParseFromString(f.read())

  #print(graph_def.node)

  for node in graph_def.node:
    print(node.name)
