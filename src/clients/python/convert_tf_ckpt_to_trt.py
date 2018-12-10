# Importing TensorFlow and TensorRT
import tensorflow as tf
import tensorflow.contrib.tensorrt as trt

# Inference with TF-TRT workflow
graph = tf.Graph()
with graph.as_default():
   with tf.Session() as sess:
       # Import model
       # saver = tf.train.import_meta_graph("your/path/to/model/def.meta")
       # saver.restore(sess, "your/path/to/checkpoint.ckpt")
       saver = tf.train.import_meta_graph("/data5/xin/commercials_2400k/commercials_2400k_irv2.ckpt")
       saver.restore(sess, "/data5/xin/commercials_2400k/commercials_2400k_irv2.ckpt.data-00000-of-00001")
#----------------------------- (TF-TRT specific part) --------------------------
       # Freeze graph
       frozen_graph = tf.graph_util.convert_variables_to_constants(
           sess,
           tf.get_default_graph().as_graph_def(),
           output_node_names=["your_output_node_name"])

       # Create TensorRT inference graph
       trt_graph = trt.create_inference_graph(
           input_graph_def=frozen_graph,
           outputs=["your_output_node_name"],
           max_batch_size=your_batch_size,
     max_workspace_size_bytes=max_GPU_mem_size_for_TRT,
           precision_mode=your_precision_mode) # ["FP32", "FP16","INT8"]

       # Import TensorRT graph into new graph and run
       output_node = tf.import_graph_def(
           trt_graph,
           return_elements=["your_output_node_name"])
#-----------------------------------------------------------------------------
       sess.run(output_node)