import tensorrt as trt
import uff
from tensorrt.parsers import uffparser
import tensorflow as tf
from google.protobuf import text_format


config = {
    # # Training params
    # "train_data_dir": "/home/data/train",  # training data
    # "val_data_dir": "/home/data/val",  # validation data 
    # "train_batch_size": 16,  # training batch size
    # "epochs": 3,  # number of training epochs
    # "num_train_samples" : 2936,  # number of training examples
    # "num_val_samples" : 734,  # number of test examples

    # Where to save models (Tensorflow + TensorRT)
    "graphdef_file": "/models/commercials_2400k_irv2/1/model.savedmodel/",
    "frozen_model_file": "./frozen_graph.pb",
    # "snapshot_dir": "/home/data/model/snapshot",
    "engine_save_dir": "./xin",
    
    # Needed for TensorRT
    "image_dim": 229,  # the image size (square images)
    "inference_batch_size": 32,  # inference batch size
    "input_layer": "images",  # name of the input tensor in the TF computational graph
    "out_layer": "scores",  # name of the output tensorf in the TF conputational graph
    "output_size" : 4,  # number of classes in output (5)
    "precision": "fp32",  # desired precision (fp32, fp16)

    # "test_image_path" : "/home/data/val/roses"
}

G_LOGGER = trt.infer.ConsoleLogger(trt.infer.LogSeverity.INFO)

def create_and_save_inference_engine():
    # Define network parameters, including inference batch size, name & dimensionality of input/output layers
    INPUT_LAYERS = [config['input_layer']]
    OUTPUT_LAYERS = [config['out_layer']]
    INFERENCE_BATCH_SIZE = config['inference_batch_size']

    INPUT_C = 3
    INPUT_H = config['image_dim']
    INPUT_W = config['image_dim']

    # Load your newly created Tensorflow frozen model and convert it to UFF
    uff_model = uff.from_tensorflow_frozen_model(config['frozen_model_file'], OUTPUT_LAYERS)
    # uff_model = uff.from_tensorflow(config['graphdef_file'] + 'saved_model.pb', OUTPUT_LAYERS)
    # uff_model = uff.from_tensorflow('saved_model.pb', OUTPUT_LAYERS)

    print('>>>> Done!!!!')
    # Create a UFF parser to parse the UFF file created from your TF Frozen model
    parser = uffparser.create_uff_parser()
    parser.register_input(INPUT_LAYERS[0], (INPUT_C,INPUT_H,INPUT_W),0)
    parser.register_output(OUTPUT_LAYERS[0])

    # Build your TensorRT inference engine
    if(config['precision'] == 'fp32'):
        engine = trt.utils.uff_to_trt_engine(
            G_LOGGER, 
            uff_model, 
            parser, 
            INFERENCE_BATCH_SIZE, 
            1<<20, 
            trt.infer.DataType.FLOAT
        )

    elif(config['precision'] == 'fp16'):
        engine = trt.utils.uff_to_trt_engine(
            G_LOGGER, 
            uff_model, 
            parser, 
            INFERENCE_BATCH_SIZE, 
            1<<20, 
            trt.infer.DataType.HALF
        )

    # Serialize TensorRT engine to a file for when you are ready to deploy your model.
    save_path = str(config['engine_save_dir']) + "keras_vgg19_b" \
        + str(INFERENCE_BATCH_SIZE) + "_"+ str(config['precision']) + ".engine"

    trt.utils.write_engine_to_file(save_path, engine.serialize())
    
    print("Saved TRT engine to {}".format(save_path))


def print_all_ops_frozen_graph():
    FILE = config['frozen_model_file']

    graph = tf.Graph()
    with graph.as_default():
      od_graph_def = tf.GraphDef()
      with tf.gfile.GFile(FILE, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

      sess = tf.Session()
      op = sess.graph.get_operations()

      for m in op:
        print m.values()

def print_all_ops_saved_model():
    _EXPORT_DIR = config['graphdef_file']
    with tf.Session(graph=tf.Graph()) as sess:
        tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], _EXPORT_DIR)
        trainable_coll = sess.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        vars = {v.name:sess.run(v.value()) for v in trainable_coll}
    print vars

    # with tf.gfile.GFile("saved_model.pb", "rb") as f:
    #     # graph_def = tf.GraphDef()
    #     proto_b=f.read()
    #     graph_def = tf.GraphDef()
    #     text_format.Merge(proto_b, graph_def) 
    #     # graph_def.ParseFromString(f.read())
    #     print graph_def


create_and_save_inference_engine()
