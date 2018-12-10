from tf_trt_models.classification import download_classification_checkpoint, build_classification_graph
from tf_trt_models.detection import download_detection_model, build_detection_graph

import tensorrt as trt
import uff
from tensorrt.parsers import uffparser
import tensorflow as tf
from google.protobuf import text_format

# mode = 'CLASSIFICATION'

mode = 'DETECTION'
PRECISION = 'FP32'

if mode == 'CLASSIFICATION':
    MODEL = 'inception_resnet_v2'
    checkpoint_path = download_classification_checkpoint(MODEL)

    frozen_graph, input_names, output_names = build_classification_graph(
        model=MODEL,
        checkpoint=checkpoint_path,
        num_classes=1001
    )
else:  # DETECTION
    MODEL = 'ssd_mobilenet_v2_coco'
    config_path, checkpoint_path = download_detection_model(MODEL, 'data')

    frozen_graph, input_names, output_names = build_detection_graph(
        config=config_path,
        checkpoint=checkpoint_path,
        score_threshold=0.3,
        batch_size=1
    )
print('>>>>>>>>>> input_names', input_names)
print('>>>>>>>>>> output_names', output_names)

# --------------------------------------------------------------------------

# import tensorflow.contrib.tensorrt as trt_2
# trt_graph = trt_2.create_inference_graph(
#     input_graph_def=frozen_graph,
#     outputs=output_names,
#     max_batch_size=1,
#     max_workspace_size_bytes=1 << 25,
#     precision_mode=PRECISION,
#     minimum_segment_size=50
# )

# save_path = 'official_' + PRECISION + 'xin.engine'

# with open(save_path, 'wa+') as f:
#     f.write(trt_graph.SerializeToString())
# print("Saved TRT engine to {}".format(save_path))

# --------------------------------------------------------------------------

INPUT_LAYERS = input_names
OUTPUT_LAYERS = output_names
INFERENCE_BATCH_SIZE = 32

INPUT_C = 3
INPUT_H = 299
INPUT_W = 299

G_LOGGER = trt.infer.ConsoleLogger(trt.infer.LogSeverity.INFO)

uff_model = uff.from_tensorflow(frozen_graph, output_names)
parser = uffparser.create_uff_parser()
parser.register_input(INPUT_LAYERS[0], (INPUT_C,INPUT_H,INPUT_W),0)
parser.register_output(OUTPUT_LAYERS[0])

# Build your TensorRT inference engine
if(PRECISION == 'FP32'):
    engine = trt.utils.uff_to_trt_engine(
        G_LOGGER, 
        uff_model, 
        parser, 
        INFERENCE_BATCH_SIZE, 
        1<<20, 
        trt.infer.DataType.FLOAT
    )
#### not working
elif(PRECISION == 'FP16'):
    engine = trt.utils.uff_to_trt_engine(
        G_LOGGER, 
        uff_model, 
        parser, 
        INFERENCE_BATCH_SIZE, 
        1<<20, 
        trt.infer.DataType.HALF
    )
# engine = trt.utils.uff_to_trt_engine(
#     G_LOGGER, 
#     uff_model, 
#     parser, 
#     INFERENCE_BATCH_SIZE, 
#     1<<20, 
#     trt.infer.DataType.FLOAT
# )

# Serialize TensorRT engine to a file for when you are ready to deploy your model.
# save_path = str(config['engine_save_dir']) + "keras_vgg19_b" \
#     + str(INFERENCE_BATCH_SIZE) + "_"+ str(config['precision']) + ".engine"
save_path = 'official_' + PRECISION + 'xin.engine'

trt.utils.write_engine_to_file(save_path, engine.serialize())
print("Saved TRT engine to {}".format(save_path))
# --------------------------------------------------------------------------


