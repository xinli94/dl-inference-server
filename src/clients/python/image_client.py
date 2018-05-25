#!/usr/bin/python

# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import argparse
import numpy as np
from builtins import range
from PIL import Image
from inference_server.api import *
import inference_server.api.model_config_pb2 as model_config

FLAGS = None

def model_dtype_to_np(model_dtype):
    if model_dtype == model_config.TYPE_BOOL:
        return np.bool
    elif model_dtype == model_config.TYPE_INT8:
        return np.int8
    elif model_dtype == model_config.TYPE_INT16:
        return np.int16
    elif model_dtype == model_config.TYPE_INT32:
        return np.int32
    elif model_dtype == model_config.TYPE_INT64:
        return np.int64
    elif model_dtype == model_config.TYPE_UINT8:
        return np.uint8
    elif model_dtype == model_config.TYPE_UINT16:
        return np.uint16
    elif model_dtype == model_config.TYPE_FP16:
        return np.float16
    elif model_dtype == model_config.TYPE_FP32:
        return np.float32
    elif model_dtype == model_config.TYPE_FP64:
        return np.float64
    return None

def parse_model(url, model_name, batch_size, verbose=False):
    """
    Check the configuration of a model to make sure it meets the
    requirements for an image classification network (as expected by
    this client)
    """

    ctx = ServerStatusContext(url, model_name, verbose)
    server_status = ctx.get_server_status()

    if model_name not in server_status.model_status:
        raise Exception("unable to get status for '" + model_name + "'")

    status = server_status.model_status[model_name]
    config = status.config

    if len(config.input) != 1:
        raise Exception("expecting 1 input, got " + len(config.input))
    if len(config.output) != 1:
        raise Exception("expecting 1 output, got " + len(config.output))

    input = config.input[0]
    output = config.output[0]

    if output.data_type != model_config.TYPE_FP32:
        raise Exception("expecting output datatype to be TYPE_FP32, model '" +
                        model_name + "' output type is " +
                        model_config.DataType.Name(output.data_type))

    # Output is expected to be a vector. But allow any number of
    # dimensions as long as all but 1 is size 1 (e.g. { 10 }, { 1, 10
    # }, { 10, 1, 1 } are all ok).
    non_one_cnt = 0
    for dim in output.dims:
        if dim > 1:
            non_one_cnt += 1
            if non_one_cnt > 1:
                raise Exception("expecting model output to be a vector")

    # Model specifying maximum batch size of 0 indicates that batching
    # is not supported and so the input tensors do not expect an "N"
    # dimension (and 'batch_size' should be 1 so that only a single
    # image instance is inferred at a time).
    max_batch_size = config.max_batch_size
    if max_batch_size == 0:
        if batch_size != 1:
            raise Exception("batching not supported for model '" + model_name + "'")
    else: # max_batch_size > 0
        if batch_size > max_batch_size:
            raise Exception("expecting batch size <= " + max_batch_size +
                            " for model '" + model_name + "'")

    # Model input must have 3 dims, either CHW or HWC
    if len(input.dims) != 3:
        raise Exception("expecting input to have 3 dimensions, model '" +
                        model_name + "' input has " << len(input.dims))

    if ((input.format != model_config.ModelInput.FORMAT_NCHW) and
        (input.format != model_config.ModelInput.FORMAT_NHWC)):
        raise Exception("unexpected input format " + model_config.ModelInput.Format.Name(input.format) +
                        ", expecting " +
                        model_config.ModelInput.Format.Name(model_config.ModelInput.FORMAT_NCHW) +
                        " or " +
                        model_config.ModelInput.Format.Name(model_config.ModelInput.FORMAT_NHWC))

    if input.format == model_config.ModelInput.FORMAT_NHWC:
        h = input.dims[0]
        w = input.dims[1]
        c = input.dims[2]
    else:
        c = input.dims[0]
        h = input.dims[1]
        w = input.dims[2]

    return (input.name, output.name, c, h, w, input.format, model_dtype_to_np(input.data_type))

def preprocess(img, format, dtype, c, h, w, scaling):
    """
    Pre-process an image to meet the size, type and format
    requirements specified by the parameters.
    """
    #np.set_printoptions(threshold='nan')

    if c == 1:
        sample_img = img.convert('L')
    else:
        sample_img = img.convert('RGB')

    resized_img = sample_img.resize((h, w), Image.BILINEAR)
    resized = np.array(resized_img)
    if resized.ndim == 2:
        resized = resized[:,:,np.newaxis]

    typed = resized.astype(dtype)

    if scaling == 'INCEPTION':
        scaled = (typed / 128) - 1
    elif scaling == 'VGG':
        if c == 1:
            scaled = typed - np.asarray((128,), dtype=dtype)
        else:
            scaled = typed - np.asarray((123, 117, 104), dtype=dtype)
    else:
        scaled = typed

    # Swap to CHW if necessary
    if format == model_config.ModelInput.FORMAT_NCHW:
        ordered = np.transpose(scaled, (2, 0, 1))
    else:
        ordered = scaled

    # Channels are in RGB order. Currently model configuration data
    # doesn't provide any information as to other channel orderings
    # (like BGR) so we just assume RGB.
    return ordered

def postprocess(results, batch_size, num_classes, verbose=False):
    """
    Post-process results to show classifications.
    """
    show_all = verbose or ((batch_size == 1) and (num_classes > 1))
    if show_all:
        print("Output probabilities:")

    if len(results) != 1:
        raise Exception("expected 1 result, got " + len(results))

    result = list(results.values())[0]
    if len(result) != batch_size:
        raise Exception("expected " + batch_size + " results, got " + len(result))

    # For each result in the batch count the top prediction. Since we
    # used the same image for every entry in the batch we expect the
    # top prediction to be the same for each entry... but this code
    # doesn't assume that.
    counts = dict()
    predictions = dict()
    for (batch_index, batch_result) in enumerate(result):
        label = batch_result[0][2]
        if label not in counts:
            counts[label] = 0

        counts[label] += 1
        predictions[label] = batch_result[0]

        # If requested, print all the class results for the entry
        if show_all:
            for cls in batch_result:
                print("batch {}: {} ({}) = {}".format(batch_index, cls[0], cls[2], cls[1]))

    # Summary
    print("Prediction totals:")
    for (label, cnt) in counts.items():
        cls = predictions[label]
        print("\tcnt={}\t({}) {}".format(cnt, cls[0], cls[2]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose', action="store_true", required=False, default=False,
                        help='Enable verbose output')
    parser.add_argument('-m', '--model-name', type=str, required=True,
                        help='Name of model')
    parser.add_argument('-x', '--model-version', type=str, required=False,
                        help='Version of model. Default is to use latest version.')
    parser.add_argument('-b', '--batch-size', type=int, required=False, default=1,
                        help='Batch size. Default is 1.')
    parser.add_argument('-c', '--classes', type=int, required=False, default=1,
                        help='Number of class results to report. Default is 1.')
    parser.add_argument('-s', '--scaling', type=str, choices=['NONE', 'INCEPTION', 'VGG'],
                        required=False, default='NONE',
                        help='Type of scaling to apply to image pixels. Default is NONE.')
    parser.add_argument('-u', '--url', type=str, required=False, default='localhost:8000',
                        help='Inference server URL. Default is localhost:8000.')
    parser.add_argument('-p', '--preprocessed', type=str, required=False,
                        metavar='FILE', help='Write preprocessed image to specified file.')
    parser.add_argument('image_filename', type=str, nargs='?', default=None,
                        help='Input image.')
    FLAGS = parser.parse_args()

    # Make sure the model matches our requirements, and get some
    # properties of the model that we need for preprocessing
    input_name, output_name, c, h, w, format, dtype = parse_model(
        FLAGS.url, FLAGS.model_name,
        FLAGS.batch_size, FLAGS.verbose)

    # Load and preprocess the image
    img = Image.open(FLAGS.image_filename)
    input_tensor = preprocess(img, format, dtype, c, h, w, FLAGS.scaling)

    if FLAGS.preprocessed is not None:
        with open(preprocessed, "w") as file:
            file.write(input_tensor.tobytes())

    # Need 'batch_size' copies of the input tensor...
    input_tensors = [input_tensor for b in range(FLAGS.batch_size)]

    ctx = InferContext(FLAGS.url, FLAGS.model_name, FLAGS.model_version, FLAGS.verbose)
    results = ctx.run(
        { input_name : input_tensors },
        { output_name : (InferContext.ResultFormat.CLASS, FLAGS.classes) },
        FLAGS.batch_size)

    postprocess(results, FLAGS.batch_size, FLAGS.classes, FLAGS.verbose)
