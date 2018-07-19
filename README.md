# NVIDIA Inference Server Clients

The NVIDIA Inference Server provides a cloud inferencing solution
optimized for NVIDIA GPUs. The inference server provides deep-learning
inferencing via HTTP and gRPC endpoints, allowing remote clients to
request inferencing for any model being managed by the server.

This repo contains C++ and Python client libraries that make it easy
to communicate with the inference server. Also included are a couple
of example applications.

- C++ and Python versions of *image\_client*, an example application
  that uses the C++ or Python client library to execute image
  classification models on the inference server.

- Python version of *grpc\_image\_client*, an example application that
  is functionally equivalent to *image\_client* but that uses gRPC
  generated client code to communicate with the inference server
  (instead of the inference server client library).

- C++ version of *perf\_client*, an example application that issues a
  large number of concurrent requests to the inference server to
  measure latency and throughput for a given model. You can use this
  to experiment with different model configuration settings for your
  models.

The inference server itself is delivered as a containerized solution
on the [NVIDIA GPU
Cloud](https://www.nvidia.com/en-us/gpu-cloud/). See the [Inference
Server User
Guide](https://docs.nvidia.com/deeplearning/sdk/inference-user-guide/index.html)
for information on how to install and configure the inference server.

Use [Issues](https://github.com/NVIDIA/dl-inference-server/issues) to
report any issues or questions about the client libraries and
examples. A [DevTalk
forum](https://devtalk.nvidia.com/default/board/262/container-inference-server)
is also available for inference server issues and questions.

## Branches

**master**: Active development branch. Typically will be compatible
 with the currently released Inference Server container, but not
 guaranteed.

**yy.mm**: Branch compatible with Inference Server yy.mm, for
  example *18.07*.

## Building the Clients

Before building the client libraries and applications you must first
install some prerequisites. The following instructions assume Ubuntu
16.04. OpenCV is used by image\_client to preprocess images before
sending them to the inference server for inferencing. The python-pil
package is required by the Python image\_client example.

    sudo apt-get update
    sudo apt-get install build-essential libcurl3-dev libopencv-dev libopencv-core-dev python-pil \
                         software-properties-common autoconf automake libtool pkg-config
    
Creating the whl file for the Python client library requires setuptools. 
And grpcio-tools is required for gRPC support in Python client library.

    pip install --no-cache-dir --upgrade setuptools grpcio-tools

With those prerequisites installed, the C++ and Python client libraries
and example image\_client and perf\_client applications can be built:

    make -f Makefile.clients all pip

Build artifacts are in build/.  The Python whl file is generated in
build/dist/dist/ and can be installed with a command like the following:

    pip install --no-cache-dir --upgrade build/dist/dist/inference_server-1.0.0-cp27-cp27mu-linux_x86_64.whl

## Building the Clients with Docker

A Dockerfile is provided for building the client libraries and examples 
inside a container. 

Before building, edit the Dockerfile to set PYVER to either 2.7 or 3.5
to select the version of Python that you will be using. The following
will build the C++ client library, C++ examples and create a Python
wheel file for the Python client library.

    $ docker build -t inference_server_clients .

The easiest way to extract the built libraries and examples from the
docker image is to mount a host driver into the container and then
copy the images.

    $ docker run --rm -it -v /tmp:/tmp/host inference_server_clients
    # cp build/image_client /tmp/host/.
    # cp build/perf_client /tmp/host/.
    # cp build/dist/dist/inference_server-*.whl /tmp/host/.

You can now access image\_client and perf\_client from /tmp on the
host system. Before running the C++ or Python examples on the host the
appropriate dependencies must be installed. 

OpenCV is used by the C++ image\_client example to preprocess images
before sending them to the inference server for inferencing.

    $ sudo apt-get install libcurl3-dev libopencv-dev libopencv-core-dev

The Python whl file can be installed using pip:

    $ pip install --no-cache-dir --upgrade /tmp/inference_server-0.4.0-cp27-cp27mu-linux_x86_64.whl

The Python image\_client example requires Pillow for image processing
so install that package before running.

    $ pip install --no-cache-dir --upgrade pillow

## Image Classification Example

The image classification example that uses the C++ client API is
available at src/clients/c++/image\_client.cc. After building as
described above and copying to the host system, the executable is
available at /tmp/image\_client. The python version of the image
classification client is available at
src/clients/python/image\_client.py.

To use image\_client (or image\_client.py) you must first have an
inference server that is serving one or more image classification
models. The image\_client example requires that the model have a
single image input and produce a single classification output.

An example model store containing a single Caffe2 ResNet50 model
is provided in the examples/models directory that you can use to
demonstrate image\_client. Before using the example model store you
must fetch any missing model definition files from their public model
zoos.

    $ cd examples
    $ ./fetch_models.sh

Following the instructions in the [Inference Server User
Guide](https://docs.nvidia.com/deeplearning/sdk/inference-user-guide/index.html),
launch the inference server container pointing to that model
store. For example:

    $ nvidia-docker run --rm -p8000:8000 -p8001:8001 -v/path/to/dl-inference-server/examples/models:/models nvcr.io/nvidia/inferenceserver:18.07-py2 inference_server --model-store=/models

Make sure you choose the most recent version of
nvcr.io/nvidia/inferenceserver. Port 8000 exposes the inference server
HTTP endpoint and port 8001 exposes the gRPC endpoint. Replace
/path/to/dl-inference-server/examples/models with the corresponding
path in your local clone of this repo. Once the server is running you
can use image\_client application to send inference requests to the
server. Note that the LD\_LIBRARY\_PATH is needed for the gRPC
libraries you copied from the container.

    $ LD_LIBRARY_PATH=/tmp /tmp/image_client -m resnet50_netdef -s INCEPTION examples/data/mug.jpg

The Python version of the application accepts the same command-line
arguments.

    $ LD_LIBRARY_PATH=/tmp src/clients/python/image_client.py -m resnet50_netdef -s INCEPTION examples/data/mug.jpg

The image\_client and image\_client.py examples use the inference
server client library to talk to the inference server. By default they
instruct the client library to use HTTP protocol to talk to the
inference server, but you can use gRPC protocol by providing the -i
flag. You must also use the -u flag to point at the gRPC endpoint on
the inference server.

    $ LD_LIBRARY_PATH=/tmp /tmp/image_client -i grpc -u localhost:8001 -m resnet50_netdef -s INCEPTION examples/data/mug.jpg

By default the client prints the most probable classification for the image.

    Prediction totals:
            cnt=1   (504) COFFEE MUG

Use the -c flag to see more classifications.

    $ LD_LIBRARY_PATH=/tmp /tmp/image_client -m resnet50_netdef -s INCEPTION -c 3 examples/data/mug.jpg
    Output probabilities:
    batch 0: 504 (COFFEE MUG) = 0.777365267277
    batch 0: 968 (CUP) = 0.213909029961
    batch 0: 967 (ESPRESSO) = 0.00294389552437
    Prediction totals:
            cnt=1	(504) COFFEE MUG

The -b flag allows you to send a batch of images for inferencing.
Currently, the image\_client examples just send the same image
multiple times, so you will just see the same classification results
repeated (as indicated by the 'cnt' value).

    $ LD_LIBRARY_PATH=/tmp /tmp/image_client -m resnet50_netdef -s INCEPTION -b 2 examples/data/mug.jpg
    Prediction totals:
            cnt=2	(504) COFFEE MUG

The grpc\_image\_client.py example behaves the same as the image\_client
examples except that instead of using the inference server client
library it uses the gRPC generated client library to communicate with
the inference server.

    $ LD_LIBRARY_PATH=/tmp src/clients/python/grpc_image_client.py -m resnet50_netdef -s INCEPTION examples/data/mug.jpg -u localhost:8001


## Perf Example

The perf\_client example uses the C++ client API to send concurrent
requests to the inference server. After building as described above,
the executable is available at build/perf\_client.

You can use perf\_client with any kind of model. It sends random data
as input and ignores the output. Any model in the example model store
can be used to demonstrate perf\_client. Before using the example
model store you must fetch any missing model definition files from
their public model zoos.

    $ cd examples
    $ ./fetch_models.sh

Following the instructions in the [Inference Server User
Guide](https://docs.nvidia.com/deeplearning/sdk/inference-user-guide/index.html),
launch the inference server container pointing to that model
store. For example:

    $ nvidia-docker run --rm -p8000:8000 -p8001:8001 -v/path/to/dl-inference-server/examples/models:/models nvcr.io/nvidia/inferenceserver:18.07-py2 inference_server --model-store=/models

Make sure you choose the most recent version of
nvcr.io/nvidia/inferenceserver. Port 8000 exposes the inference server
HTTP endpoint and port 8001 exposes the gRPC endpoint. Replace
/path/to/dl-inference-server/examples/models with the corresponding
path in your local clone of this repo.

You can use the --help flag to see all perf\_client command-line
options. You vary thread count (-t) to control concurrency. Use
warmup-passes (-w) to avoid start-up overhead and measurement-passes
(-p) to control the length of the run. For example,

    $ LD_LIBRARY_PATH=/tmp /tmp/perf_client -m resnet50_netdef -t4 -w5 -p100
    *** Warming up ***
    *** Begin ***
    *** End ***
    *** Results ***
    Thread count: 4
    Batch size: 1
    Server:
      Inference count: 400
      Cumulative Time: 3900472 usec (avg 9751 usec)
        Run Wait: 387484 usec (avg 968 usec)
        Run: 3489622 usec (avg 8724 usec)
    Client:
      Inference count: 400
      Total Time: 3085831 usec
      Throughput: 129 infer/sec

The results give timing as seen by the server and from the client. In
this case we see 4 threads each did 100 measurement passes for a total
of 400 inferences using the Caffe2 ResNet50 model. The server results
tell us that on average each inference took 9751 usecs. Of that an
average of 968 usecs was waiting on an in-progress inference and the
other 8724 usecs was the actual time to execute the model.

By increasing concurrency (-t) and batch size (-b) we can trade off
higher throughput for increased per-inference latency. For example,

    $ build/perf_client -m resnet50_netdef -t16 -b64 -w5 -p100
    *** Warming up ***
    *** Begin ***
    *** End ***
    *** Results ***
    Thread count: 16
    Batch size: 64
    Server:
      Inference count: 102400
      Cumulative Time: 1241498216 usec (avg 12124 usec)
        Run Wait: 1089736082 usec (avg 10641 usec)
        Run: 151584509 usec (avg 1480 usec)
    Client:
      Inference count: 102400
      Total Time: 86234614 usec
      Throughput: 1187 infer/sec

With 16 threads and a batch size of 64 the throughput (inferences per
second) increases significantly, but at the expense of an increase in
wait time (10641 usecs) and thus latency for any individual inference.

## C++ API

The C++ client API exposes a class-based interface for querying server
and model status and for performing inference. The commented interface
is available at src/clients/c++/request.h.

The following shows an example of the basic steps required for
inferencing (error checking not included to improve clarity, see
image_client.cc for full error checking):

```c++
// Create the context object for inferencing using the latest version
// of the 'mnist' model. Use the HTTP protocol for communication with
// the inference server (to use gRPC use InferGrpcContext::Create).
std::unique_ptr<InferContext> ctx;
InferHttpContext::Create(&ctx, "localhost:8000", "mnist");

// Get handle to model input and output.
std::shared_ptr<InferContext::Input> input;
ctx->GetInput(input_name, &input);

std::shared_ptr<InferContext::Output> output;
ctx->GetOutput(output_name, &output);

// Set options so that subsequent inference runs are for a given batch_size
// and return a result for ‘output’. The ‘output’ result is returned as a
// classification result of the ‘k’ most probable classes.
std::unique_ptr<InferContext::Options> options;
InferContext::Options::Create(&options);
options->SetBatchSize(batch_size);
options->AddClassResult(output, k);
ctx->SetRunOptions(*options);

// Provide input data for each batch.
input->Reset();
for (size_t i = 0; i < batch_size; ++i) {
  input->SetRaw(input_data[i]);
}

// Run inference and get the results. When the Run() call returns the ctx
// can be used for another inference run. Results are owned by the caller
// and can be retained as long as necessary.
std::vector<std::unique_ptr<InferContext::Result>> results;
ctx->Run(&results);

// For each entry in the batch print the top prediction.
for (size_t i = 0; i < batch_size; ++i) {
  InferContext::Result::ClassResult cls;
  results[0]->GetClassAtCursor(i, &cls);
  std::cout << "batch " << i << ": " << cls.label << std::endl;
}
```

## Python API

The Python client API provides similar capabilities as the C++
API. The commented interface for StatusContext and InferContext
classes is available at src/clients/python/\_\_init\_\_.py.

The following shows an example of the basic steps required for
inferencing (error checking not included to improve clarity):

```python
from inference_server.api import *

# Create input with random data
input_list = list()
for b in range(batch_size):
    in = np.random.randint(size=input_size, dtype=input_dtype)
    input_list.append(in)

# Run inferencing and get the top-3 classes
ctx = InferContext("localhost:8000", ProtocolType.HTTP, "mnist")
results = ctx.run(
    { "data" : input_list },
    { "prob" : (InferContext.ResultFormat.CLASS, 3) },
    batch_size)


# Print results
for (result_name, result_val) in iteritems(results):
    for b in range(batch_size):
        print("output {}, batch {}: {}".format(result_name, b, result_val[b]))
```
