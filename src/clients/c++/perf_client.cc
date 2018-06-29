// Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include "src/clients/c++/request.h"

#include <csignal>
#include <iostream>
#include <string>
#include <thread>
#include <time.h>
#include <unistd.h>
#include "src/core/constants.h"

namespace ni = nvidia::inferenceserver;
namespace nic = nvidia::inferenceserver::client;

namespace {

volatile bool early_exit = false;

void
SignalHandler(int signum)
{
  std::cout
    << "Interrupt signal (" << signum << ") received." << std::endl
    << "Waiting for in-flight inferences to complete." << std::endl;

  early_exit = true;
}

enum ProtocolType {
  HTTP = 0,
  GRPC = 1
};

ProtocolType
ParseProtocol(const std::string& str)
{
  std::string protocol(str);
  std::transform(
    protocol.begin(), protocol.end(), protocol.begin(), ::tolower);
  if (protocol == "http") {
    return ProtocolType::HTTP;
  } else if (protocol == "grpc") {
    return ProtocolType::GRPC;
  }

  std::cerr
    << "unexpected protocol type \"" << str
    << "\", expecting HTTP or gRPC" << std::endl;
  exit(1);

  return ProtocolType::HTTP;
}

nic::Error
StartProfile(
  const std::string& url, const ProtocolType protocol,
  const bool verbose = false)
{
  std::unique_ptr<nic::ProfileContext> ctx;
  nic::Error err;
  if (protocol == ProtocolType::HTTP) {
    err = nic::ProfileHttpContext::Create(&ctx, url, verbose);
  } else {
    err = nic::ProfileGrpcContext::Create(&ctx, url, verbose);
  }
  if (!err.IsOk()) {
    return err;
  }

  return ctx->StartProfile();
}

nic::Error
StopProfile(
  const std::string& url, const ProtocolType protocol,
  const bool verbose = false)
{
  std::unique_ptr<nic::ProfileContext> ctx;
   nic::Error err;
  if (protocol == ProtocolType::HTTP) {
    err = nic::ProfileHttpContext::Create(&ctx, url, verbose);
  } else {
    err = nic::ProfileGrpcContext::Create(&ctx, url, verbose);
  }
  if (!err.IsOk()) {
    return err;
  }

  return ctx->StopProfile();
}

nic::Error
GetModelStatus(
  ni::ModelStatus* model_status, const std::string& url,
  const ProtocolType protocol,
  const std::string& model_name, const bool verbose = false)
{
  std::unique_ptr<nic::ServerStatusContext> ctx;
  nic::Error err;
  if (protocol == ProtocolType::HTTP) {
    err = nic::ServerStatusHttpContext::Create(
      &ctx, url, model_name, verbose);
  } else {
    err = nic::ServerStatusGrpcContext::Create(
      &ctx, url, model_name, verbose);
  }
  if (err.IsOk()) {
    ni::ServerStatus server_status;
    err = ctx->GetServerStatus(&server_status);
    if (err.IsOk()) {
      const auto& itr = server_status.model_status().find(model_name);
      if (itr == server_status.model_status().end()) {
        err =
          nic::Error(
            ni::RequestStatusCode::INTERNAL, "unable to find status for model");
      } else {
        model_status->CopyFrom(itr->second);
      }
    }
  }

  return err;
}

nic::Error
Report(
  const size_t thread_cnt, const int model_version, const size_t batch_size,
  const struct timespec& client_start_time,
  const struct timespec& client_end_time, const ni::ModelStatus& start_status,
  const ni::ModelStatus& end_status)
{
  nic::Error err(ni::RequestStatusCode::SUCCESS);

  // If model_version is -1 then look in the end status to find the
  // latest (highest valued version) and use that as the version.
  uint32_t status_model_version = 0;
  if (model_version < 0) {
    for (const auto& vp : end_status.version_status()) {
       status_model_version = std::max(status_model_version, vp.first);
    }
  }

  const auto& vend_itr = end_status.version_status().find(status_model_version);
  if (vend_itr == end_status.version_status().end()) {
    err =
      nic::Error(
        ni::RequestStatusCode::INTERNAL, "missing model version status");
  } else {
    const auto& end_itr = vend_itr->second.infer_stats().find(batch_size);
    if (end_itr == vend_itr->second.infer_stats().end()) {
      err =
        nic::Error(
          ni::RequestStatusCode::INTERNAL, "missing inference stats");
    } else {
      uint64_t start_cnt = 0;
      uint64_t start_cumm_time_ns = 0;
      uint64_t start_run_wait_time_ns = 0;
      uint64_t start_run_time_ns = 0;

      const auto& vstart_itr =
        start_status.version_status().find(status_model_version);
      if (vstart_itr != start_status.version_status().end()) {
        const auto& start_itr =
          vstart_itr->second.infer_stats().find(batch_size);
        if (start_itr != vstart_itr->second.infer_stats().end()) {
          start_cnt = start_itr->second.success().count();
          start_cumm_time_ns = start_itr->second.success().total_time_ns();
          start_run_wait_time_ns = start_itr->second.run_wait().total_time_ns();
          start_run_time_ns = start_itr->second.run().total_time_ns();
        }
      }

      const uint64_t cnt =
        batch_size * (end_itr->second.success().count() - start_cnt);

      const uint64_t cumm_time_ns =
        end_itr->second.success().total_time_ns() - start_cumm_time_ns;
      const uint64_t cumm_time_us = cumm_time_ns / 1000;
      const uint64_t cumm_avg_us = cumm_time_us / cnt;

      const uint64_t run_wait_time_ns =
        end_itr->second.run_wait().total_time_ns() - start_run_wait_time_ns;
      const uint64_t run_wait_time_us = run_wait_time_ns / 1000;
      const uint64_t run_wait_avg_us = run_wait_time_us / cnt;

      const uint64_t run_time_ns =
        end_itr->second.run().total_time_ns() - start_run_time_ns -
        run_wait_time_ns;
      const uint64_t run_time_us = run_time_ns / 1000;
      const uint64_t run_avg_us = run_time_us / cnt;

      std::cout << "*** Results ***" << std::endl;
      std::cout << "Thread count: " << thread_cnt << std::endl;
      std::cout << "Batch size: " << batch_size << std::endl;
      std::cout
        << "Server: " << std::endl
        << "  Inference count: " << cnt << std::endl
        << "  Cumulative Time: " << cumm_time_us << " usec (avg "
        << cumm_avg_us << " usec)" << std::endl
        << "    Run Wait: " << run_wait_time_us << " usec (avg "
        << run_wait_avg_us << " usec)" << std::endl
        << "    Run: " << run_time_us << " usec (avg "
        << run_avg_us << " usec)" << std::endl;

      uint64_t client_start_ns =
        client_start_time.tv_sec * ni::NANOS_PER_SECOND +
        client_start_time.tv_nsec;
      uint64_t client_end_ns =
        client_end_time.tv_sec * ni::NANOS_PER_SECOND + client_end_time.tv_nsec;
      uint64_t client_duration_ns =
        (client_start_ns > client_end_ns) ? 0 : client_end_ns - client_start_ns;
      uint64_t client_duration_us = client_duration_ns / 1000;
      float client_duration_sec =
        (float)client_duration_ns / ni::NANOS_PER_SECOND;

      std::cout
        << "Client: " << std::endl
        << "  Inference count: " << cnt << std::endl
        << "  Total Time: " << client_duration_us << " usec" << std::endl
        << "  Throughput: " << ((int)(cnt / client_duration_sec)) << " infer/sec"
        << std::endl;
    }
  }

  return err;
}

void
Infer(
  std::shared_ptr<nic::Error> err, const std::string& url,
  const ProtocolType protocol,
  const std::string& model_name, const int model_version,
  const size_t pass_cnt, const size_t batch_size, const bool verbose = false)
{
  // Create the context for inference of the specified model.
  std::unique_ptr<nic::InferContext> ctx;
  if (protocol == ProtocolType::HTTP) {
    *err = nic::InferHttpContext::Create(
      &ctx, url, model_name, model_version, verbose);
  } else {
    *err = nic::InferGrpcContext::Create(
      &ctx, url, model_name, model_version, verbose);
  }
  if (!err->IsOk()) {
    return;
  }

  if (batch_size > ctx->MaxBatchSize()) {
    *err =
      nic::Error(
        ni::RequestStatusCode::INVALID_ARG,
        "expecting batch size <= " + std::to_string(ctx->MaxBatchSize()) +
        " for model '" + ctx->ModelName() + "'");
    return;
  }

  // Prepare context for 'batch_size' batches. Request that all
  // outputs be returned.
  std::unique_ptr<nic::InferContext::Options> options;
  *err = nic::InferContext::Options::Create(&options);
  if (!err->IsOk()) {
    return;
  }

  options->SetBatchSize(batch_size);
  for (const auto& output : ctx->Outputs()) {
    options->AddRawResult(output);
  }

  *err = ctx->SetRunOptions(*options);
  if (!err->IsOk()) {
    return;
  }

  // Create a randomly initialized buffer that is large enough to
  // provide the largest needed input. We (re)use this buffer for all
  // input values.
  size_t max_input_byte_size = 0;
  for (const auto& input : ctx->Inputs()) {
    max_input_byte_size = std::max(max_input_byte_size, input->ByteSize());
  }

  std::vector<uint8_t> input_buf(max_input_byte_size);
  for (size_t i = 0; i < input_buf.size(); ++i) {
    input_buf[i] = rand();
  }

  // Run inference for a specified number of passes.
  for (size_t i = 0; i < pass_cnt; ++i) {
    // Initialize inputs to use random values...
    for (const auto& input : ctx->Inputs()) {
      *err = input->Reset();
      if (!err->IsOk()) {
        return;
      }

      for (size_t i = 0; i < batch_size; ++i) {
        *err = input->SetRaw(&input_buf[0], input->ByteSize());
        if (!err->IsOk()) {
          return;
        }
      }
    }

    // Run inference to get output
    std::vector<std::unique_ptr<nic::InferContext::Result>> results;
    *err = ctx->Run(&results);
    if (!err->IsOk()) {
      return;
    }

    // Stop inferencing if an early exit has been signaled.
    if (early_exit) {
      break;
    }
  }
}

void
Usage(char** argv, const std::string& msg = std::string())
{
  if (!msg.empty()) {
    std::cerr << "error: " << msg << std::endl;
  }

  std::cerr
    << "Usage: " << argv[0] << " [options] <image filename>" << std::endl;
  std::cerr << "\t-v" << std::endl;
  std::cerr << "\t-b <batch size>" << std::endl;
  std::cerr << "\t-t <thread count>" << std::endl;
  std::cerr << "\t-w <warmup passes>" << std::endl;
  std::cerr << "\t-p <measurement passes>" << std::endl;
  std::cerr << "\t-n" << std::endl;
  std::cerr << "\t-m <model name>" << std::endl;
  std::cerr << "\t-x <model version>" << std::endl;
  std::cerr << "\t-u <URL for inference service>" << std::endl;
  std::cerr << "\t-i <Protocol used to communicate with inference service>"
    << std::endl;
  std::cerr << std::endl;
  std::cerr
    << "The -n flag enables profiling for the duration of the run" << std::endl;
  std::cerr
    << "If -x is not specified the most recent version (that is, the highest "
    << "numbered version) of the model will be used." << std::endl;
  std::cerr
    << "For -i, available protocols are gRPC and HTTP. Default is HTTP."
    << std::endl;

  exit(1);
}

} //namespace

int
main(int argc, char** argv)
{
  bool verbose = false;
  bool profile = false;
  int32_t batch_size = 1;
  int32_t thread_cnt = 1;
  int32_t warmup_pass_cnt = 0;
  int32_t pass_cnt = 1;
  std::string model_name;
  int model_version = -1;
  std::string url("localhost:8000");
  ProtocolType protocol = ProtocolType::HTTP;

  // Parse commandline...
  int opt;
  while ((opt = getopt(argc, argv, "vnu:m:x:b:t:p:w:i:")) != -1) {
    switch (opt) {
      case 'v':
        verbose = true;
        break;
      case 'n':
        profile = true;
        break;
      case 'u':
        url = optarg;
        break;
      case 'm':
        model_name = optarg;
        break;
      case 'x':
        model_version = atoi(optarg);
        break;
      case 'b':
        batch_size = atoi(optarg);
        break;
      case 't':
        thread_cnt = atoi(optarg);
        break;
      case 'p':
        pass_cnt = atoi(optarg);
        break;
      case 'w':
        warmup_pass_cnt = atoi(optarg);
        break;
      case 'i':
        protocol = ParseProtocol(optarg);
        break;
      case '?':
        Usage(argv);
        break;
    }
  }

  if (model_name.empty()) { Usage(argv, "-m flag must be specified"); }
  if (batch_size <= 0) { Usage(argv, "batch size must be > 0"); }
  if (thread_cnt <= 0) { Usage(argv, "thread count must be > 0"); }
  if (pass_cnt <= 0) { Usage(argv, "pass count must be > 0"); }
  if (warmup_pass_cnt < 0) { Usage(argv, "warmup pass count must be >= 0"); }

  // trap SIGINT to allow threads to exit gracefully
  signal(SIGINT, SignalHandler);

  // Warmup server if requested.
  if (warmup_pass_cnt > 0) {
    std::cout << "*** Warming up ***" << std::endl;
    std::shared_ptr<nic::Error> warmup_status =
      std::make_shared<nic::Error>(ni::RequestStatusCode::SUCCESS);
    Infer(
      warmup_status, url, protocol, model_name, model_version,
      warmup_pass_cnt, batch_size, verbose);
    if (!warmup_status->IsOk()) {
      std::cerr << "warmup failed: " << *warmup_status << std::endl;
      exit(1);
    }
  }

  // Get server status before running inferences.
  ni::ModelStatus start_status;
  nic::Error err = GetModelStatus(
    &start_status, url, protocol, model_name, verbose);
  if (!err.IsOk()) {
    std::cerr << err << std::endl;
    exit(1);
  }

  // Start profiling on the server if requested.
  if (profile) {
    err = StartProfile(url, protocol, verbose);
    if (!err.IsOk()) {
      std::cerr << err << std::endl;
      exit(1);
    }
  }

  std::cout << "*** Begin ***" << std::endl;

  struct timespec start_time;
  clock_gettime(CLOCK_MONOTONIC, &start_time);

  // Launch all threads for inferencing
  std::vector<std::thread> threads;
  std::vector<std::shared_ptr<nic::Error>> threads_status;
  for (int i = 0; i < thread_cnt; ++i) {
    threads_status.emplace_back(new nic::Error(ni::RequestStatusCode::SUCCESS));
    threads.emplace_back(
      Infer, threads_status.back(), url, protocol, model_name, model_version,
      pass_cnt, batch_size, verbose);
  }

  // Wait for all threads to complete.
  int ret = 0;
  size_t cnt = 0;
  for (auto& thread : threads) {
    thread.join();
    if (!threads_status[cnt]->IsOk()) {
      std::cerr << *threads_status[cnt] << std::endl;
      ret = 1;
    }
    cnt++;
  }

  struct timespec end_time;
  clock_gettime(CLOCK_MONOTONIC, &end_time);

  std::cout << "*** End ***" << std::endl;

  // Stop profiling on the server if requested.
  if (profile) {
    err = StopProfile(url, protocol, verbose);
    if (!err.IsOk()) {
      std::cerr << err << std::endl;
      exit(1);
    }
  }

  // Get server status and then print report on difference between
  // before and after status.
  ni::ModelStatus end_status;
  err = GetModelStatus(&end_status, url, protocol, model_name, verbose);
  if (!err.IsOk()) {
    std::cerr << err << std::endl;
    exit(1);
  }

  err = Report(
    thread_cnt, model_version, batch_size, start_time, end_time,
    start_status, end_status);
  if (!err.IsOk()) {
    std::cerr << err << std::endl;
    exit(1);
  }

  return ret;
}
