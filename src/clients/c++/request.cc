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

#include <iostream>
#include <memory>
#include <curl/curl.h>
#include <google/protobuf/text_format.h>
#include "src/core/constants.h"

namespace nvidia { namespace inferenceserver { namespace client {

//==============================================================================

// Global initialization for libcurl. Libcurl requires global
// initialization before any other threads are created and before any
// curl methods are used. The curl_global static object is used to
// perform this initialization.
class CurlGlobal {
public:
  CurlGlobal();
  ~CurlGlobal();

  const Error& Status() const { return err_; }

private:
  Error err_;
};

CurlGlobal::CurlGlobal()
  : err_(RequestStatusCode::SUCCESS)
{
  if (curl_global_init(CURL_GLOBAL_ALL) != 0) {
    err_ = Error(RequestStatusCode::INTERNAL, "global initialization failed");
  }
}

CurlGlobal::~CurlGlobal()
{
  curl_global_cleanup();
}

static CurlGlobal curl_global;

//==============================================================================

// Use map to keep track of gRPC channels. <key, value> : <url, Channel*>
// If context is created on url that has established Channel, then reuse it.
std::map<std::string, std::shared_ptr<grpc::Channel>> grpc_channel_map_;
std::shared_ptr<grpc::Channel> GetChannel(const std::string& url)
{
  const auto& channel_itr = grpc_channel_map_.find(url);
  if (channel_itr != grpc_channel_map_.end()) {
    return channel_itr->second;
  } else {
    grpc::ChannelArguments arguments;
    arguments.SetMaxSendMessageSize(MAX_GRPC_MESSAGE_SIZE);
    arguments.SetMaxReceiveMessageSize(MAX_GRPC_MESSAGE_SIZE);
    std::shared_ptr<grpc::Channel> channel = grpc::CreateCustomChannel(
      url, grpc::InsecureChannelCredentials(), arguments);
    grpc_channel_map_.insert(std::make_pair(url, channel));
    return channel;
  }
}

//==============================================================================

const Error Error::Success(RequestStatusCode::SUCCESS);

Error::Error(RequestStatusCode code, const std::string& msg)
  : code_(code), msg_(msg), request_id_(0)
{
}

Error::Error(RequestStatusCode code)
  : code_(code), request_id_(0)
{
}

Error::Error(const RequestStatus& status)
  : Error(status.code(), status.msg())
{
  server_id_ = status.server_id();
  request_id_ = status.request_id();
}

std::ostream&
operator<<(std::ostream& out, const Error& err)
{
  out
    << "[" << err.server_id_ << " " << err.request_id_ << "] "
    << RequestStatusCode_Name(err.code_);
  if (!err.msg_.empty()) {
    out << " - " << err.msg_;
  }
  return out;
}

//==============================================================================

ServerHealthContext::ServerHealthContext(bool verbose)
  : verbose_(verbose)
{
}

//==============================================================================

ServerStatusContext::ServerStatusContext(bool verbose)
  : verbose_(verbose)
{
}

//==============================================================================

class OptionsImpl : public InferContext::Options {
public:
  OptionsImpl();
  ~OptionsImpl() = default;

  size_t BatchSize() const override { return batch_size_; }
  void SetBatchSize(size_t batch_size) override { batch_size_ = batch_size; }

  Error AddRawResult(
    const std::shared_ptr<InferContext::Output>& output) override;
  Error AddClassResult(
    const std::shared_ptr<InferContext::Output>& output, uint64_t k) override;

  // Options for an output
  struct OutputOptions {
    OutputOptions(InferContext::Result::ResultFormat f, uint64_t n=0)
      : result_format(f), u64(n) { }
    InferContext::Result::ResultFormat result_format;
    uint64_t u64;
  };

  using OutputOptionsPair =
    std::pair<std::shared_ptr<InferContext::Output>, OutputOptions>;

  const std::vector<OutputOptionsPair>& Outputs() const { return outputs_; }

private:
  size_t batch_size_;
  std::vector<OutputOptionsPair> outputs_;
};

OptionsImpl::OptionsImpl()
  : batch_size_(0)
{
}

Error
OptionsImpl::AddRawResult(const std::shared_ptr<InferContext::Output>& output)
{
  outputs_.emplace_back(
    std::make_pair(
      output, OutputOptions(InferContext::Result::ResultFormat::RAW)));
  return Error::Success;
}

Error
OptionsImpl::AddClassResult(
  const std::shared_ptr<InferContext::Output>& output, uint64_t k)
{
  outputs_.emplace_back(
    std::make_pair(
      output, OutputOptions(InferContext::Result::ResultFormat::CLASS, k)));
  return Error::Success;
}

Error
InferContext::Options::Create(std::unique_ptr<InferContext::Options>* options)
{
  options->reset(new OptionsImpl());
  return Error::Success;
}

//==============================================================================

class InputImpl : public InferContext::Input {
public:
  InputImpl(const ModelInput& mio);
  ~InputImpl() = default;

  const std::string& Name() const override { return mio_.name(); }
  size_t ByteSize() const override { return byte_size_; }
  DataType DType() const override { return mio_.data_type(); }
  ModelInput::Format Format() const override { return mio_.format(); }
  const DimsList& Dims() const override { return mio_.dims(); }

  void SetBatchSize(size_t batch_size) { batch_size_ = batch_size; }

  Error Reset() override;
  Error SetRaw(const std::vector<uint8_t>& input) override;
  Error SetRaw(const uint8_t* input, size_t input_byte_size) override;

  // Copy into 'buf' up to 'size' bytes of this input's data. Return
  // the actual amount copied in 'input_bytes' and if the end of input
  // is reached in 'end_of_input'
  Error GetNext(
    uint8_t* buf, size_t size, size_t* input_bytes, bool* end_of_input);

  // Copy the pointer of the raw buffer at 'batch_idx' into 'buf'
  Error GetRaw(size_t batch_idx, const uint8_t** buf) const;

  // Prepare to send this input as part of a request.
  Error PrepareForRequest();

private:
  const ModelInput mio_;
  const size_t byte_size_;
  size_t batch_size_;
  std::vector<const uint8_t*> bufs_;
  size_t bufs_idx_, buf_pos_;
};

InputImpl::InputImpl(const ModelInput& mio)
  : mio_(mio), byte_size_(GetSize(mio)),
    batch_size_(0), bufs_idx_(0), buf_pos_(0)
{
}

Error
InputImpl::SetRaw(const uint8_t* input, size_t input_byte_size)
{
  if (input_byte_size != byte_size_) {
    bufs_.clear();
    return
      Error(
        RequestStatusCode::INVALID_ARG,
        "invalid size " + std::to_string(input_byte_size) +
        " bytes for input '" + Name() + "', expects " +
        std::to_string(byte_size_) + " bytes");
  }

  if (bufs_.size() >= batch_size_) {
    bufs_.clear();
    return
      Error(
        RequestStatusCode::INVALID_ARG,
        "expecting " + std::to_string(batch_size_) +
        " invocations of SetRaw for input '" + Name() +
        "', one per batch entry");
  }

  bufs_.push_back(input);
  return Error::Success;
}

Error
InputImpl::SetRaw(const std::vector<uint8_t>& input)
{
  return SetRaw(&input[0], input.size());
}

Error
InputImpl::GetNext(
  uint8_t* buf, size_t size, size_t* input_bytes, bool* end_of_input)
{
  size_t total_size = 0;

  while ((bufs_idx_ < bufs_.size()) && (size > 0)) {
    const size_t csz = std::min(byte_size_ - buf_pos_, size);
    if (csz > 0) {
      const uint8_t* input_ptr = bufs_[bufs_idx_] + buf_pos_;
      std::copy(input_ptr, input_ptr + csz, buf);
      buf_pos_ += csz;
      buf += csz;
      size -= csz;
      total_size += csz;
    }

    if (buf_pos_ == byte_size_) {
      bufs_idx_++;
      buf_pos_ = 0;
    }
  }

  *input_bytes = total_size;
  *end_of_input = (bufs_idx_ >= bufs_.size());
  return Error::Success;
}

Error
InputImpl::GetRaw(
  size_t batch_idx, const uint8_t** buf) const
{
  if (batch_idx >= batch_size_) {
    return
      Error(
        RequestStatusCode::INVALID_ARG,
        "unexpected batch entry " + std::to_string(batch_idx) +
        " requested for input '" + Name() +
        "', batch size is " + std::to_string(batch_size_));
  }

  *buf = bufs_[batch_idx];
  return Error::Success;
}

Error
InputImpl::Reset()
{
  bufs_.clear();
  bufs_idx_ = 0;
  buf_pos_ = 0;
  return Error::Success;
}

Error
InputImpl::PrepareForRequest()
{
  if (bufs_.size() != batch_size_) {
    return
      Error(
        RequestStatusCode::INVALID_ARG,
        "expecting " + std::to_string(batch_size_) +
        " invocations of SetRaw for input '" + Name() +
        "', have " + std::to_string(bufs_.size()));
  }

  // Reset position so request sends entire input.
  bufs_idx_ = 0;
  buf_pos_ = 0;
  return Error::Success;
}

//==============================================================================

class OutputImpl : public InferContext::Output {
public:
  OutputImpl(const ModelOutput& mio);
  ~OutputImpl() = default;

  const std::string& Name() const override { return mio_.name(); }
  size_t ByteSize() const override { return byte_size_; }
  DataType DType() const override { return mio_.data_type(); }
  const DimsList& Dims() const override { return mio_.dims(); }

  InferContext::Result::ResultFormat ResultFormat() const {
    return result_format_;
  }
  void SetResultFormat(InferContext::Result::ResultFormat result_format) {
    result_format_ = result_format;
  }

  private:
  const ModelOutput mio_;
  const size_t byte_size_;
  InferContext::Result::ResultFormat result_format_;
};

OutputImpl::OutputImpl(const ModelOutput& mio)
  : mio_(mio), byte_size_(GetSize(mio)),
    result_format_(InferContext::Result::ResultFormat::RAW)
{
}

//==============================================================================

class ResultImpl : public InferContext::Result {
public:
  ResultImpl(
    const std::shared_ptr<InferContext::Output>& output, uint64_t batch_size,
    InferContext::Result::ResultFormat result_format);
  ~ResultImpl() = default;

  const std::string& ModelName() const override { return model_name_; }
  uint32_t ModelVersion() const override { return model_version_; }

  const std::shared_ptr<InferContext::Output> GetOutput() const override {
    return output_;
  }

  Error GetRaw(
    size_t batch_idx, const std::vector<uint8_t>** buf) const override;
  Error GetRawAtCursor(
    size_t batch_idx, const uint8_t** buf, size_t adv_byte_size) override;
  Error GetClassCount(size_t batch_idx, size_t* cnt) const override;
  Error GetClassAtCursor(size_t batch_idx, ClassResult* result) override;
  Error ResetCursors() override;
  Error ResetCursor(size_t batch_idx) override;

  // Get the result format for this result.
  InferContext::Result::ResultFormat ResultFormat() const {
    return result_format_;
  }

  // Set information about the model that produced this result.
  void SetModel(const std::string& name, const uint32_t version) {
    model_name_ = name;
    model_version_ = version;
  }

  // Set results for a CLASS format result.
  void SetClassResult(const InferResponseHeader::Output& result) {
    class_result_ = result;
  }

  // For RAW format result, copy into the output up to 'size' bytes of
  // output data from 'buf'. Return the actual amount copied in
  // 'result_bytes'.
  Error SetNextRawResult(
    const uint8_t* buf, size_t size, size_t* result_bytes);

private:
  const std::shared_ptr<InferContext::Output> output_;
  const size_t byte_size_;
  const size_t batch_size_;
  const InferContext::Result::ResultFormat result_format_;

  std::vector<std::vector<uint8_t>> bufs_;
  size_t bufs_idx_;
  std::vector<size_t> bufs_pos_;

  std::string model_name_;
  uint32_t model_version_;

  InferResponseHeader::Output class_result_;
  std::vector<size_t> class_pos_;
};

ResultImpl::ResultImpl(
  const std::shared_ptr<InferContext::Output>& output, uint64_t batch_size,
  InferContext::Result::ResultFormat result_format)
  : output_(output), byte_size_(output->ByteSize()),
    batch_size_(batch_size), result_format_(result_format),
    bufs_(batch_size), bufs_idx_(0), bufs_pos_(batch_size),
    class_pos_(batch_size)
{
}

Error
ResultImpl::GetRaw(
  size_t batch_idx, const std::vector<uint8_t>** buf) const
{
  if (result_format_ != InferContext::Result::ResultFormat::RAW) {
    return
      Error(
        RequestStatusCode::UNSUPPORTED,
        "raw result not available for non-RAW output '" +
        output_->Name() + "'");
  }

  if (batch_idx >= batch_size_) {
    return
      Error(
        RequestStatusCode::INVALID_ARG,
        "unexpected batch entry " + std::to_string(batch_idx) +
        " requested for output '" + output_->Name() +
        "', batch size is " + std::to_string(batch_size_));
  }

  *buf = &bufs_[batch_idx];
  return Error::Success;
}

Error
ResultImpl::GetRawAtCursor(
  size_t batch_idx, const uint8_t** buf, size_t adv_byte_size)
{
  if (result_format_ != InferContext::Result::ResultFormat::RAW) {
    return
      Error(
        RequestStatusCode::UNSUPPORTED,
        "raw result not available for non-RAW output '" +
        output_->Name() + "'");
  }

  if (batch_idx >= batch_size_) {
    return
      Error(
        RequestStatusCode::INVALID_ARG,
        "unexpected batch entry " + std::to_string(batch_idx) +
        "requested for output '" + output_->Name() +
        "', batch size is " + std::to_string(batch_size_));
  }

  if ((bufs_pos_[batch_idx] + adv_byte_size) > byte_size_) {
    return
      Error(
        RequestStatusCode::UNSUPPORTED,
        "attempt to read beyond end of result for output output '" +
        output_->Name() + "'");
  }

  *buf = &bufs_[batch_idx][bufs_pos_[batch_idx]];
  bufs_pos_[batch_idx] += adv_byte_size;
  return Error::Success;
}

Error
ResultImpl::GetClassCount(size_t batch_idx, size_t* cnt) const
{
  if (result_format_ != InferContext::Result::ResultFormat::CLASS) {
    return
      Error(
        RequestStatusCode::UNSUPPORTED,
        "class result not available for non-CLASS output '" +
        output_->Name() + "'");
  }

  // Number of classifications should equal expected batch size but
  // check both to be careful and to protext class_pos_ accesses.
  if ((batch_idx >= (size_t)class_result_.batch_classes().size()) ||
      (batch_idx >= batch_size_)) {
    return
      Error(
        RequestStatusCode::INVALID_ARG,
        "unexpected batch entry " + std::to_string(batch_idx) +
        "requested for output '" + output_->Name() +
        "', batch size is " + std::to_string(batch_size_));
  }

  const InferResponseHeader::Output::Classes& classes =
    class_result_.batch_classes(batch_idx);

  *cnt = classes.cls().size();
  return Error::Success;
}

Error
ResultImpl::GetClassAtCursor(
  size_t batch_idx, InferContext::Result::ClassResult* result)
{
  if (result_format_ != InferContext::Result::ResultFormat::CLASS) {
    return
      Error(
        RequestStatusCode::UNSUPPORTED,
        "class result not available for non-CLASS output '" +
        output_->Name() + "'");
  }

  // Number of classifications should equal expected batch size but
  // check both to be careful and to protext class_pos_ accesses.
  if ((batch_idx >= (size_t)class_result_.batch_classes().size()) ||
      (batch_idx >= batch_size_)) {
    return
      Error(
        RequestStatusCode::INVALID_ARG,
        "unexpected batch entry " + std::to_string(batch_idx) +
        "requested for output '" + output_->Name() +
        "', batch size is " + std::to_string(batch_size_));
  }

  const InferResponseHeader::Output::Classes& classes =
    class_result_.batch_classes(batch_idx);

  if (class_pos_[batch_idx] >= (size_t)classes.cls().size()) {
    return
      Error(
        RequestStatusCode::UNSUPPORTED,
        "attempt to read beyond end of result for output output '" +
        output_->Name() + "'");
  }

  const InferResponseHeader::Output::Class& cls =
    classes.cls(class_pos_[batch_idx]);

  result->idx = cls.idx();
  result->value = cls.value();
  result->label = cls.label();

  class_pos_[batch_idx]++;
  return Error::Success;
}

Error
ResultImpl::ResetCursors()
{
  std::fill(bufs_pos_.begin(), bufs_pos_.end(), 0);
  std::fill(class_pos_.begin(), class_pos_.end(), 0);
  return Error::Success;
}

Error
ResultImpl::ResetCursor(size_t batch_idx)
{
  if (batch_idx >= batch_size_) {
    return
      Error(
        RequestStatusCode::INVALID_ARG,
        "unexpected batch entry " + std::to_string(batch_idx) +
        "requested for output '" + output_->Name() +
        "', batch size is " + std::to_string(batch_size_));
  }

  bufs_pos_[batch_idx] = 0;
  class_pos_[batch_idx] = 0;
  return Error::Success;
}

Error
ResultImpl::SetNextRawResult(
  const uint8_t* buf, size_t size, size_t* result_bytes)
{
  size_t total_size = 0;

  while ((bufs_idx_ < bufs_.size()) && (size > 0)) {
    const size_t csz = std::min(byte_size_ - bufs_pos_[bufs_idx_], size);
    if (csz > 0) {
      std::copy(buf, buf + csz, std::back_inserter(bufs_[bufs_idx_]));
      bufs_pos_[bufs_idx_] += csz;
      buf += csz;
      size -= csz;
      total_size += csz;
    }

    if (bufs_pos_[bufs_idx_] == byte_size_) {
      bufs_idx_++;
    }
  }

  *result_bytes = total_size;
  return Error::Success;
}

//==============================================================================

InferContext::RequestTimers::RequestTimers()
{
  Reset();
}

Error
InferContext::RequestTimers::Reset()
{
  request_start_.tv_sec = 0;
  request_end_.tv_sec = 0;
  send_start_.tv_sec = 0;
  send_end_.tv_sec = 0;
  receive_start_.tv_sec = 0;
  receive_end_.tv_sec = 0;
  request_start_.tv_nsec = 0;
  request_end_.tv_nsec = 0;
  send_start_.tv_nsec = 0;
  send_end_.tv_nsec = 0;
  receive_start_.tv_nsec = 0;
  receive_end_.tv_nsec = 0;
  return Error::Success;
}

Error
InferContext::RequestTimers::Record(Kind kind)
{
  switch (kind) {
    case Kind::REQUEST_START:
      clock_gettime(CLOCK_MONOTONIC, &request_start_);
      break;
    case Kind::REQUEST_END:
      clock_gettime(CLOCK_MONOTONIC, &request_end_);
      break;
    case Kind::SEND_START:
      clock_gettime(CLOCK_MONOTONIC, &send_start_);
      break;
    case Kind::SEND_END:
      clock_gettime(CLOCK_MONOTONIC, &send_end_);
      break;
    case Kind::RECEIVE_START:
      clock_gettime(CLOCK_MONOTONIC, &receive_start_);
      break;
    case Kind::RECEIVE_END:
      clock_gettime(CLOCK_MONOTONIC, &receive_end_);
      break;
  }
  return Error::Success;
}

//==============================================================================

InferContext::InferContext(
  const std::string& model_name, int model_version, bool verbose)
  : model_name_(model_name), model_version_(model_version),
    verbose_(verbose), total_input_byte_size_(0), batch_size_(0),
    input_pos_idx_(0), result_pos_idx_(0)
{
}

Error
InferContext::GetInput(
  const std::string& name, std::shared_ptr<Input>* input) const
{
  for (const auto& io : inputs_) {
    if (io->Name() == name) {
      *input = io;
      return Error::Success;
    }
  }

  return
    Error(
      RequestStatusCode::INVALID_ARG,
      "unknown input '" + name + "' for '" + model_name_ + "'");
}

Error
InferContext::GetOutput(
  const std::string& name, std::shared_ptr<Output>* output) const
{
  for (const auto& io : outputs_) {
    if (io->Name() == name) {
      *output = io;
      return Error::Success;
    }
  }

  return
    Error(
      RequestStatusCode::INVALID_ARG,
      "unknown output '" + name + "' for '" + model_name_ + "'");
}

Error
InferContext::SetRunOptions(const InferContext::Options& boptions)
{
  const OptionsImpl& options = reinterpret_cast<const OptionsImpl&>(boptions);

  if (options.BatchSize() > max_batch_size_) {
    return
      Error(
        RequestStatusCode::INVALID_ARG,
        "run batch size " + std::to_string(options.BatchSize()) +
        " exceeds maximum batch size " + std::to_string(max_batch_size_) +
        " allowed for model '" + model_name_ + "'");
  }

  batch_size_ = options.BatchSize();
  total_input_byte_size_ = 0;

  // Create the InferRequestHeader protobuf. This protobuf will be
  // used for all subsequent requests.
  infer_request_.Clear();

  infer_request_.set_batch_size(batch_size_);

  for (const auto& io : inputs_) {
    reinterpret_cast<InputImpl*>(io.get())->SetBatchSize(batch_size_);
    total_input_byte_size_ += io->ByteSize() * batch_size_;

    auto rinput = infer_request_.add_input();
    rinput->set_name(io->Name());
    rinput->set_byte_size(io->ByteSize());
  }

  requested_outputs_.clear();
  requested_results_.clear();

  for (const auto& p : options.Outputs()) {
    const std::shared_ptr<Output>& output = p.first;
    const OptionsImpl::OutputOptions& ooptions = p.second;

    reinterpret_cast<OutputImpl*>(output.get())->
      SetResultFormat(ooptions.result_format);
    requested_outputs_.emplace_back(output);

    auto routput = infer_request_.add_output();
    routput->set_name(output->Name());
    routput->set_byte_size(output->ByteSize());
    if (ooptions.result_format == Result::ResultFormat::CLASS) {
      routput->mutable_cls()->set_count(ooptions.u64);
    }
  }

  return Error::Success;
}

Error
InferContext::GetStat(Stat* stat)
{
  stat->completed_request_count = context_stat_.completed_request_count;
  stat->cumulative_total_request_time_ns =
    context_stat_.cumulative_total_request_time_ns;
  stat->cumulative_send_time_ns = context_stat_.cumulative_send_time_ns;
  stat->cumulative_receive_time_ns = context_stat_.cumulative_receive_time_ns;
  return Error::Success;
}

Error
InferContext::PreRunProcessing()
{
  infer_response_buffer_.clear();

  // Reset all the position indicators so that we send all inputs
  // correctly.
  request_status_.Clear();
  input_pos_idx_ = 0;
  result_pos_idx_ = 0;

  for (auto& io : inputs_) {
    reinterpret_cast<InputImpl*>(io.get())->PrepareForRequest();
  }

  // Initialize the results vector to collect the requested results.
  requested_results_.clear();
  for (const auto& io : requested_outputs_) {
    std::unique_ptr<ResultImpl>
      rp(
        new ResultImpl(
          io, batch_size_,
          reinterpret_cast<OutputImpl*>(io.get())->ResultFormat()));
    requested_results_.emplace_back(std::move(rp));
  }
  return Error::Success;
}

Error
InferContext::PostRunProcessing(InferResponseHeader& infer_response)
{
  // At this point, the RAW requested results have their result values
  // set. Now need to initialize non-RAW results.
  for (auto& rr : requested_results_) {
    ResultImpl* r = reinterpret_cast<ResultImpl*>(rr.get());
    r->SetModel(infer_response.model_name(), infer_response.model_version());
    switch (r->ResultFormat()) {
      case Result::ResultFormat::RAW:
        r->ResetCursors();
        break;

      case Result::ResultFormat::CLASS: {
        for (const auto& ir : infer_response.output()) {
          if (ir.name() == r->GetOutput()->Name()) {
            r->SetClassResult(ir);
            break;
          }
        }
        break;
      }
    }
  }
  return Error::Success;
}

Error
InferContext::GetNextInput(uint8_t* buf, size_t size, size_t* input_bytes)
{
  *input_bytes = 0;

  while ((size > 0) && (input_pos_idx_ < inputs_.size())) {
    InputImpl* io =
      reinterpret_cast<InputImpl*>(inputs_[input_pos_idx_].get());
    size_t ib = 0;
    bool eoi = false;
    Error err = io->GetNext(buf, size, &ib, &eoi);
    if (!err.IsOk()) {
      return err;
    }
    // If input was completely read then move to the next.
    if (eoi) {
      input_pos_idx_++;
    }
    if (ib != 0) {
      *input_bytes += ib;
      size -= ib;
      buf += ib;
    }
  }
  // Sent all input bytes
  if (input_pos_idx_ >= inputs_.size()) {
    request_timer_.Record(RequestTimers::Kind::SEND_END);
  }

  return Error::Success;
}

Error
InferContext::SetNextRawResult(
  const uint8_t* buf, size_t size, size_t* result_bytes)
{
  *result_bytes = 0;

  if (request_timer_.receive_start_.tv_sec == 0) {
    request_timer_.Record(RequestTimers::Kind::RECEIVE_START);
  }

  while ((size > 0) && (result_pos_idx_ < requested_results_.size())) {
    ResultImpl* io =
      reinterpret_cast<ResultImpl*>(requested_results_[result_pos_idx_].get());
    size_t ob = 0;

    // Only try to read raw result for RAW
    if (io->ResultFormat() == Result::ResultFormat::RAW) {
      Error err = io->SetNextRawResult(buf, size, &ob);
      if (!err.IsOk()) {
        return err;
      }
    }

    // If output couldn't accept any more bytes then move to the next.
    if (ob == 0) {
      result_pos_idx_++;
    } else {
      *result_bytes += ob;
      size -= ob;
      buf += ob;
    }
  }

  // If there is any bytes left then they belong to the response
  // header, since all the RAW results have been filled.
  if (size > 0) {
    infer_response_buffer_.append(reinterpret_cast<const char*>(buf), size);
    *result_bytes += size;
  }

  return Error::Success;
}

Error InferContext::UpdateStat(const RequestTimers& timer)
{
  uint64_t request_start_ns =
    timer.request_start_.tv_sec * NANOS_PER_SECOND +
    timer.request_start_.tv_nsec;
  uint64_t request_end_ns =
    timer.request_end_.tv_sec * NANOS_PER_SECOND +
    timer.request_end_.tv_nsec;
  uint64_t send_start_ns =
    timer.send_start_.tv_sec * NANOS_PER_SECOND +
    timer.send_start_.tv_nsec;
  uint64_t send_end_ns =
    timer.send_end_.tv_sec * NANOS_PER_SECOND +
    timer.send_end_.tv_nsec;
  uint64_t receive_start_ns =
    timer.receive_start_.tv_sec * NANOS_PER_SECOND +
    timer.receive_start_.tv_nsec;
  uint64_t receive_end_ns =
    timer.receive_end_.tv_sec * NANOS_PER_SECOND +
    timer.receive_end_.tv_nsec;
  if ((request_start_ns >= request_end_ns) ||
        (send_start_ns > send_end_ns) || (receive_start_ns > receive_end_ns)) {
    return Error(RequestStatusCode::INVALID_ARG, "Timer not set correctly.");
  }

  uint64_t request_time_ns = request_end_ns - request_start_ns;
  uint64_t send_time_ns = send_end_ns - send_start_ns;
  uint64_t receive_time_ns = receive_end_ns - receive_start_ns;

  context_stat_.completed_request_count++;
  context_stat_.cumulative_total_request_time_ns += request_time_ns;
  context_stat_.cumulative_send_time_ns += send_time_ns;
  context_stat_.cumulative_receive_time_ns += receive_time_ns;
  return Error::Success;
}

//==============================================================================

ProfileContext::ProfileContext(bool verbose)
  : verbose_(verbose)
{
}

Error
ProfileContext::StartProfile()
{
  return SendCommand("start");
}

Error
ProfileContext::StopProfile()
{
  return SendCommand("stop");
}

//==============================================================================

Error
ServerHealthHttpContext::Create(
  std::unique_ptr<ServerHealthContext>* ctx,
  const std::string& server_url, bool verbose)
{
  ctx->reset(
    static_cast<ServerHealthContext*>(
      new ServerHealthHttpContext(server_url, verbose)));
  return Error::Success;
}

ServerHealthHttpContext::ServerHealthHttpContext(
  const std::string& server_url, bool verbose)
  : ServerHealthContext(verbose), url_(server_url + "/" + kHealthRESTEndpoint)
{
}

Error
ServerHealthHttpContext::GetHealth(const std::string& url, bool* health)
{
  if (!curl_global.Status().IsOk()) {
    return curl_global.Status();
  }

  CURL* curl = curl_easy_init();
  if (!curl) {
    return
      Error(RequestStatusCode::INTERNAL, "failed to initialize HTTP client");
  }

  curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
  curl_easy_setopt(curl, CURLOPT_USERAGENT, "libcurl-agent/1.0");
  if (verbose_) {
    curl_easy_setopt(curl, CURLOPT_VERBOSE, 1L);
  }

  CURLcode res = curl_easy_perform(curl);
  if (res != CURLE_OK) {
    curl_easy_cleanup(curl);
    return
      Error(
        RequestStatusCode::INTERNAL, "HTTP client failed: " +
        std::string(curl_easy_strerror(res)));
  }

  // Must use 64-bit integer with curl_easy_getinfo
  int64_t http_code;
  curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);

  curl_easy_cleanup(curl);

  *health = (http_code == 200) ? true : false;

  return Error::Success;
}

Error
ServerHealthHttpContext::GetReady(bool* ready)
{
  return GetHealth(url_ + "/ready", ready);
}

Error
ServerHealthHttpContext::GetLive(bool* live)
{
  return GetHealth(url_ + "/live", live);
}

//==============================================================================

Error
ServerStatusHttpContext::Create(
  std::unique_ptr<ServerStatusContext>* ctx,
  const std::string& server_url, bool verbose)
{
  ctx->reset(
    static_cast<ServerStatusContext*>(
      new ServerStatusHttpContext(server_url, verbose)));
  return Error::Success;
}

Error
ServerStatusHttpContext::Create(
  std::unique_ptr<ServerStatusContext>* ctx,
  const std::string& server_url, const std::string& model_name, bool verbose)
{
  ctx->reset(
    static_cast<ServerStatusContext*>(
      new ServerStatusHttpContext(server_url, model_name, verbose)));
  return Error::Success;
}

ServerStatusHttpContext::ServerStatusHttpContext(
  const std::string& server_url, bool verbose)
  : ServerStatusContext(verbose), url_(server_url + "/" + kStatusRESTEndpoint)
{
}

ServerStatusHttpContext::ServerStatusHttpContext(
  const std::string& server_url, const std::string& model_name, bool verbose)
  : ServerStatusContext(verbose),
    url_(server_url + "/" + kStatusRESTEndpoint + "/" + model_name)
{
}

Error
ServerStatusHttpContext::GetServerStatus(ServerStatus* server_status)
{
  server_status->Clear();
  request_status_.Clear();
  response_.clear();

  if (!curl_global.Status().IsOk()) {
    return curl_global.Status();
  }

  CURL* curl = curl_easy_init();
  if (!curl) {
    return
      Error(RequestStatusCode::INTERNAL, "failed to initialize HTTP client");
  }

  // Want binary representation of the status.
  std::string full_url = url_ + "?format=binary";
  curl_easy_setopt(curl, CURLOPT_URL, full_url.c_str());
  curl_easy_setopt(curl, CURLOPT_USERAGENT, "libcurl-agent/1.0");
  if (verbose_) {
    curl_easy_setopt(curl, CURLOPT_VERBOSE, 1L);
  }

  // response headers handled by ResponseHeaderHandler()
  curl_easy_setopt(curl, CURLOPT_HEADERFUNCTION, ResponseHeaderHandler);
  curl_easy_setopt(curl, CURLOPT_HEADERDATA, this);

  // response data handled by ResponseHandler()
  curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, ResponseHandler);
  curl_easy_setopt(curl, CURLOPT_WRITEDATA, this);

  CURLcode res = curl_easy_perform(curl);
  if (res != CURLE_OK) {
    curl_easy_cleanup(curl);
    return
      Error(
        RequestStatusCode::INTERNAL, "HTTP client failed: " +
        std::string(curl_easy_strerror(res)));
  }

  // Must use 64-bit integer with curl_easy_getinfo
  int64_t http_code;
  curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);

  curl_easy_cleanup(curl);

  // Should have a request status, if not then create an error status.
  if (request_status_.code() == RequestStatusCode::INVALID) {
    request_status_.Clear();
    request_status_.set_code(RequestStatusCode::INTERNAL);
    request_status_.set_msg("status request did not return status");
  }

  // If request has failing HTTP status or the request's explicit
  // status is not SUCCESS, then signal an error.
  if ((http_code != 200) ||
      (request_status_.code() != RequestStatusCode::SUCCESS)) {
    return Error(request_status_);
  }

  // Parse the response as a ModelConfigList...
  if (!server_status->ParseFromString(response_)) {
    return Error(RequestStatusCode::INTERNAL, "failed to parse server status");
  }

  if (verbose_) {
    std::cout << server_status->DebugString() << std::endl;
  }

  return Error(request_status_);
}

size_t
ServerStatusHttpContext::ResponseHeaderHandler(
  void* contents, size_t size, size_t nmemb, void* userp)
{
  ServerStatusHttpContext* ctx =
    reinterpret_cast<ServerStatusHttpContext*>(userp);

  char* buf = reinterpret_cast<char*>(contents);
  size_t byte_size = size * nmemb;

  size_t idx = strlen(kStatusHTTPHeader);
  if ((idx < byte_size) &&
      !strncasecmp(buf, kStatusHTTPHeader, idx)) {
    while ((idx < byte_size) && (buf[idx] != ':')) {
      ++idx;
    }

    if (idx < byte_size) {
      std::string hdr(buf + idx + 1, byte_size - idx - 1);

      if (!google::protobuf::TextFormat::ParseFromString(
          hdr, &ctx->request_status_)) {
        ctx->request_status_.Clear();
      }
    }
  }

  return byte_size;
}

size_t
ServerStatusHttpContext::ResponseHandler(
  void* contents, size_t size, size_t nmemb, void* userp)
{
  ServerStatusHttpContext* ctx =
    reinterpret_cast<ServerStatusHttpContext*>(userp);
  uint8_t* buf = reinterpret_cast<uint8_t*>(contents);
  size_t result_bytes = size * nmemb;
  std::copy(buf, buf + result_bytes, std::back_inserter(ctx->response_));
  return result_bytes;
}

//==============================================================================

Error
InferHttpContext::Create(
  std::unique_ptr<InferContext>* ctx, const std::string& server_url,
  const std::string& model_name, int model_version, bool verbose)
{
  InferHttpContext* ctx_ptr =
    new InferHttpContext(server_url, model_name, model_version, verbose);

  // Get status of the model and create the inputs and outputs.
  std::unique_ptr<ServerStatusContext> sctx;
  Error err =
    ServerStatusHttpContext::Create(
      &sctx, server_url, model_name, verbose);
  if (err.IsOk()) {
    ServerStatus server_status;
    err = sctx->GetServerStatus(&server_status);
    if (err.IsOk()) {
      const auto& itr = server_status.model_status().find(model_name);
      if (itr == server_status.model_status().end()) {
        err =
          Error(
            RequestStatusCode::INTERNAL,
            "unable to find status information for \"" + model_name + "\"");
      } else {
        const ModelConfig& model_info = itr->second.config();

        ctx_ptr->max_batch_size_ =
          static_cast<uint64_t>(std::max(0, model_info.max_batch_size()));

        // Create inputs and outputs
        for (const auto& io : model_info.input()) {
          ctx_ptr->inputs_.emplace_back(std::make_shared<InputImpl>(io));
        }
        for (const auto& io : model_info.output()) {
          ctx_ptr->outputs_.emplace_back(std::make_shared<OutputImpl>(io));
        }
      }
    }
  }

  if (err.IsOk()) {
    ctx->reset(static_cast<InferContext*>(ctx_ptr));
  } else {
    ctx->reset();
  }

  return err;
}

InferHttpContext::InferHttpContext(
  const std::string& server_url, const std::string& model_name,
  int model_version, bool verbose)
  : InferContext(model_name, model_version, verbose), curl_(nullptr)
{
  // Process url for HTTP request
  // URL doesn't contain the version portion if using the latest version.
  url_ = server_url + "/" + kInferRESTEndpoint + "/" + model_name;
  if (model_version_ >= 0) {
    url_ += "/" + std::to_string(model_version_);
  }
}

InferHttpContext::~InferHttpContext()
{
  if (curl_ != nullptr) {
    curl_easy_cleanup(curl_);
  }
}

Error
InferHttpContext::Run(std::vector<std::unique_ptr<Result>>* results)
{
  request_timer_.Reset();
  InferResponseHeader infer_response;

  PreRunProcessing();

  if (!curl_global.Status().IsOk()) {
    return curl_global.Status();
  }

  if (curl_ == nullptr) {
    curl_ = curl_easy_init();
    std::string full_url = url_ + "?format=binary";
    curl_easy_setopt(curl_, CURLOPT_URL, full_url.c_str());
    curl_easy_setopt(curl_, CURLOPT_USERAGENT, "libcurl-agent/1.0");
    curl_easy_setopt(curl_, CURLOPT_POST, 1L);
    // Avoid delaying sending on small TCP packets (end of input tensors)
    // Otherwise, ~40 ms delay happens when reusing the connection.
    curl_easy_setopt(curl_, CURLOPT_TCP_NODELAY, 1L);
    if (verbose_) {
      curl_easy_setopt(curl_, CURLOPT_VERBOSE, 1L);
    }

    // request data provided by RequestProvider()
    curl_easy_setopt(curl_, CURLOPT_READFUNCTION, RequestProvider);
    curl_easy_setopt(curl_, CURLOPT_READDATA, this);

    // response headers handled by ResponseHeaderHandler()
    curl_easy_setopt(curl_, CURLOPT_HEADERFUNCTION, ResponseHeaderHandler);
    curl_easy_setopt(curl_, CURLOPT_HEADERDATA, this);

    // response data handled by ResponseHandler()
    curl_easy_setopt(curl_, CURLOPT_WRITEFUNCTION, ResponseHandler);
    curl_easy_setopt(curl_, CURLOPT_WRITEDATA, this);

    // set the expected POST size. If you want to POST large amounts of
    // data, consider CURLOPT_POSTFIELDSIZE_LARGE
    curl_easy_setopt(curl_, CURLOPT_POSTFIELDSIZE, total_input_byte_size_);
  }
  CURL* curl = curl_;
  if (!curl) {
    return
      Error(RequestStatusCode::INTERNAL, "failed to initialize HTTP client");
  }


  // Headers to specify input and output tensors
  infer_request_str_.clear();
  infer_request_str_ =
    std::string(kInferRequestHTTPHeader) + ":" +
    infer_request_.ShortDebugString();
  struct curl_slist *list = NULL;
  list = curl_slist_append(list, "Expect:");
  list = curl_slist_append(list, "Content-Type: application/octet-stream");
  list = curl_slist_append(list, infer_request_str_.c_str());
  curl_easy_setopt(curl, CURLOPT_HTTPHEADER, list);

  // Take run time
  request_timer_.Record(RequestTimers::Kind::REQUEST_START);
  request_timer_.Record(RequestTimers::Kind::SEND_START);
  CURLcode res = curl_easy_perform(curl);
  request_timer_.Record(RequestTimers::Kind::RECEIVE_END);
  request_timer_.Record(RequestTimers::Kind::REQUEST_END);

  if (res != CURLE_OK) {
    curl_slist_free_all(list);
    curl_easy_cleanup(curl);
    requested_results_.clear();
    return
      Error(
        RequestStatusCode::INTERNAL, "HTTP client failed: " +
        std::string(curl_easy_strerror(res)));
  }

  // Must use 64-bit integer with curl_easy_getinfo
  int64_t http_code;
  curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);

  curl_slist_free_all(list);

  // Should have a request status, if not then create an error status.
  if (request_status_.code() == RequestStatusCode::INVALID) {
    request_status_.Clear();
    request_status_.set_code(RequestStatusCode::INTERNAL);
    request_status_.set_msg("infer request did not return status");
  }

  // If request has failing HTTP status or the request's explicit
  // status is not SUCCESS, then signal an error.
  if ((http_code != 200) ||
      (request_status_.code() != RequestStatusCode::SUCCESS)) {
    requested_results_.clear();
    return Error(request_status_);
  }

  // The infer response header should be available...
  if (infer_response_buffer_.empty()) {
    requested_results_.clear();
    return
      Error(
        RequestStatusCode::INTERNAL,
        "infer request did not return result header");
  }

  infer_response.ParseFromString(infer_response_buffer_);

  PostRunProcessing(infer_response);

  results->swap(requested_results_);
  Error err = UpdateStat(request_timer_);
  if (!err.IsOk()) {
    std::cerr << "Failed to update context stat: " << err << std::endl;
  }

  return Error(request_status_);
}

size_t
InferHttpContext::RequestProvider(
  void* contents, size_t size, size_t nmemb, void* userp)
{
  InferHttpContext* ctx = reinterpret_cast<InferHttpContext*>(userp);

  size_t input_bytes = 0;
  Error err =
    ctx->GetNextInput(
      reinterpret_cast<uint8_t*>(contents), size * nmemb, &input_bytes);
  if (!err.IsOk()) {
    std::cerr << "RequestProvider: " << err << std::endl;
    return CURL_READFUNC_ABORT;
  }

  return input_bytes;
}

size_t
InferHttpContext::ResponseHeaderHandler(
  void* contents, size_t size, size_t nmemb, void* userp)
{
  InferHttpContext* ctx = reinterpret_cast<InferHttpContext*>(userp);
  char* buf = reinterpret_cast<char*>(contents);
  size_t byte_size = size * nmemb;

  size_t idx = strlen(kStatusHTTPHeader);
  if ((idx < byte_size) &&
      !strncasecmp(buf, kStatusHTTPHeader, idx)) {
    while ((idx < byte_size) && (buf[idx] != ':')) {
      ++idx;
    }

    if (idx < byte_size) {
      std::string hdr(buf + idx + 1, byte_size - idx - 1);
      if (!google::protobuf::TextFormat::ParseFromString(
          hdr, &ctx->request_status_)) {
        ctx->request_status_.Clear();
      }
    }
  }

  return byte_size;
}

size_t
InferHttpContext::ResponseHandler(
  void* contents, size_t size, size_t nmemb, void* userp)
{
  InferHttpContext* ctx = reinterpret_cast<InferHttpContext*>(userp);
  size_t result_bytes = 0;

  Error err =
    ctx->SetNextRawResult(
      reinterpret_cast<uint8_t*>(contents), size * nmemb, &result_bytes);
  if (!err.IsOk()) {
    std::cerr << "ResponseHandler: " << err << std::endl;
    return 0;
  }

  return result_bytes;
}

//==============================================================================

Error
ProfileHttpContext::Create(
  std::unique_ptr<ProfileContext>* ctx,
  const std::string& server_url, bool verbose)
{
  ctx->reset(
    static_cast<ProfileContext*>(
      new ProfileHttpContext(server_url, verbose)));
  return Error::Success;
}

ProfileHttpContext::ProfileHttpContext(
  const std::string& server_url, bool verbose)
  : ProfileContext(verbose), url_(server_url + "/" + kProfileRESTEndpoint)
{
}

Error
ProfileHttpContext::SendCommand(const std::string& cmd_str)
{
  request_status_.Clear();

  if (!curl_global.Status().IsOk()) {
    return curl_global.Status();
  }

  CURL* curl = curl_easy_init();
  if (!curl) {
    return
      Error(RequestStatusCode::INTERNAL, "failed to initialize HTTP client");
  }

  // Want binary representation of the status.
  std::string full_url = url_ + "?cmd=" + cmd_str;
  curl_easy_setopt(curl, CURLOPT_URL, full_url.c_str());
  curl_easy_setopt(curl, CURLOPT_USERAGENT, "libcurl-agent/1.0");
  if (verbose_) {
    curl_easy_setopt(curl, CURLOPT_VERBOSE, 1L);
  }

  // response headers handled by ResponseHeaderHandler()
  curl_easy_setopt(curl, CURLOPT_HEADERFUNCTION, ResponseHeaderHandler);
  curl_easy_setopt(curl, CURLOPT_HEADERDATA, this);

  CURLcode res = curl_easy_perform(curl);
  if (res != CURLE_OK) {
    curl_easy_cleanup(curl);
    return
      Error(
        RequestStatusCode::INTERNAL, "HTTP client failed: " +
        std::string(curl_easy_strerror(res)));
  }

  // Must use 64-bit integer with curl_easy_getinfo
  int64_t http_code;
  curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);

  curl_easy_cleanup(curl);

  // Should have a request status, if not then create an error status.
  if (request_status_.code() == RequestStatusCode::INVALID) {
    request_status_.Clear();
    request_status_.set_code(RequestStatusCode::INTERNAL);
    request_status_.set_msg("profile request did not return status");
  }

  return Error(request_status_);
}

size_t
ProfileHttpContext::ResponseHeaderHandler(
  void* contents, size_t size, size_t nmemb, void* userp)
{
  ProfileHttpContext* ctx = reinterpret_cast<ProfileHttpContext*>(userp);

  char* buf = reinterpret_cast<char*>(contents);
  size_t byte_size = size * nmemb;

  size_t idx = strlen(kStatusHTTPHeader);
  if ((idx < byte_size) &&
      !strncasecmp(buf, kStatusHTTPHeader, idx)) {
    while ((idx < byte_size) && (buf[idx] != ':')) {
      ++idx;
    }

    if (idx < byte_size) {
      std::string hdr(buf + idx + 1, byte_size - idx - 1);

      if (!google::protobuf::TextFormat::ParseFromString(
          hdr, &ctx->request_status_)) {
        ctx->request_status_.Clear();
      }
    }
  }

  return byte_size;
}

//==============================================================================

Error
ServerHealthGrpcContext::Create(
  std::unique_ptr<ServerHealthContext>* ctx,
  const std::string& server_url, bool verbose)
{
  ctx->reset(
    static_cast<ServerHealthContext*>(
      new ServerHealthGrpcContext(server_url, verbose)));
  return Error::Success;
}

ServerHealthGrpcContext::ServerHealthGrpcContext(
  const std::string& server_url, bool verbose)
  : ServerHealthContext(verbose),
    stub_(GRPCService::NewStub(GetChannel(server_url)))
{
}

Error
ServerHealthGrpcContext::GetHealth(const std::string& mode, bool* health)
{
  Error err;

  HealthRequest request;
  HealthResponse response;
  grpc::ClientContext context;

  request.set_mode(mode);
  grpc::Status grpc_status = stub_->Health(&context, request, &response);
  if (grpc_status.ok()) {
    *health = response.health();
    err = Error(response.request_status());
  } else {
    // Something wrong with the gRPC conncection
    err =
      Error(
        RequestStatusCode::INTERNAL, "gRPC client failed: " +
        std::to_string(grpc_status.error_code()) + ": " +
        grpc_status.error_message());
  }

  if (verbose_ && err.IsOk()) {
    std::cout << mode << ": " << *health << std::endl;
  }

  return err;
}

Error
ServerHealthGrpcContext::GetReady(bool* ready)
{
  return GetHealth("ready", ready);
}

Error
ServerHealthGrpcContext::GetLive(bool* live)
{
  return GetHealth("live", live);
}

//==============================================================================

Error
ServerStatusGrpcContext::Create(
  std::unique_ptr<ServerStatusContext>* ctx,
  const std::string& server_url, bool verbose)
{
  ctx->reset(
    static_cast<ServerStatusContext*>(
      new ServerStatusGrpcContext(server_url, verbose)));
  return Error::Success;
}

Error
ServerStatusGrpcContext::Create(
  std::unique_ptr<ServerStatusContext>* ctx,
  const std::string& server_url, const std::string& model_name, bool verbose)
{
  ctx->reset(
    static_cast<ServerStatusContext*>(
      new ServerStatusGrpcContext(server_url, model_name, verbose)));
  return Error::Success;
}

ServerStatusGrpcContext::ServerStatusGrpcContext(
  const std::string& server_url, bool verbose)
  : ServerStatusContext(verbose), model_name_(""),
    stub_(GRPCService::NewStub(GetChannel(server_url)))
{
}

ServerStatusGrpcContext::ServerStatusGrpcContext(
  const std::string& server_url, const std::string& model_name, bool verbose)
  : ServerStatusContext(verbose), model_name_(model_name),
    stub_(GRPCService::NewStub(GetChannel(server_url)))
{
}

Error
ServerStatusGrpcContext::GetServerStatus(ServerStatus* server_status)
{
  server_status->Clear();

  Error grpc_status;

  StatusRequest request;
  StatusResponse response;
  grpc::ClientContext context;

  request.set_model_name(model_name_);
  grpc::Status status = stub_->Status(&context, request, &response);
  if (status.ok()) {
    server_status->Swap(response.mutable_server_status());
    grpc_status = Error(response.request_status());
  } else {
    // Something wrong with the gRPC conncection
    grpc_status = Error(RequestStatusCode::INTERNAL, "gRPC client failed: " +
      std::to_string(status.error_code()) + ": " + status.error_message());
  }

  // Log server status if request is SUCCESS and verbose is true.
  if (grpc_status.Code() == RequestStatusCode::SUCCESS && verbose_) {
    std::cout << server_status->DebugString() << std::endl;
  }
  return grpc_status;
}

//==============================================================================

Error
InferGrpcContext::Create(
  std::unique_ptr<InferContext>* ctx, const std::string& server_url,
  const std::string& model_name, int model_version, bool verbose)
{
  InferGrpcContext* ctx_ptr =
      new InferGrpcContext(server_url, model_name, model_version, verbose);

  // Get status of the model and create the inputs and outputs.
  std::unique_ptr<ServerStatusContext> sctx;
  Error err =
    ServerStatusGrpcContext::Create(
      &sctx, server_url, model_name, verbose);
  if (err.IsOk()) {
    ServerStatus server_status;
    err = sctx->GetServerStatus(&server_status);
    if (err.IsOk()) {
      const auto& itr = server_status.model_status().find(model_name);
      if (itr == server_status.model_status().end()) {
        err =
          Error(
            RequestStatusCode::INTERNAL,
            "unable to find status information for \"" + model_name + "\"");
      } else {
        const ModelConfig& model_info = itr->second.config();

        ctx_ptr->max_batch_size_ =
          static_cast<uint64_t>(std::max(0, model_info.max_batch_size()));

        // Create inputs and outputs
        for (const auto& io : model_info.input()) {
          ctx_ptr->inputs_.emplace_back(std::make_shared<InputImpl>(io));
        }
        for (const auto& io : model_info.output()) {
          ctx_ptr->outputs_.emplace_back(std::make_shared<OutputImpl>(io));
        }
      }
    }
  }

  if (err.IsOk()) {
    ctx->reset(static_cast<InferContext*>(ctx_ptr));
  } else {
    ctx->reset();
  }

  return err;
}

InferGrpcContext::InferGrpcContext(
  const std::string& server_url, const std::string& model_name,
  int model_version, bool verbose)
  : InferContext(model_name, model_version, verbose),
    stub_(GRPCService::NewStub(GetChannel(server_url)))
{
}

Error
InferGrpcContext::Run(std::vector<std::unique_ptr<Result>>* results)
{
  request_timer_.Reset();
  results->clear();

  Error grpc_status;
  InferResponseHeader infer_response;

  PreRunProcessing();

  InferRequest request;
  InferResponse response;
  grpc::ClientContext context;

  // Use send timer to measure time for marshalling infer request
  request_timer_.Record(RequestTimers::Kind::SEND_START);

  request.set_model_name(model_name_);
  request.set_version(std::to_string(model_version_));
  request.mutable_meta_data()->MergeFrom(infer_request_);

  while (input_pos_idx_ < inputs_.size()) {
    InputImpl* io =
      reinterpret_cast<InputImpl*>(inputs_[input_pos_idx_].get());
    std::string* new_input = request.add_raw_input();
    // Append all batches of one input together
    for (size_t batch_idx = 0; batch_idx < batch_size_; batch_idx++) {
      const uint8_t* data_ptr;
      io->GetRaw(batch_idx, &data_ptr);
      new_input->append(
        reinterpret_cast<const char*>(data_ptr), io->ByteSize());
    }
    input_pos_idx_++;
  }

  request_timer_.Record(RequestTimers::Kind::SEND_END);

  // Take run time
  // (we can only take run time for gRPC because when bytes get sent/received
  //  is encapsulated.)
  request_timer_.Record(RequestTimers::Kind::REQUEST_START);
  grpc::Status status = stub_->Infer(&context, request, &response);
  request_timer_.Record(RequestTimers::Kind::REQUEST_END);

  if (status.ok()) {
    // Use send timer to measure time for unmarshalling infer response
    request_timer_.Record(RequestTimers::Kind::RECEIVE_START);
    infer_response.Swap(response.mutable_meta_data());
    request_status_.Swap(response.mutable_request_status());
    // Reset position index for sanity check, this will be used
    // in InferContext::SetNextRawResult
    result_pos_idx_ = 0;
    for (std::string output : response.raw_output()) {
      size_t size = output.size();
      size_t result_bytes = 0;
      SetNextRawResult(
        reinterpret_cast<uint8_t*>(&output[0]), size, &result_bytes);
      if (result_bytes != size) {
        grpc_status = Error(RequestStatusCode::INVALID,
          "Written bytes doesn't match received bytes.");
      }
    }
    grpc_status = Error(request_status_);
    request_timer_.Record(RequestTimers::Kind::RECEIVE_END);
  } else {
    // Something wrong with the gRPC conncection
    grpc_status = Error(RequestStatusCode::INTERNAL, "gRPC client failed: " +
      std::to_string(status.error_code()) + ": " + status.error_message());
  }

  // Only continue to process result if gRPC status is SUCCESS
  if (grpc_status.Code() != RequestStatusCode::SUCCESS) {
    return grpc_status;
  }

  PostRunProcessing(infer_response);

  results->swap(requested_results_);
  Error err = UpdateStat(request_timer_);
  if (!err.IsOk()) {
    std::cerr << "Failed to update context stat: " << err << std::endl;
  }

  return Error(request_status_);
}

//==============================================================================

Error
ProfileGrpcContext::Create(
  std::unique_ptr<ProfileContext>* ctx,
  const std::string& server_url, bool verbose)
{
  ctx->reset(
    static_cast<ProfileContext*>(
      new ProfileGrpcContext(server_url, verbose)));
  return Error::Success;
}

ProfileGrpcContext::ProfileGrpcContext(
  const std::string& server_url, bool verbose)
  : ProfileContext(verbose),
    stub_(GRPCService::NewStub(GetChannel(server_url)))
{
}

Error
ProfileGrpcContext::SendCommand(const std::string& cmd_str)
{
  ProfileRequest request;
  ProfileResponse response;
  grpc::ClientContext context;

  request.set_cmd(cmd_str);
  grpc::Status status = stub_->Profile(&context, request, &response);
  if (status.ok()) {
    return Error(response.request_status());
  } else {
    // Something wrong with the gRPC conncection
    return Error(RequestStatusCode::INTERNAL, "gRPC client failed: " +
      std::to_string(status.error_code()) + ": " + status.error_message());
  }
}

}}} // namespace nvidia::inferenceserver::client
