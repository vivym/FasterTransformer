#include "src/fastertransformer/triton_backend/vit/ViTTritonModelInstance.h"
#include "src/fastertransformer/triton_backend/triton_utils.hpp"

namespace ft = fastertransformer;

template<typename T>
void triton_stream_callback(std::unordered_map<std::string, ft::Tensor>* output_tensors, void* ctx) {
    ViTTritonModelInstance<T>* model  = reinterpret_cast<ViTTritonModelInstance<T>*>(ctx);
    auto                       result = ViTTritonModelInstance<T>::convert_outputs(*output_tensors);

    model->stream_cb_(result, model->stream_ctx_);
}

template<typename T>
ViTTritonModelInstance<T>::ViTTritonModelInstance(std::unique_ptr<ft::ViTTransformer<T>>                  vit,
                                                  std::shared_ptr<ft::ViTWeight<T>>                       vit_weight,
                                                  std::unique_ptr<ft::Allocator<ft::AllocatorType::CUDA>> allocator,
                                                  std::unique_ptr<ft::cublasAlgoMap>                      cublas_algo_map,
                                                  std::unique_ptr<std::mutex>                             cublas_wrapper_mutex,
                                                  std::unique_ptr<ft::cublasMMWrapper>                    cublas_wrapper,
                                                  std::unique_ptr<cudaDeviceProp>                         cuda_device_prop_ptr,
                                                  size_t                                                  seq_len,
                                                  size_t                                                  embed_dim)
    : vit_(std::move(vit)),
      vit_weight_(vit_weight),
      allocator_(std::move(allocator)),
      cublas_algo_map_(std::move(cublas_algo_map)),
      cublas_wrapper_mutex_(std::move(cublas_wrapper_mutex)),
      cublas_wrapper_(std::move(cublas_wrapper)),
      cuda_device_prop_ptr_(std::move(cuda_device_prop_ptr)),
      seq_len_(seq_len),
      embed_dim_(embed_dim)
{
}

template<typename T>
ViTTritonModelInstance<T>::~ViTTritonModelInstance()
{
    freeBuffer();
}

template<typename T>
std::shared_ptr<std::vector<triton::Tensor>>
ViTTritonModelInstance<T>::forward(std::shared_ptr<std::vector<triton::Tensor>> input_tensors)
{
    ft::FT_CHECK(false);
    return nullptr;
}

template<typename T>
ft::TensorMap ViTTritonModelInstance<T>::convert_inputs(
    std::shared_ptr<std::unordered_map<std::string, triton::Tensor>> input_tensors)
{
    move_tensor_H2D(input_tensors->at("input_image"), d_input_image_, &allocator_);

    ft::TensorMap ft_input_tensors(
        {{"input_image", as_GPU_tensor(input_tensors->at("input_image"), d_input_image_)}});

    return ft_input_tensors;
}

template<typename T>
std::shared_ptr<std::unordered_map<std::string, triton::Tensor>>
ViTTritonModelInstance<T>::convert_outputs(ft::TensorMap& output_tensors)
{
    std::unordered_map<std::string, triton::Tensor>* outputs_mapping =
        new std::unordered_map<std::string, triton::Tensor>();

    for (auto it = output_tensors.begin(); it != output_tensors.end(); it++) {
        outputs_mapping->insert({it->first, triton::Tensor::convertFtTensorToTriton(it->second)});
    }

    return std::shared_ptr<std::unordered_map<std::string, triton::Tensor>>(outputs_mapping);
}

template<typename T>
std::shared_ptr<std::unordered_map<std::string, triton::Tensor>>
ViTTritonModelInstance<T>::forward(std::shared_ptr<std::unordered_map<std::string, triton::Tensor>> input_tensors)
{
    const size_t batch_size     = input_tensors->at("input_image").shape[0];
    const size_t img_size       = input_tensors->at("input_image").shape[1];
    const size_t img_size2      = input_tensors->at("input_image").shape[2];
    const size_t in_chans       = input_tensors->at("input_image").shape[3];

    if (img_size != img_size2) {
        ft::TensorMap output_tensors = ft::TensorMap(
            {{"error_message", ft::Tensor{ft::MEMORY_CPU, ft::TYPE_BYTES, {1}, "Image must be square"}}});
        return convert_outputs(output_tensors);
    }

    ft::TensorMap ft_input_tensors = convert_inputs(input_tensors);

    std::vector<ft::Tensor> input_tensor_list = {ft_input_tensors.at("input_image")};
    std::vector<ft::Tensor> output_tensor_list = {
        ft::Tensor{ft::MEMORY_GPU, ft::getTensorType<T>(),
        std::vector<size_t>{batch_size, seq_len_, embed_dim_},
        d_output_hidden_state_}};
    ft::TensorMap output_tensors = ft::TensorMap();

    allocateBuffer(batch_size, seq_len_, embed_dim_);

    try {
        vit_->forward(&output_tensor_list, &input_tensor_list, vit_weight_.get());
        cudaStreamSynchronize(vit_->getStream());
        output_tensors.insert({"output_hidden_state", output_tensor_list[0]});
    }
    catch (...) {
        h_exception_ = std::current_exception();
        output_tensors.insert({"error_message", ft::Tensor{ft::MEMORY_CPU, ft::TYPE_BYTES, {1}, &h_exception_}});
    }

    return convert_outputs(output_tensors);
}

template<typename T>
void ViTTritonModelInstance<T>::allocateBuffer(const size_t batch_size,
                                               const size_t seq_len,
                                               const size_t hidden_units)
{
    d_output_hidden_state_ =
        (T*)(allocator_->reMalloc(d_output_hidden_state_, sizeof(T) * batch_size * seq_len * hidden_units, false));
}

template<typename T>
void ViTTritonModelInstance<T>::freeBuffer()
{
    if (d_output_hidden_state_ != nullptr) {
        allocator_->free((void**)(&d_output_hidden_state_));
    }
    if (d_input_image_ != nullptr) {
        allocator_->free((void**)(&d_input_image_));
    }
}

template struct ViTTritonModelInstance<float>;
template struct ViTTritonModelInstance<half>;
#ifdef ENABLE_BF16
template struct ViTTritonModelInstance<__nv_bfloat16>;
#endif
