#pragma once

#include "src/fastertransformer/models/vit/ViT.h"
#include "src/fastertransformer/triton_backend/transformer_triton_backend.hpp"

namespace ft = fastertransformer;

template<typename T>
struct ViTTritonModelInstance: public AbstractTransformerModelInstance {
    ViTTritonModelInstance(std::unique_ptr<ft::ViTTransformer<T>>                  vit,
                           std::shared_ptr<ft::ViTWeight<T>>                       vit_weight,
                           std::unique_ptr<ft::Allocator<ft::AllocatorType::CUDA>> allocator,
                           std::unique_ptr<ft::cublasAlgoMap>                      cublas_algo_map,
                           std::unique_ptr<std::mutex>                             cublas_wrapper_mutex,
                           std::unique_ptr<ft::cublasMMWrapper>                    cublas_wrapper,
                           std::unique_ptr<cudaDeviceProp>                         cuda_device_prop_ptr,
                           size_t                                                  seq_len,
                           size_t                                                  embed_dim);
    ~ViTTritonModelInstance();

    std::shared_ptr<std::vector<triton::Tensor>>
    forward(std::shared_ptr<std::vector<triton::Tensor>> input_tensors) override;

    std::shared_ptr<std::unordered_map<std::string, triton::Tensor>>
    forward(std::shared_ptr<std::unordered_map<std::string, triton::Tensor>> input_tensors) override;

    static std::shared_ptr<std::unordered_map<std::string, triton::Tensor>>
    convert_outputs(ft::TensorMap& output_tensors);

private:
    const std::unique_ptr<ft::ViTTransformer<T>>                  vit_;
    const std::shared_ptr<ft::ViTWeight<T>>                       vit_weight_;
    const std::unique_ptr<ft::Allocator<ft::AllocatorType::CUDA>> allocator_;
    const std::unique_ptr<ft::cublasAlgoMap>                      cublas_algo_map_;
    const std::unique_ptr<std::mutex>                             cublas_wrapper_mutex_;
    const std::unique_ptr<ft::cublasMMWrapper>                    cublas_wrapper_;
    const std::unique_ptr<cudaDeviceProp>                         cuda_device_prop_ptr_;

    ft::TensorMap convert_inputs(std::shared_ptr<std::unordered_map<std::string, triton::Tensor>> input_tensors);

    void allocateBuffer(const size_t batch_size, const size_t seq_len, const size_t hidden_units);
    void freeBuffer();

    size_t seq_len_;
    size_t embed_dim_;

    T*   d_input_image_         = nullptr;
    T*   d_output_hidden_state_ = nullptr;

    std::exception_ptr h_exception_ = nullptr;
};
