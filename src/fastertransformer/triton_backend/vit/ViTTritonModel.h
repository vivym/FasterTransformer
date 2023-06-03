#pragma once

#include "src/fastertransformer/models/vit/ViT.h"
#include "src/fastertransformer/triton_backend/transformer_triton_backend.hpp"

namespace ft = fastertransformer;

template<typename T>
struct ViTTritonModel: public AbstractTransformerModel {
    ViTTritonModel(size_t      tensor_para_size,
                   size_t      pipeline_para_size,
                   bool        enable_custom_all_reduce,
                   std::string model_dir,
                   int         int8_mode,
                   bool        is_sparse);

    virtual std::unique_ptr<AbstractTransformerModelInstance>
    createModelInstance(int                                                               deviceId,
                        int                                                               rank,
                        cudaStream_t                                                      stream,
                        std::pair<std::vector<ft::NcclParam>, std::vector<ft::NcclParam>> nccl_params,
                        std::shared_ptr<ft::AbstractCustomComm> custom_all_reduce_comm = nullptr) override;

    virtual void createSharedWeights(int deviceId, int rank) override;

    virtual void createCustomComms(std::vector<std::shared_ptr<ft::AbstractCustomComm>>* custom_all_reduce_comms,
                                   int                                                   world_size) override;

    virtual std::string toString() override;
    virtual int         getTensorParaSize() override;
    virtual int         getPipelineParaSize() override;

private:
    size_t img_size_;
    size_t in_chans_;
    size_t patch_size_;
    size_t embed_dim_;
    size_t head_num_;
    size_t inter_size_;
    size_t num_layer_;
    bool with_cls_token_;
    float q_scaling_;


    size_t              tensor_para_size_;
    size_t              pipeline_para_size_;
    bool                is_sparse_;

    std::string                                     model_name_;
    std::string                                     model_dir_;
    int                                             int8_mode_                = 0;
    bool                                            enable_custom_all_reduce_ = 0;
    std::vector<std::shared_ptr<ft::ViTWeight<T>>>  shared_weights_;
};
