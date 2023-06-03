#include "3rdparty/INIReader.h"

#include "src/fastertransformer/triton_backend/vit/ViTTritonModel.h"
#include "src/fastertransformer/triton_backend/vit/ViTTritonModelInstance.h"

namespace ft = fastertransformer;

// std::shared_ptr<AbstractTransformerModel> AbstractTransformerModel::createViTModel(std::string model_dir)
// {
//     INIReader reader = INIReader(model_dir + "/config.ini");
//     if (reader.ParseError() < 0) {
//         std::cout << "[ERROR] Can't load '" << model_dir << "/config.ini"
//                   << "'\n";
//         return nullptr;
//     }

//     const std::string data_type = reader.Get("ft_instance_hyperparameter", "data_type");
//     if (data_type == "fp16") {
//         return std::make_shared<ViTTritonModel<half>>(reader, model_dir);
//     }
// #ifdef ENABLE_BF16
//     else if (data_type == "bf16") {
//         return std::make_shared<ViTTritonModel<__nv_bfloat16>>(reader, model_dir);
//     }
// #endif
//     else if (data_type == "fp32") {
//         return std::make_shared<ViTTritonModel<float>>(reader, model_dir);
//     }
//     else {
//         FT_LOG_ERROR("Unsupported data type " + data_type);
//         exit(-1);
//     }
// }

template<typename T>
ViTTritonModel<T>::ViTTritonModel(size_t        tensor_para_size,
                                 size_t         pipeline_para_size,
                                 bool           enable_custom_all_reduce,
                                 std::string    model_dir,
                                 int            int8_mode,
                                 bool           is_sparse):
    tensor_para_size_(tensor_para_size),
    pipeline_para_size_(pipeline_para_size),
    shared_weights_(std::vector<std::shared_ptr<ft::ViTWeight<T>>>(ft::getDeviceCount())),
    enable_custom_all_reduce_(enable_custom_all_reduce),
    model_dir_(model_dir),
    int8_mode_(int8_mode),
    is_sparse_(is_sparse)
{
    FT_CHECK_WITH_INFO(int8_mode_ == 0, "still not support int8 in vit backend");
    FT_CHECK_WITH_INFO(is_sparse == false, "still not support sparse in vit backend");

    INIReader reader = INIReader(model_dir + "/config.ini");
    if (reader.ParseError() < 0) {
        std::cout << "[ERROR] Can't load '" << model_dir << "/config.ini"
                  << "'\n";
        ft::FT_CHECK(false);
    }

    /* ViT base Configuration File Example
    [vit]
    model_name = vit
    img_size = 224
    in_chans = 3
    patch_size = 16
    embed_dim = 768
    head_num = 12
    inter_size = 3072
    num_layer = 12
    with_cls_token = True
    */

    model_name_ = reader.Get("vit", "model_name", "vit");
    // TODO: infer max_batch_size from triton model config
    // max_batch_size_ = reader.GetInteger("vit", "max_batch_size", 1);
    img_size_   = reader.GetInteger("vit", "img_size", 224);
    in_chans_   = reader.GetInteger("vit", "in_chans", 3);
    patch_size_ = reader.GetInteger("vit", "patch_size", 16);
    embed_dim_  = reader.GetInteger("vit", "embed_dim", 768);
    head_num_   = reader.GetInteger("vit", "head_num", 12);
    inter_size_    = reader.GetInteger("vit", "intermediate_size", 3072);
    num_layer_     = reader.GetInteger("vit", "num_layer", 12);
    with_cls_token_ = reader.GetBoolean("vit", "with_cls_token", true);
    q_scaling_     = reader.GetReal("vit", "q_scaling", 1.0f);
}

template<typename T>
std::unique_ptr<AbstractTransformerModelInstance>
ViTTritonModel<T>::createModelInstance(int                                                               device_id,
                                       int                                                               rank,
                                       cudaStream_t                                                      stream,
                                       std::pair<std::vector<ft::NcclParam>, std::vector<ft::NcclParam>> nccl_params,
                                       std::shared_ptr<ft::AbstractCustomComm> custom_all_reduce_comm)
{
    ft::check_cuda_error(cudaSetDevice(device_id));
    const int comms_rank         = device_id % (tensor_para_size_ * pipeline_para_size_);
    const int tensor_para_rank   = rank % tensor_para_size_;
    const int pipeline_para_rank = rank / tensor_para_size_;

    std::unique_ptr<ft::Allocator<ft::AllocatorType::CUDA>> allocator(
        new ft::Allocator<ft::AllocatorType::CUDA>(device_id));

    allocator->setStream(stream);

    cudnnHandle_t    cudnn_handle;
    cublasHandle_t   cublas_handle;
    cublasLtHandle_t cublaslt_handle;

    checkCUDNN(cudnnCreate(&cudnn_handle));
    checkCUDNN(cudnnSetStream(cudnn_handle, stream));
    cublasCreate(&cublas_handle);
    cublasLtCreate(&cublaslt_handle);
    cublasSetStream(cublas_handle, stream);

    std::unique_ptr<ft::cublasAlgoMap>   cublas_algo_map(new ft::cublasAlgoMap("gemm_config.in"));
    std::unique_ptr<std::mutex>          cublas_wrapper_mutex(new std::mutex());
    std::unique_ptr<ft::cublasMMWrapper> cublas_wrapper(new ft::cublasMMWrapper(
        cublas_handle, cublaslt_handle, stream, cublas_algo_map.get(), cublas_wrapper_mutex.get(), allocator.get()));

    std::unique_ptr<cudaDeviceProp> cuda_device_prop_ptr(new cudaDeviceProp);
    ft::check_cuda_error(cudaGetDeviceProperties(cuda_device_prop_ptr.get(), device_id));

    if (std::is_same<T, half>::value) {
        cublas_wrapper->setGemmConfig(CUDA_R_16F, CUDA_R_16F, CUDA_R_16F, CUDA_R_32F);
    }
#ifdef ENABLE_BF16
    else if (std::is_same<T, __nv_bfloat16>::value) {
        cublas_wrapper->setBF16GemmConfig();
    }
#endif
    else if (std::is_same<T, float>::value) {
        cublas_wrapper->setFP32GemmConfig();
    }

    ft::NcclParam tensor_para   = nccl_params.first[comms_rank];
    ft::NcclParam pipeline_para = nccl_params.second[comms_rank];

    const int seq_len = (img_size_ / patch_size_) * (img_size_ / patch_size_) + (with_cls_token_ ? 1 : 0);
    const int head_dim = embed_dim_ / head_num_;
    const int sm = ft::getSMVersion();
    ft::AttentionType attention_type = ft::getAttentionType<T>(head_dim, sm, true, seq_len);

    auto vit = std::make_unique<ft::ViTTransformer<T>>(0,   // max_batch_size, FT will adjust the buffer automatically.
                                                       img_size_,
                                                       in_chans_,
                                                       patch_size_,
                                                       embed_dim_,
                                                       head_num_,
                                                       inter_size_,
                                                       num_layer_,
                                                       with_cls_token_,
                                                       sm,
                                                       q_scaling_,
                                                       stream,
                                                       cudnn_handle,
                                                       cublas_wrapper.get(),
                                                       allocator.get(),
                                                       false,
                                                       attention_type);

    // TODO: is_sparse

    return std::unique_ptr<ViTTritonModelInstance<T>>(new ViTTritonModelInstance<T>(std::move(vit),
                                                                                    shared_weights_[device_id],
                                                                                    std::move(allocator),
                                                                                    std::move(cublas_algo_map),
                                                                                    std::move(cublas_wrapper_mutex),
                                                                                    std::move(cublas_wrapper),
                                                                                    std::move(cuda_device_prop_ptr),
                                                                                    seq_len,
                                                                                    embed_dim_));
}

template<typename T>
void ViTTritonModel<T>::createSharedWeights(int device_id, int rank) {
    ft::check_cuda_error(cudaSetDevice(device_id));
    shared_weights_[device_id]   = std::make_shared<ft::ViTWeight<T>>(embed_dim_,
                                                                      inter_size_,
                                                                      num_layer_,
                                                                      img_size_,
                                                                      patch_size_,
                                                                      in_chans_,
                                                                      with_cls_token_);

    shared_weights_[device_id]->ImportWeights(model_dir_);
    return;
}

template<typename T>
std::string ViTTritonModel<T>::toString()
{
    std::stringstream ss;
    ss << "Model: " << model_name_ << "\nmodel_dir: " << model_dir_
       << "\nimg_size: " << img_size_ << "\nin_chans: " << in_chans_
       << "\npatch_size: " << patch_size_ << "\nembed_dim: " << embed_dim_
       << "\nhead_num: " << head_num_ << "\ninter_size: " << inter_size_
       << "\nnum_layer: " << num_layer_ << "\nwith_cls_token: " << with_cls_token_
       << "\nq_scaling: " << q_scaling_
       << "\ntensor_para_size: " << tensor_para_size_
       << "\npipeline_para_size: " << pipeline_para_size_
       << "\nint8_mode:" << int8_mode_
       << "\nenable_custom_all_reduce:" << enable_custom_all_reduce_
       << "\nis_sparse: " << is_sparse_ << std::endl;

    return ss.str();
}

template<typename T>
void ViTTritonModel<T>::createCustomComms(
    std::vector<std::shared_ptr<ft::AbstractCustomComm>>* custom_all_reduce_comms, int world_size)
{
    using commDataType = typename ft::CustomARCommTypeConverter<T>::Type;
    ft::initCustomAllReduceComm<commDataType>(custom_all_reduce_comms, enable_custom_all_reduce_, world_size);
}

template<typename T>
int ViTTritonModel<T>::getTensorParaSize()
{
    return tensor_para_size_;
}

template<typename T>
int ViTTritonModel<T>::getPipelineParaSize()
{
    return pipeline_para_size_;
}

template struct ViTTritonModel<float>;
template struct ViTTritonModel<half>;
#ifdef ENABLE_BF16
// template struct ViTTritonModel<__nv_bfloat16>;
#endif
