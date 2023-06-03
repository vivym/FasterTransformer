cmake_minimum_required(VERSION 3.8)

set(vit_triton_backend_files
    ViTTritonModel.cc
    ViTTritonModelInstance.cc
)

add_library(ViTTritonBackend STATIC ${vit_triton_backend_files})
set_property(TARGET ViTTritonBackend PROPERTY POSITION_INDEPENDENT_CODE  ON)
target_link_libraries(ViTTritonBackend PRIVATE ViT TransformerTritonBackend -lcublasLt)
target_compile_features(ViTTritonBackend PRIVATE cxx_std_14)
