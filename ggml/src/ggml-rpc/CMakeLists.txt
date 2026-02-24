message(STATUS "Using RPC backend")

ggml_add_backend_library(ggml-rpc
                         ggml-rpc.cpp
                        )

if (WIN32)
    target_link_libraries(ggml-rpc PRIVATE ws2_32)
endif()

# RDMA transport support
option(GGML_RDMA "Enable RDMA transport for RPC" OFF)
if (GGML_RDMA)
    find_library(RDMACM_LIB rdmacm)
    find_library(IBVERBS_LIB ibverbs)
    if (RDMACM_LIB AND IBVERBS_LIB)
        message(STATUS "RDMA transport enabled")
        target_compile_definitions(ggml-rpc PRIVATE GGML_USE_RDMA)
        target_link_libraries(ggml-rpc PRIVATE ${RDMACM_LIB} ${IBVERBS_LIB})
    else()
        message(WARNING "RDMA libraries not found, disabling RDMA transport")
    endif()
endif()
