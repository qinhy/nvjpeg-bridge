// Minimal subset of the DLPack ABI (https://github.com/dmlc/dlpack), legacy
// (unversioned) DLManagedTensor form, which every consumer that supports
// DLPack understands (PyTorch >= 1.10, CuPy, JAX, TensorFlow, ...).
//
// Only the definitions needed to *produce* a CUDA uint8 tensor are declared.
// The layout below matches dlpack.h v0.6 - v1.x for these structs.
#pragma once

#include <cstdint>

extern "C" {

typedef enum {
    kDLCPU = 1,
    kDLCUDA = 2,
} DLDeviceType;

typedef struct {
    int32_t device_type;  // DLDeviceType
    int32_t device_id;
} DLDevice;

typedef enum {
    kDLInt = 0,
    kDLUInt = 1,
    kDLFloat = 2,
} DLDataTypeCode;

typedef struct {
    uint8_t code;
    uint8_t bits;
    uint16_t lanes;
} DLDataType;

typedef struct {
    void* data;
    DLDevice device;
    int32_t ndim;
    DLDataType dtype;
    int64_t* shape;
    int64_t* strides;  // in elements, not bytes
    uint64_t byte_offset;
} DLTensor;

typedef struct DLManagedTensor {
    DLTensor dl_tensor;
    void* manager_ctx;
    void (*deleter)(struct DLManagedTensor* self);
} DLManagedTensor;

}  // extern "C"
