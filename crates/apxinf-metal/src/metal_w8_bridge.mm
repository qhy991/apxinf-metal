#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <new>

namespace {

constexpr uint32_t kRowsPerThreadgroup = 8;
constexpr uint32_t kTopKThreads = 256;
constexpr uint32_t kTopK = 4;

struct KernelParams {
    uint32_t columns;
    uint32_t rows;
    uint32_t groups_per_row;
    uint32_t partial_count;
};

struct ApxinfMetalW8Handle {
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    id<MTLComputePipelineState> rows_pipeline;
    id<MTLComputePipelineState> final_pipeline;
    id<MTLBuffer> weights;
    id<MTLBuffer> scales;
    id<MTLBuffer> hidden;
    id<MTLBuffer> partial;
    id<MTLBuffer> tokens;
    KernelParams params;
};

#ifdef APXINF_METAL_EXPERIMENTS
struct ApxinfMetalW8MatVecHandle {
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    id<MTLComputePipelineState> pipeline;
    id<MTLBuffer> weights;
    id<MTLBuffer> scales;
    id<MTLBuffer> input;
    id<MTLBuffer> output;
    KernelParams params;
};
#endif

#include "metal_w8_source.inc"
#ifdef APXINF_METAL_EXPERIMENTS
#include "metal_w8_matvec_source.inc"
#endif

void write_error(char *output, size_t capacity, const char *message) {
    if (output == nullptr || capacity == 0) {
        return;
    }
    std::snprintf(output, capacity, "%s", message == nullptr ? "unknown Metal error" : message);
}

void write_nserror(char *output, size_t capacity, NSError *error) {
    write_error(output, capacity,
                error == nil ? "unknown Metal error" : error.localizedDescription.UTF8String);
}

}  // namespace

extern "C" int apxinf_metal_w8_create(
    const int8_t *weights, const float *scales, uint32_t rows, uint32_t columns,
    uint32_t group_size, void **output, char *error_output, size_t error_capacity) {
    @autoreleasepool {
        if (output == nullptr) {
            write_error(error_output, error_capacity, "output handle is null");
            return 1;
        }
        *output = nullptr;
        if (weights == nullptr || scales == nullptr || rows == 0 || columns == 0 ||
            group_size != 64 || columns % group_size != 0 || columns % 4 != 0) {
            write_error(error_output, error_capacity, "invalid Metal W8 packed-weight contract");
            return 1;
        }

        const size_t weight_bytes = static_cast<size_t>(rows) * columns;
        const size_t scale_count = static_cast<size_t>(rows) * (columns / group_size);
        if (weight_bytes / columns != rows || scale_count / (columns / group_size) != rows) {
            write_error(error_output, error_capacity, "Metal W8 buffer dimensions overflow");
            return 1;
        }

        auto handle = new (std::nothrow) ApxinfMetalW8Handle{};
        if (handle == nullptr) {
            write_error(error_output, error_capacity, "allocate Metal W8 handle failed");
            return 1;
        }
        handle->device = MTLCreateSystemDefaultDevice();
        if (handle->device == nil) {
            delete handle;
            write_error(error_output, error_capacity, "no system Metal device is available");
            return 1;
        }

        NSError *error = nil;
        NSString *source = [NSString stringWithUTF8String:kMetalSource];
        id<MTLLibrary> library = [handle->device newLibraryWithSource:source options:nil error:&error];
        if (library == nil) {
            delete handle;
            write_nserror(error_output, error_capacity, error);
            return 1;
        }
        id<MTLFunction> rows_function = [library newFunctionWithName:@"w8_rows_topk4"];
        id<MTLFunction> final_function = [library newFunctionWithName:@"w8_final_topk4"];
        handle->rows_pipeline =
            [handle->device newComputePipelineStateWithFunction:rows_function error:&error];
        if (handle->rows_pipeline == nil) {
            delete handle;
            write_nserror(error_output, error_capacity, error);
            return 1;
        }
        handle->final_pipeline =
            [handle->device newComputePipelineStateWithFunction:final_function error:&error];
        if (handle->final_pipeline == nil) {
            delete handle;
            write_nserror(error_output, error_capacity, error);
            return 1;
        }
        handle->queue = [handle->device newCommandQueue];
        if (handle->queue == nil) {
            delete handle;
            write_error(error_output, error_capacity, "create Metal command queue failed");
            return 1;
        }

        const MTLResourceOptions shared = MTLResourceStorageModeShared;
        handle->weights = [handle->device newBufferWithBytes:weights
                                                     length:weight_bytes
                                                    options:shared];
        handle->scales = [handle->device newBufferWithBytes:scales
                                                    length:scale_count * sizeof(float)
                                                   options:shared];
        handle->hidden = [handle->device newBufferWithLength:static_cast<size_t>(columns) * sizeof(float)
                                                    options:shared];
        const uint32_t partial_count = (rows + kRowsPerThreadgroup - 1) / kRowsPerThreadgroup;
        handle->partial = [handle->device newBufferWithLength:static_cast<size_t>(partial_count) * kTopK * 8
                                                     options:MTLResourceStorageModePrivate];
        handle->tokens = [handle->device newBufferWithLength:kTopK * sizeof(uint32_t) options:shared];
        if (handle->weights == nil || handle->scales == nil || handle->hidden == nil ||
            handle->partial == nil || handle->tokens == nil) {
            delete handle;
            write_error(error_output, error_capacity, "allocate persistent Metal W8 buffers failed");
            return 1;
        }
        handle->params = KernelParams{columns, rows, columns / group_size, partial_count};
        *output = handle;
        return 0;
    }
}

extern "C" int apxinf_metal_w8_topk4(
    void *opaque_handle, const float *hidden, uint32_t hidden_count,
    uint32_t *output_tokens, char *error_output, size_t error_capacity) {
    @autoreleasepool {
        auto handle = static_cast<ApxinfMetalW8Handle *>(opaque_handle);
        if (handle == nullptr || hidden == nullptr || output_tokens == nullptr ||
            hidden_count != handle->params.columns) {
            write_error(error_output, error_capacity, "invalid Metal W8 decode input");
            return 1;
        }
        std::memcpy(handle->hidden.contents, hidden,
                    static_cast<size_t>(hidden_count) * sizeof(float));

        id<MTLCommandBuffer> command = [handle->queue commandBuffer];
        if (command == nil) {
            write_error(error_output, error_capacity, "create Metal command buffer failed");
            return 1;
        }
        id<MTLComputeCommandEncoder> rows = [command computeCommandEncoder];
        if (rows == nil) {
            write_error(error_output, error_capacity, "create first Metal encoder failed");
            return 1;
        }
        [rows setComputePipelineState:handle->rows_pipeline];
        [rows setBuffer:handle->weights offset:0 atIndex:0];
        [rows setBuffer:handle->scales offset:0 atIndex:1];
        [rows setBuffer:handle->hidden offset:0 atIndex:2];
        [rows setBuffer:handle->partial offset:0 atIndex:3];
        [rows setBytes:&handle->params length:sizeof(handle->params) atIndex:4];
        [rows dispatchThreadgroups:MTLSizeMake(handle->params.partial_count, 1, 1)
                 threadsPerThreadgroup:MTLSizeMake(kRowsPerThreadgroup * 32, 1, 1)];
        [rows endEncoding];

        id<MTLComputeCommandEncoder> final = [command computeCommandEncoder];
        if (final == nil) {
            write_error(error_output, error_capacity, "create final Metal encoder failed");
            return 1;
        }
        [final setComputePipelineState:handle->final_pipeline];
        [final setBuffer:handle->partial offset:0 atIndex:0];
        [final setBuffer:handle->tokens offset:0 atIndex:1];
        [final setBytes:&handle->params length:sizeof(handle->params) atIndex:2];
        [final dispatchThreadgroups:MTLSizeMake(1, 1, 1)
                  threadsPerThreadgroup:MTLSizeMake(kTopKThreads, 1, 1)];
        [final endEncoding];

        [command commit];
        [command waitUntilCompleted];
        if (command.status != MTLCommandBufferStatusCompleted) {
            write_nserror(error_output, error_capacity, command.error);
            return 1;
        }
        std::memcpy(output_tokens, handle->tokens.contents, kTopK * sizeof(uint32_t));
        return 0;
    }
}

extern "C" void apxinf_metal_w8_destroy(void *opaque_handle) {
    auto handle = static_cast<ApxinfMetalW8Handle *>(opaque_handle);
    delete handle;
}

#ifdef APXINF_METAL_EXPERIMENTS
extern "C" int apxinf_metal_w8_matvec_create(
    const int8_t *weights, const float *scales, uint32_t rows, uint32_t columns,
    uint32_t group_size, void **output, char *error_output, size_t error_capacity) {
    @autoreleasepool {
        if (output == nullptr) {
            write_error(error_output, error_capacity, "output handle is null");
            return 1;
        }
        *output = nullptr;
        if (weights == nullptr || scales == nullptr || rows == 0 || columns == 0 ||
            group_size != 64 || columns % group_size != 0 || columns % 4 != 0) {
            write_error(error_output, error_capacity, "invalid Metal W8 matvec packed-weight contract");
            return 1;
        }
        const size_t weight_bytes = static_cast<size_t>(rows) * columns;
        const size_t groups_per_row = columns / group_size;
        const size_t scale_count = static_cast<size_t>(rows) * groups_per_row;
        if (weight_bytes / columns != rows || scale_count / groups_per_row != rows ||
            static_cast<size_t>(rows) > std::numeric_limits<size_t>::max() / sizeof(float)) {
            write_error(error_output, error_capacity, "Metal W8 matvec buffer dimensions overflow");
            return 1;
        }

        auto handle = new (std::nothrow) ApxinfMetalW8MatVecHandle{};
        if (handle == nullptr) {
            write_error(error_output, error_capacity, "allocate Metal W8 matvec handle failed");
            return 1;
        }
        handle->device = MTLCreateSystemDefaultDevice();
        if (handle->device == nil) {
            delete handle;
            write_error(error_output, error_capacity, "no system Metal device is available");
            return 1;
        }
        NSError *error = nil;
        NSString *source = [NSString stringWithUTF8String:kMetalMatVecSource];
        id<MTLLibrary> library = [handle->device newLibraryWithSource:source options:nil error:&error];
        if (library == nil) {
            delete handle;
            write_nserror(error_output, error_capacity, error);
            return 1;
        }
        id<MTLFunction> function = [library newFunctionWithName:@"w8_rows_matvec"];
        handle->pipeline = [handle->device newComputePipelineStateWithFunction:function error:&error];
        if (handle->pipeline == nil) {
            delete handle;
            write_nserror(error_output, error_capacity, error);
            return 1;
        }
        handle->queue = [handle->device newCommandQueue];
        const MTLResourceOptions shared = MTLResourceStorageModeShared;
        handle->weights = [handle->device newBufferWithBytes:weights length:weight_bytes options:shared];
        handle->scales = [handle->device newBufferWithBytes:scales
                                                    length:scale_count * sizeof(float)
                                                   options:shared];
        handle->input = [handle->device newBufferWithLength:static_cast<size_t>(columns) * sizeof(float)
                                                    options:shared];
        handle->output = [handle->device newBufferWithLength:static_cast<size_t>(rows) * sizeof(float)
                                                     options:shared];
        if (handle->queue == nil || handle->weights == nil || handle->scales == nil ||
            handle->input == nil || handle->output == nil) {
            delete handle;
            write_error(error_output, error_capacity, "allocate persistent Metal W8 matvec buffers failed");
            return 1;
        }
        const uint32_t partial_count = (rows + kRowsPerThreadgroup - 1) / kRowsPerThreadgroup;
        handle->params = KernelParams{columns, rows, static_cast<uint32_t>(groups_per_row), partial_count};
        *output = handle;
        return 0;
    }
}

extern "C" int apxinf_metal_w8_matvec_multiply(
    void *opaque_handle, const float *input, uint32_t input_count,
    float *output, uint32_t output_count, char *error_output, size_t error_capacity) {
    @autoreleasepool {
        auto handle = static_cast<ApxinfMetalW8MatVecHandle *>(opaque_handle);
        if (handle == nullptr || input == nullptr || output == nullptr ||
            input_count != handle->params.columns || output_count != handle->params.rows) {
            write_error(error_output, error_capacity, "invalid Metal W8 matvec input or output");
            return 1;
        }
        std::memcpy(handle->input.contents, input,
                    static_cast<size_t>(input_count) * sizeof(float));
        id<MTLCommandBuffer> command = [handle->queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        if (command == nil || encoder == nil) {
            write_error(error_output, error_capacity, "create Metal W8 matvec command failed");
            return 1;
        }
        [encoder setComputePipelineState:handle->pipeline];
        [encoder setBuffer:handle->weights offset:0 atIndex:0];
        [encoder setBuffer:handle->scales offset:0 atIndex:1];
        [encoder setBuffer:handle->input offset:0 atIndex:2];
        [encoder setBuffer:handle->output offset:0 atIndex:3];
        [encoder setBytes:&handle->params length:sizeof(handle->params) atIndex:4];
        [encoder dispatchThreadgroups:MTLSizeMake(handle->params.partial_count, 1, 1)
                   threadsPerThreadgroup:MTLSizeMake(kRowsPerThreadgroup * 32, 1, 1)];
        [encoder endEncoding];
        [command commit];
        [command waitUntilCompleted];
        if (command.status != MTLCommandBufferStatusCompleted) {
            write_nserror(error_output, error_capacity, command.error);
            return 1;
        }
        std::memcpy(output, handle->output.contents,
                    static_cast<size_t>(output_count) * sizeof(float));
        return 0;
    }
}

extern "C" void apxinf_metal_w8_matvec_destroy(void *opaque_handle) {
    auto handle = static_cast<ApxinfMetalW8MatVecHandle *>(opaque_handle);
    delete handle;
}
#endif
