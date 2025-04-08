#undef __HIP_NO_HALF_CONVERSIONS__

#include <ATen/native/hip/ck_gemm/ck_gemm_template.h>


namespace at::native {
    // TODO: need to decide padding and layout here
void
ck_gemm_kernel_bf16bf16bf16_256_128x128x64_32x32_2x2_8x32x1_8x32x1_1x32x1x8_4_Intrawave_v3(CUDABLAS_GEMM_ARGTYPES(at::BFloat16)) {

}
}; // namespace at::native
