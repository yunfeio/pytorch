#undef __HIP_NO_HALF_CONVERSIONS__

#include <ATen/native/hip/ck_gemm/ck_gemm.h>
#include <ATen/native/hip/ck_gemm/ck_gemm_template.h>
#include <ATen/native/hip/ck_gemm/ck_gemm_kernels/ck_kernel_collection.h>
#include <ck/utility/sequence.hpp>

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

namespace at::native {

void dispatch_bfloat16_gemm(CUDABLAS_GEMM_ARGTYPES(at::BFloat16)) {
  //TODO_ANDY: this logic will mostly move to the individual kernels and be replaced with the size selection
  // If any of the shapes cant be tiled, we must use padding.
  // Dispatch to best implementation.

  // Determine padding and kernel size
  auto [kernel_size, use_padding] = get_kernel_size(m, n, k);
  std::cout << kernel_size << "" << use_padding << std::endl;
  if (kernel_size == KernelSize::Small) {
    ck_gemm_kernel_bf16bf16bf16_256_128x128x64_32x32_2x2_8x32x1_8x32x1_1x32x1x8_4_Intrawave_v3(
                                            CUDABLAS_GEMM_ARGS(at::BFloat16), use_padding);
  } else if (kernel_size == KernelSize::Large) {
    ck_gemm_kernel_bf16bf16bf16_256_256x128x64_16x16_8x4_8x32x1_8x32x1_1x32x1x8_4_Intrawave_v3(
                                            CUDABLAS_GEMM_ARGS(at::BFloat16), use_padding);
  } else { // kernel_size == KernelSize::Medium
    ck_gemm_kernel_bf16bf16bf16_256_128x128x64_32x32_2x2_8x32x1_8x32x1_1x32x1x8_4_Intrawave_v3(
                                            CUDABLAS_GEMM_ARGS(at::BFloat16), use_padding);
  }



}
  /*

  bool use_padding = ((m % 256 != 0) || (n % 128 != 0) || (k % 64 != 0));
  bool transa_ = std::tolower(transa) != 'n';
  bool transb_ = std::tolower(transb) != 'n';
  auto [size, use_pad] = get_kernel_size(m, n, k);
  std::cout << "kernel size: " << size << std::endl;
  std::cout << "use_padding: " << use_pad << std::endl;
  if (use_padding) {
    if (m <= 128) {
      if(transa_ && transb_) {
         gemm_impl<
            at::BFloat16,
            256,            // BLOCK_SIZE
            128,            // MBLOCK
            128,            // NBLOCK
            64,             // KBLOCK
            8,              // AK1
            8,              // BK1
            32,             // MPER_XDL
            32,             // NPER_XDL
            2,              // MPER_WAVE
            2,              // NPER_WAVE
            S<8,32,1>,      // ABLOCK_CLUSTER_LENS
            S<0,2,1>,       // ABLOCK_CLUSTER_ORDER
            S<0,2,1>,       // ABLOCK_SRC_ORDER
            1,              // ABLOCK_VECTOR_DIM
            4,              // ABLOCK_SCALAR_VEC
            8,              // ABLOCK_SCALAR_VEC_AK1
            0,              // ABLOCK_LDS_EXTRAM
            S<8,32,1>,      // BBLOCK_CLUSTER_LENS
            S<1,0,2>,       // BBLOCK_CLUSTER_ORDER
            S<1,0,2>,       // BBLOCK_SRC_ORDER
            2,              // BBLOCK_VECTOR_DIM
            8,              // BBLOCK_SCALAR_VEC
            8,              // BBLOCK_SCALAR_VEC_AK1
            0,              // BBLOCK_LDS_EXTRAN
            1,              // CMPER_WAVE
            1,              // CNPER_WAVE
            S<1, 32, 1, 8>, // BLOCK_CLUSTER_LENS
            S<4>,           // CDE_SCALAR_VEC
            true,           // PADDING
            true,           // TRANSA
            true>           // TRANSB
            (CUDABLAS_GEMM_ARGS(at::BFloat16));
      }
      else if(transa_ && !transb_) {
        gemm_impl<
            at::BFloat16,
            256,
            128,
            128,
            64,
            8,
            8,
            32,
            32,
            2,
            2,
            S<8,32,1>,
            S<1,0,2>,
            S<1,0,2>,
            2,
            8,
            8,
            0,
            S<8,32,1>,
            S<1,0,2>,
            S<1,0,2>,
            2,
            8,
            8,
            0,
            1,
            1,
            S<1, 32, 1, 8>,
            S<4>,
            true,
            true,
            false>
            (CUDABLAS_GEMM_ARGS(at::BFloat16));
      }
      else if(!transa_ && transb_) {
          gemm_impl<
            at::BFloat16,
            256,
            128,
            128,
            64,
            8,
            8,
            32,
            32,
            2,
            2,
            S<8,32,1>,
            S<0,2,1>,
            S<0,2,1>,
            1,
            4,
            8,
            0,
            S<8,32,1>,
            S<0,2,1>,
            S<0,2,1>,
            1,
            4,
            8,
            0,
            1,
            1,
            S<1, 32, 1, 8>,
            S<4>,
            true,
            false,
            true>
            (CUDABLAS_GEMM_ARGS(at::BFloat16));
      }
      else if(!transa_ && !transb_) {
          gemm_impl<
           at::BFloat16,
            256,
            128,
            128,
            64,
            8,
            8,
            32,
            32,
            2,
            2,
            S<8,32,1>,
            S<1,0,2>,
            S<1,0,2>,
            2,
            8,
            8,
            0,
            S<8,32,1>,
            S<0,2,1>,
            S<0,2,1>,
            1,
            4,
            8,
            0,
            1,
            1,
            S<1, 32, 1, 8>,
            S<4>,
            true,
            false,
            false>
            (CUDABLAS_GEMM_ARGS(at::BFloat16));
      }
      else {
        TORCH_CHECK(false, "unreachable");
      }

    } else {
      if(transa_ && transb_) {
        gemm_impl<
            at::BFloat16,
            256,
            128,
            128,
            64,
            8,
            8,
            32,
            32,
            2,
            2,
            S<8,32,1>,
            S<0,2,1>,
            S<0,2,1>,
            1,
            4,
            8,
            0,
            S<8,32,1>,
            S<1,0,2>,
            S<1,0,2>,
            2,
            8,
            8,
            0,
            1,
            1,
            S<1, 32, 1, 8>,
            S<4>,
            true,
            true,
            true>
            (CUDABLAS_GEMM_ARGS(at::BFloat16));
      }
      else if(transa_ && !transb_) {
          gemm_impl<
            at::BFloat16,
            256,
            128,
            128,
            64,
            8,
            8,
            32,
            32,
            2,
            2,
            S<8,32,1>,
            S<1,0,2>,
            S<1,0,2>,
            2,
            8,
            8,
            0,
            S<8,32,1>,
            S<1,0,2>,
            S<1,0,2>,
            2,
            8,
            8,
            0,
            1,
            1,
            S<1, 32, 1, 8>,
            S<4>,
            true,
            true,
            false>
            (CUDABLAS_GEMM_ARGS(at::BFloat16));
      }
      else if(!transa_ && transb_) {
          gemm_impl<
            at::BFloat16,
            256,
            128,
            128,
            64,
            8,
            8,
            32,
            32,
            2,
            2,
            S<8,32,1>,
            S<0,2,1>,
            S<0,2,1>,
            1,
            4,
            8,
            0,
            S<8,32,1>,
            S<0,2,1>,
            S<0,2,1>,
            1,
            4,
            8,
            0,
            1,
            1,
            S<1, 32, 1, 8>,
            S<4>,
            true,
            false,
            true>
            (CUDABLAS_GEMM_ARGS(at::BFloat16));
      }
      else if(!transa_ && !transb_) {
          gemm_impl<
            at::BFloat16,
            256,
            128,
            128,
            64,
            8,
            8,
            32,
            32,
            2,
            2,
            S<8,32,1>,
            S<1,0,2>,
            S<1,0,2>,
            2,
            8,
            8,
            0,
            S<8,32,1>,
            S<0,2,1>,
            S<0,2,1>,
            1,
            4,
            8,
            0,
            1,
            1,
            S<1, 32, 1, 8>,
            S<4>,
            true,
            false,
            false>
            (CUDABLAS_GEMM_ARGS(at::BFloat16));
      }
      else {
        TORCH_CHECK(false, "unreachable");
      }
    }
  } else {
    {
      if(transa_ && transb_) {
          gemm_impl<
            at::BFloat16,
            256,
            128,
            128,
            64,
            8,
            8,
            32,
            32,
            2,
            2,
            S<8,32,1>,
            S<0,2,1>,
            S<0,2,1>,
            1,
            4,
            8,
            0,
            S<8,32,1>,
            S<1,0,2>,
            S<1,0,2>,
            2,
            8,
            8,
            0,
            1,
            1,
            S<1, 32, 1, 8>,
            S<4>,
            false,
            true,
            true>
            (CUDABLAS_GEMM_ARGS(at::BFloat16));
      }
      else if(transa_ && !transb_) {
          gemm_impl<
            at::BFloat16,
            256,
            128,
            128,
            64,
            8,
            8,
            32,
            32,
            2,
            2,
            S<8,32,1>,
            S<1,0,2>,
            S<1,0,2>,
            2,
            8,
            8,
            0,
            S<8,32,1>,
            S<1,0,2>,
            S<1,0,2>,
            2,
            8,
            8,
            0,
            1,
            1,
            S<1, 32, 1, 8>,
            S<4>,
            false,
            true,
            false>
            (CUDABLAS_GEMM_ARGS(at::BFloat16));
      }
      else if(!transa_ && transb_) {
          gemm_impl<
            at::BFloat16,
            256,
            128,
            128,
            64,
            8,
            8,
            32,
            32,
            2,
            2,
            S<8,32,1>,
            S<0,2,1>,
            S<0,2,1>,
            1,
            4,
            8,
            0,
            S<8,32,1>,
            S<0,2,1>,
            S<0,2,1>,
            1,
            4,
            8,
            0,
            1,
            1,
            S<1, 32, 1, 8>,
            S<4>,
            false,
            false,
            true>
            (CUDABLAS_GEMM_ARGS(at::BFloat16));
      }
      else if(!transa_ && !transb_) {
          gemm_impl<
            at::BFloat16,
            256,
            128,
            128,
            64,
            8,
            8,
            32,
            32,
            2,
            2,
            S<8,32,1>,
            S<1,0,2>,
            S<1,0,2>,
            2,
            8,
            8,
            0,
            S<8,32,1>,
            S<0,2,1>,
            S<0,2,1>,
            1,
            4,
            8,
            0,
            1,
            1,
            S<1, 32, 1, 8>,
            S<4>,
            false,
            false,
            false>
            (CUDABLAS_GEMM_ARGS(at::BFloat16));
      }
      else {
        TORCH_CHECK(false, "unreachable");
      }
    }
  }
}

*/
void dispatch_bfloat16_gemm_wmma(CUDABLAS_GEMM_ARGTYPES(at::BFloat16)) {
  // If any of the shapes cant be tiled, we must use padding.
  bool use_padding = ((m % 256 != 0) || (n % 128 != 0) || (k % 64 != 0));
  // Dispatch to best implementation.
  // TODO add more configurations. Optimize.

  bool transa_ = std::tolower(transa) != 'n';
  bool transb_ = std::tolower(transb) != 'n';

  if (use_padding) {
      if(transa_ && transb_) { // col , col
          gemm_impl_wmma<
            at::BFloat16,
            256,
            128,
            256,
            64,
            8,
            16,
            16,
            4,
            4,
            S<4, 64, 1>,
            S<0, 2, 1>,
            S<0, 2, 1>,
            1,
            1,
            8,
            true,
            S<4, 64, 1>,
            S<1, 0, 2>,
            S<1, 0, 2>,
            2,
            8,
            8,
            true,
            1,
            1,
            S<1, 32, 1,  8>,
            8,
            true,
            true,
            true>
            (CUDABLAS_GEMM_ARGS(at::BFloat16));
      }
      else if(transa_ && !transb_) { // row, col
          gemm_impl_wmma<
            at::BFloat16,
            256,
            128,
            256,
            64,
            8,
            16,
            16,
            4,
            4,
            S<4, 64, 1>,
            S<1, 0, 2>,
            S<1, 0, 2>,
            2,
            8,
            8,
            true,
            S<4, 64, 1>,
            S<1, 0, 2>,
            S<1, 0, 2>,
            2,
            8,
            8,
            true,
            1,
            1,
            S<1, 32, 1,  8>,
            8,
            true,
            true,
            false>
            (CUDABLAS_GEMM_ARGS(at::BFloat16));
      }
      else if(!transa_ && transb_) { //col, row
          gemm_impl_wmma<
            at::BFloat16,
            256,
            128,
            256,
            64,
            8,
            16,
            16,
            4,
            4,
            S<4, 64, 1>,
            S<0, 2, 1>,
            S<0, 2, 1>,
            1,
            1,
            8,
            true,
            S<4, 64, 1>,
            S<0, 2, 1>,
            S<0, 2, 1>,
            1,
            1,
            8,
            true,
            1,
            1,
            S<1, 32, 1,  8>,
            8,
            true,
            false,
            true>
            (CUDABLAS_GEMM_ARGS(at::BFloat16));
      }
      else if(!transa_ && !transb_) { //row, row
          gemm_impl_wmma<
            at::BFloat16,
            256,
            128,
            256,
            64,
            8,
            16,
            16,
            4,
            4,
            S<4, 64, 1>,
            S<1, 0, 2>,
            S<1, 0, 2>,
            2,
            8,
            8,
            true,
            S<4, 64, 1>,
            S<0, 2, 1>,
            S<0, 2, 1>,
            1,
            1,
            8,
            true,
            1,
            1,
            S<1, 32, 1,  8>,
            8,
            true,
            false,
            false>
            (CUDABLAS_GEMM_ARGS(at::BFloat16));
      }
      else {
        TORCH_CHECK(false, "unreachable");
      }
  } else {
         if(transa_ && transb_) { // col , col
          gemm_impl_wmma<
            at::BFloat16,
            256,
            128,
            256,
            64,
            8,
            16,
            16,
            4,
            4,
            S<4, 64, 1>,
            S<0, 2, 1>,
            S<0, 2, 1>,
            1,
            1,
            8,
            true,
            S<4, 64, 1>,
            S<1, 0, 2>,
            S<1, 0, 2>,
            2,
            8,
            8,
            true,
            1,
            1,
            S<1, 32, 1,  8>,
            8,
            false,
            true,
            true>
            (CUDABLAS_GEMM_ARGS(at::BFloat16));
      }
      else if(transa_ && !transb_) { // row, col
          gemm_impl_wmma<
            at::BFloat16,
            256,
            128,
            256,
            64,
            8,
            16,
            16,
            4,
            4,
            S<4, 64, 1>,
            S<1, 0, 2>,
            S<1, 0, 2>,
            2,
            8,
            8,
            true,
            S<4, 64, 1>,
            S<1, 0, 2>,
            S<1, 0, 2>,
            2,
            8,
            8,
            true,
            1,
            1,
            S<1, 32, 1,  8>,
            8,
            false,
            true,
            false>
            (CUDABLAS_GEMM_ARGS(at::BFloat16));
      }
      else if(!transa_ && transb_) { //col, row
          gemm_impl_wmma<
            at::BFloat16,
            256,
            128,
            256,
            64,
            8,
            16,
            16,
            4,
            4,
            S<4, 64, 1>,
            S<0, 2, 1>,
            S<0, 2, 1>,
            1,
            1,
            8,
            true,
            S<4, 64, 1>,
            S<0, 2, 1>,
            S<0, 2, 1>,
            1,
            1,
            8,
            true,
            1,
            1,
            S<1, 32, 1,  8>,
            8,
            false,
            false,
            true>
            (CUDABLAS_GEMM_ARGS(at::BFloat16));
      }
      else if(!transa_ && !transb_) { //row, row
          gemm_impl_wmma<
            at::BFloat16,
            256,
            128,
            256,
            64,
            8,
            16,
            16,
            4,
            4,
            S<4, 64, 1>,
            S<1, 0, 2>,
            S<1, 0, 2>,
            2,
            8,
            8,
            true,
            S<4, 64, 1>,
            S<0, 2, 1>,
            S<0, 2, 1>,
            1,
            1,
            8,
            true,
            1,
            1,
            S<1, 32, 1,  8>, 8,
            false,
            false,
            false>
            (CUDABLAS_GEMM_ARGS(at::BFloat16));
      }
      else {
        TORCH_CHECK(false, "unreachable");
      }
  }
}

template <>
void gemm_internal_ck<at::BFloat16>(CUDABLAS_GEMM_ARGTYPES(at::BFloat16)) {
  auto dprops = at::cuda::getCurrentDeviceProperties();
  c10::string_view arch(dprops->gcnArchName);
  if (arch == "gfx1100") {
    dispatch_bfloat16_gemm_wmma(CUDABLAS_GEMM_ARGS(at::BFloat16));
  } else{
    dispatch_bfloat16_gemm(CUDABLAS_GEMM_ARGS(at::BFloat16));
  }
}

} // namespace at::native
