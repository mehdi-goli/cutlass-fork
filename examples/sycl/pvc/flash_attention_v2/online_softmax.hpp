/***************************************************************************************************
 * Copyright (c) 2024 - 2024 Codeplay Software Ltd. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
/*! \file
  \brief Functor performing online softmax.
*/

#pragma once

#include <sycl/sycl.hpp>
#include "cutlass/cutlass.h"

#ifdef __SYCL_DEVICE_ONLY__
#define SYCL_DEVICE_OCL_FMA(x, y) SYCL_EXTERNAL x
#else
#define SYCL_DEVICE_OCL_FMA(x, y) \
  inline x                        \
  {                               \
    assert(false);                \
    return (y)0;                  \
  }
#endif

#define EXP sycl::native::exp
#define DIV sycl::native::divide
// TODO:: Temporary using OpenCL function as sycl_reduce_over_group spills on sycl::maximum<>() operation
SYCL_DEVICE_OCL_FMA(float sub_group_reduce_add(float i), float);
SYCL_DEVICE_OCL_FMA(float sub_group_reduce_max(float i), float);

#undef SYCL_DEVICE_OCL_FMA

namespace flash
{
  template <typename Element>
  struct Softmax
  {
    struct Arguments
    {
      Element scale;
    };

    using Params = Arguments;

    static constexpr Params
    to_underlying_arguments(Arguments const &args)
    {
      Arguments x{static_cast<Element>(args.scale) * static_cast<Element>(M_LOG2E)};
      return x;
    }

    template <
        bool CausalMask,
        int Vec,
        int FragsM,
        int FragsN,
        class FragAcc,
        class FragMax,
        class FragSum,
        class FragOut>
    CUTLASS_DEVICE static void run(bool is_first, FragAcc &frag_s, FragMax &max, FragSum &sum, FragOut &out, Params const &params)
    {
      if(!is_first)
      { // TODO:: For now we need to unroll it to get SIMD4 Scheduling for exp operation. This will be removed once we find a fix 
        CUTLASS_PRAGMA_UNROLL
        for (int y = 0; y < FragsM; y += 4 ) {
          CUTLASS_PRAGMA_UNROLL
          for (int x = 0; x < Vec; x++) { 
            sycl::vec<Element, 4> max_prev { max(x, y), max(x, y + 1), max(x, y + 2), max(x, y + 3)};
            CUTLASS_PRAGMA_UNROLL
            for (int z = 0; z < FragsN; z++)
            {
              max(x ,  y    ) = sycl::max(max(x,  y    ), frag_s(x, y    , z));
              max(x ,  y + 1) = sycl::max(max(x,  y + 1), frag_s(x, y + 1, z));
              max(x ,  y + 2) = sycl::max(max(x,  y + 2), frag_s(x, y + 2, z));
              max(x ,  y + 3) = sycl::max(max(x,  y + 3), frag_s(x, y + 3, z));
            }
            max(x , y    ) = sub_group_reduce_max(max(x , y    ));
            max(x , y + 1) = sub_group_reduce_max(max(x , y + 1));
            max(x , y + 2) = sub_group_reduce_max(max(x , y + 2));
            max(x , y + 3) = sub_group_reduce_max(max(x , y + 3));

            sycl::vec<Element, 4> curr_max_scale {(CausalMask && max(x, y    ) == -INFINITY) ? 0.f : max(x, y    ) * params.scale,
                                                  (CausalMask && max(x, y + 1) == -INFINITY) ? 0.f : max(x, y + 1) * params.scale, 
                                                  (CausalMask && max(x, y + 2) == -INFINITY) ? 0.f : max(x, y + 2) * params.scale, 
                                                  (CausalMask && max(x, y + 3) == -INFINITY) ? 0.f : max(x, y + 3) * params.scale};
            sycl::vec<Element, 4> eq = max_prev * params.scale - curr_max_scale; 
                              
            auto curr_scale = sycl::native::exp2(eq);
            sum(x, y    ) *= curr_scale[0];
            sum(x, y + 1) *= curr_scale[1];
            sum(x, y + 2) *= curr_scale[2];
            sum(x, y + 3) *= curr_scale[3];
                
            CUTLASS_PRAGMA_UNROLL
            for (int z = 0; z < FragsN; z++)
            {
              out(x, y    , z) *= curr_scale[0];
              out(x, y + 1, z) *= curr_scale[1];
              out(x, y + 2, z) *= curr_scale[2];
              out(x, y + 3, z) *= curr_scale[3];
              frag_s(x    , y, z) = sycl::native::exp2((frag_s(x    , y, z) * params.scale - curr_max_scale[0]));
              frag_s(x, y + 1, z) = sycl::native::exp2((frag_s(x, y + 1, z) * params.scale - curr_max_scale[1]));
              frag_s(x, y + 2, z) = sycl::native::exp2((frag_s(x, y + 2, z) * params.scale - curr_max_scale[2]));
              frag_s(x, y + 3, z) = sycl::native::exp2((frag_s(x, y + 3, z) * params.scale - curr_max_scale[3]));
              sum(x    , y) += frag_s(x    , y, z);
              sum(x, y + 1) += frag_s(x, y + 1, z);
              sum(x, y + 2) += frag_s(x, y + 2, z);
              sum(x, y + 3) += frag_s(x, y + 3, z);
            }
          }
        }
      } else {
        CUTLASS_PRAGMA_UNROLL
        for (int x = 0; x < Vec; x++)
        { 
          CUTLASS_PRAGMA_UNROLL
          for (int y = 0; y < FragsM; y++)
          { 
            CUTLASS_PRAGMA_UNROLL
            for (int z = 0; z < FragsN; z++)
            {
              max(x , y) = sycl::max(max(x , y), frag_s(x, y, z));
            }
            max(x , y) = sub_group_reduce_max(max(x , y));

            Element max_scaled {(CausalMask && max(x, y) == -INFINITY) ? 0.f : max(x, y) * params.scale};

            CUTLASS_PRAGMA_UNROLL
            for (int z = 0; z < FragsN; z += 2)
            { 
              sycl::vec<Element,2> eq {frag_s(x, y, z) * params.scale - max_scaled , frag_s(x, y, z + 1) * params.scale - max_scaled };
              frag_s(x, y, z) = sycl::native::exp2(eq[0]);
              frag_s(x, y, z + 1) = sycl::native::exp2(eq[1]);
              sum(x, y) += (frag_s(x, y, z) + frag_s(x, y, z + 1));
            }
          }
        }     
      }
    }

    Params params;
  };

}
