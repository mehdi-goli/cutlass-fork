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
        class FragSum
        >
    CUTLASS_DEVICE static constexpr void scale_exp_log2(FragAcc &acc, FragMax const max,  FragSum& sum, Element const scale)
    {
      auto g = syclcompat::get_nd_item<1>().get_sub_group();
      int x = g.get_local_id()[0] ;
      CUTLASS_PRAGMA_UNROLL
      for (int y = 0; y < FragsM; y++)
      { 
        CUTLASS_PRAGMA_UNROLL
        for (int x = 0; x < Vec; x++)
        { 
          Element max_scaled = (group_broadcast(g, max[(y*Vec + x )/16],  (y*Vec +x) % 16))* scale;
          if (CausalMask && max_scaled == -INFINITY)
          {
            max_scaled = 0.f;
          }

          CUTLASS_PRAGMA_UNROLL
          for (int z = 0; z < FragsN; z += 2)
          { 
            Element eq0 = (acc(x, y, z) * scale - max_scaled);
            Element eq1 = (acc(x, y, z + 1) * scale - max_scaled);
            acc(x, y, z) = sycl::native::exp2(eq0);
            acc(x, y, z + 1) = sycl::native::exp2(eq1);
            sum(x, y) += (acc(x, y, z) + acc(x, y, z + 1));
          }
        }
      }
    }

     template <
        int Vec,
        int FragsM,
        int FragsN,
        class FragSrc,
        class FragMax>
    CUTLASS_DEVICE static void reduce_max(FragSrc const &src, FragMax& max)
    {       
      auto g = syclcompat::get_nd_item<1>().get_sub_group();
      Element maxptr[Vec*FragsM];
      CUTLASS_PRAGMA_UNROLL
      for (int y = 0; y < FragsM; y++)
      {
        CUTLASS_PRAGMA_UNROLL
        for (int x = 0; x < Vec; x++)     
        {
          maxptr[x +  y* Vec] = (group_broadcast(g, max[(y*Vec + x )/16],  (y*Vec +x) % 16));
          CUTLASS_PRAGMA_UNROLL
          for (int z = 0; z < FragsN; z++)
          {
            maxptr[x +  y* Vec] = sycl::max(maxptr[x +  y* Vec], src(x, y, z));
          }
          maxptr[x+ y*Vec] = sub_group_reduce_max(maxptr[x+ y*Vec]);
        }
      }
      int x = g.get_local_id()[0] ;
      max = {maxptr[x], maxptr[x+16]};
    }

    template <
        int Vec,
        int FragsM,
        int FragsN,
        class FragDst>
      // reduce across the sub_group to get the final output
    CUTLASS_DEVICE static void subgroup_allreduce_sum(FragDst &dst)
    {
      CUTLASS_PRAGMA_UNROLL
      for (int y = 0; y < FragsM; y++)
      {
        CUTLASS_PRAGMA_UNROLL
        for (int x = 0; x < Vec; x++)
        {
          dst(x, y) = sub_group_reduce_add(dst(x, y));
        }
      }
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
    CUTLASS_DEVICE static void run(bool is_first, FragAcc &frag_s, FragMax &max, FragSum &sum, FragOut &out, Params const  &params)
    {
      auto max_prev = max;
      reduce_max<Vec, FragsM, FragsN>(frag_s, max);  
      static_assert(Vec==8 && FragsM==4);
      if(!is_first) {
        auto g = syclcompat::get_nd_item<1>().get_sub_group();
        auto curr_max_scale = max * params.scale;
        auto curr_scale = sycl::native::exp2(max_prev* params.scale - curr_max_scale); 
        CUTLASS_PRAGMA_UNROLL
        for (int y = 0; y < FragsM; y++ )     //4
        { 
          CUTLASS_PRAGMA_UNROLL
          for (int x =0; x < Vec; x++)  //8
          {
            const Element curr_scale_bcast = group_broadcast(g, curr_scale[(y*Vec + x )/16],  (y*Vec +x) % 16);
            Element max_scaled = (group_broadcast(g, max[(y*Vec + x )/16],  (y*Vec +x) % 16))* params.scale;
            sum(x, y) *= curr_scale_bcast;
            CUTLASS_PRAGMA_UNROLL
            for (int z = 0; z < FragsN; z++) //2
            {
              out(x, y, z) *= curr_scale_bcast;
              frag_s(x, y, z) = sycl::native::exp2((frag_s(x, y, z) * params.scale - max_scaled));
              sum(x, y) += frag_s(x, y, z);
            }
          }
        }
      } else {
        scale_exp_log2<CausalMask, Vec, FragsM, FragsN>(frag_s, max, sum, params.scale);
      }
    }
    Params params;
  };

}