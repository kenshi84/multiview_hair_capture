// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
//
// Reference: "Strand-accurate Multi-view Hair Capture"
//            G. Nam, C. Wu, M.H. Kim, Y. Sheikh (CVPR 2019)

#pragma once

#ifndef T_PER_BLOCK
#define T_PER_BLOCK 16
#endif

#ifndef MINF
#define MINF __int_as_float(0xff800000)
#endif

#ifndef CUDART_PI_F
#define CUDART_PI_F 3.141592654f
#endif

#define ORIENT2D_ROTATE_RES 180
