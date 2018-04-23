// -------------------------------------------------------------
// cuDPP -- CUDA Data Parallel Primitives library
// -------------------------------------------------------------
//  $Revision: 4730 $
//  $Date: 2009-01-14 21:38:38 -0800 (Wed, 14 Jan 2009) $
// -------------------------------------------------------------
/*
 *  CUDA SDK END USER LICENSE AGREEMENT ("Agreement")
 *  BY DOWNLOADING THE SOFTWARE AND OTHER AVAILABLE MATERIALS, YOU  ("DEVELOPER") AGREE TO BE BOUND BY THE FOLLOWING TERMS AND CONDITIONS OF THIS AGREEMENT.  IF DEVELOPER DOES NOT AGREE TO THE TERMS AND CONDITION OF THIS AGREEMENT, THEN DO NOT DOWNLOAD THE SOFTWARE AND MATERIALS.
 *  The materials available for download to Developers may include software in both sample source ("Source Code") and object code ("Object Code") versions, documentation ("Documentation"), certain art work ("Art Assets") and other materials (collectively, these materials referred to herein as "Materials").  Except as expressly indicated herein, all terms and conditions of this Agreement apply to all of the Materials.
 *  Except as expressly set forth herein, NVIDIA owns all of the Materials and makes them available to Developer only under the terms and conditions set forth in this Agreement.
 *  License:  Subject to the terms of this Agreement, NVIDIA hereby grants to Developer a royalty-free, non-exclusive license to possess and to use the Materials.  Developer may install and use multiple copies of the Materials on a shared computer or concurrently on different computers, and make multiple back-up copies of the Materials, solely for Licensee’s use within Licensee’s Enterprise. “Enterprise” shall mean individual use by Licensee or any legal entity (such as a corporation or university) and the subsidiaries it owns by more than 50 percent.  The following terms apply to the specified type of Material:
 *  Source Code:  Developer shall have the right to modify and create derivative works with the Source Code.  Developer shall own any derivative works ("Derivatives") it creates to the Source Code, provided that Developer uses the Materials in accordance with the terms and conditions of this Agreement.  Developer may distribute the Derivatives, provided that all NVIDIA copyright notices and trademarks are used properly and the Derivatives include the following statement: "This software contains source code provided by NVIDIA Corporation."
 *  Object Code:  Developer agrees not to disassemble, decompile or reverse engineer the Object Code versions of any of the Materials.  Developer acknowledges that certain of the Materials provided in Object Code version may contain third party components that may be subject to restrictions, and expressly agrees not to attempt to modify or distribute such Materials without first receiving consent from NVIDIA.
 *  Art Assets:  Developer shall have the right to modify and create Derivatives of the Art Assets, but may not distribute any of the Art Assets or Derivatives created therefrom without NVIDIA’s prior written consent.
 *  No Other License: No rights or licenses are granted by NVIDIA to Developer under this Agreement, expressly or by implication, with respect to any proprietary information or patent, copyright, trade secret or other intellectual property right owned or controlled by NVIDIA, except as expressly provided in this Agreement.
 *  Intellectual Property Ownership: All rights, title, interest and copyrights in and to the Materials (including but not limited to all images, photographs, animations, video, audio, music, text, and other information incorporated into the Materials), are owned by NVIDIA, or its suppliers. The Materials are protected by copyright laws and international treaty provisions. Accordingly, Developer is required to treat the Materials like any other copyrighted material, except as otherwise allowed pursuant to this Agreement.
 *  Term of Agreement:  This Agreement is effective until (i) automatically terminated if Developer fails to comply with any of the terms and conditions of this Agreement; or (ii) terminated by NVIDIA.  NVIDIA may terminate this Agreement (and with it, all of Developer’s right to the Materials) immediately upon written notice (which may include email) to Developer, with or without cause.
 *  Defensive Suspension: If Developer commences or participates in any legal proceeding against NVIDIA, then NVIDIA may, in its sole discretion, suspend or terminate all license grants and any other rights provided under this Agreement during the pendency of such legal proceedings.
 *  No Support:  NVIDIA has no obligation to support or to continue providing or updating any of the Materials.
 *  No Warranty:  THE SOFTWARE AND ANY OTHER MATERIALS PROVIDED BY NVIDIA TO DEVELOPER HEREUNDER ARE PROVIDED "AS IS."  NVIDIA DISCLAIMS ALL WARRANTIES, EXPRESS, IMPLIED OR STATUTORY, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTIES OF TITLE, MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 *  Limitation of Liability: NVIDIA SHALL NOT BE LIABLE TO DEVELOPER, DEVELOPER’S CUSTOMERS, OR ANY OTHER PERSON OR ENTITY CLAIMING THROUGH OR UNDER DEVELOPER FOR ANY LOSS OF PROFITS, INCOME, SAVINGS, OR ANY OTHER CONSEQUENTIAL, INCIDENTAL, SPECIAL, PUNITIVE, DIRECT OR INDIRECT DAMAGES (WHETHER IN AN ACTION IN CONTRACT, TORT OR BASED ON A WARRANTY), EVEN IF NVIDIA HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGES.  THESE LIMITATIONS SHALL APPLY NOTWITHSTANDING ANY FAILURE OF THE ESSENTIAL PURPOSE OF ANY LIMITED REMEDY.  IN NO EVENT SHALL NVIDIA’S AGGREGATE LIABILITY TO DEVELOPER OR ANY OTHER PERSON OR ENTITY CLAIMING THROUGH OR UNDER DEVELOPER EXCEED THE AMOUNT OF MONEY ACTUALLY PAID BY DEVELOPER TO NVIDIA FOR THE SOFTWARE OR ANY OTHER MATERIALS.
 *  Applicable Law: This Agreement shall be deemed to have been made in, and shall be construed pursuant to, the laws of the State of Delaware. The United Nations Convention on Contracts for the International Sale of Goods is specifically disclaimed.
 *  Feedback: In the event Developer contacts NVIDIA to request Feedback (as defined below) on how to optimize Developer’s product for use with the Materials, the following terms and conditions apply the Feedback:

 *  1.	Exchange of Feedback. Both parties agree that neither party has an obligation to give the other party any suggestions, comments or other feedback, whether verbally or in code form (“Feedback”), relating to (i) the Materials; (ii) Developer’s products; (iii) Developer’s use of the Materials; or (iv) optimization of Developer’s product with CUDA.  In the event either party provides Feedback to the other party, the party receiving the Feedback may use and include any Feedback that the other party voluntarily provides to improve the (i) Materials or other related NVIDIA technologies, respectively for the benefit of NVIDIA; or (ii) Developer’s product or other related Developer technologies, respectively for the benefit of Developer.  Accordingly, if either party provides Feedback to the other party, both parties agree that the other party and its respective Developers may freely use, reproduce, license, distribute, and otherwise commercialize the Feedback in the (i) Materials or other related technologies; or (ii) Developer’s products or other related technologies, respectively, without the payment of any royalties or fees.
 *  2.	Residual Rights. Developer agrees that NVIDIA shall be free to use any general knowledge, skills and experience, (including, but not limited to, ideas, concepts, know-how, or techniques) (“Residuals”), contained in the (i) Feedback provided by Developer to NVIDIA; (ii) Developer’s products, in source or object code form, shared or disclosed to NVIDIA in connection with the Feedback; or (c) Developer’s confidential information voluntarily provided to NVIDIA in connection with the Feedback, which are retained in the memories of NVIDIA’s employees, agents, or contractors who have had access to such (i) Feedback provided by Developer to NVIDIA; (ii) Developer’s products; or (c) Developer’s confidential information voluntarily provided to NVIDIA, in connection with the Feedback.  Subject to the terms and conditions of this Agreement, NVIDIA’s employees, agents, or contractors shall not be prevented from using Residuals as part of such employee’s, agent’s or contractor’s general knowledge, skills, experience, talent, and/or expertise.  NVIDIA shall not have any obligation to limit or restrict the assignment of such employees, agents or contractors or to pay royalties for any work resulting from the use of Residuals.
 *  3.	Disclaimer of Warranty. FEEDBACK FROM EITHER PARTY IS PROVIDED FOR THE OTHER PARTY’S USE “AS IS” AND BOTH PARTIES DISCLAIM ALL WARRANTIES, EXPRESS, IMPLIED AND STATUTORY INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.  BOTH PARTIES DO NOT REPRESENT OR WARRANT THAT THE FEEDBACK WILL MEET THE OTHER PARTY’S REQUIREMENTS OR THAT THE OPERATION OR IMPLEMENTATION OF THE FEEDBACK WILL BE UNINTERRUPTED OR ERROR-FREE.
 *  4.	No Liability for Consequential Damages. TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, IN NO EVENT SHALL EITHER PARTY OR ITS SUPPLIERS BE LIABLE FOR ANY SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES WHATSOEVER (INCLUDING, WITHOUT LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS, BUSINESS INTERRUPTION, LOSS OF BUSINESS INFORMATION, OR ANY OTHER PECUNIARY LOSS) ARISING OUT OF THE USE OF OR INABILITY TO USE THE FEEDBACK, EVEN IF THE OTHER PARTY HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGES.
 *  5.	Freedom of Action.  Developer agrees that this Agreement is nonexclusive and NVIDIA may currently or in the future be developing software, other technology or confidential information internally, or receiving confidential information from other parties that maybe similar to the Feedback and Developer’s confidential information (as provided in subsection 2 above), which may be provided to NVIDIA in connection with Feedback by Developer.  Accordingly, Developer agrees that nothing in this Agreement will be construed as a representation or inference that NVIDIA will not develop, design, manufacture, acquire, market products, or have products developed, designed, manufactured, acquired, or marketed for NVIDIA, that compete with the Developer’s products or confidential information.
 *  RESTRICTED RIGHTS NOTICE: Materials has been developed entirely at private expense and is commercial computer software provided with RESTRICTED RIGHTS. Use, duplication or disclosure by the U.S. Government or a U.S. Government subcontractor is subject to the restrictions set forth in the license agreement under which Materials was obtained pursuant to DFARS 227.7202-3(a) or as set forth in subparagraphs (c)(1) and (2) of the Commercial Computer Software - Restricted Rights clause at FAR 52.227-19, as applicable. Contractor/manufacturer is NVIDIA, 2701 San Tomas Expressway, Santa Clara, CA 95050.
 *  Miscellaneous: If any provision of this Agreement is inconsistent with, or cannot be fully enforced under, the law, such provision will be construed as limited to the extent necessary to be consistent with and fully enforceable under the law. This Agreement is the final, complete and exclusive agreement between the parties relating to the subject matter hereof, and supersedes all prior or contemporaneous understandings and agreements relating to such subject matter, whether oral or written. This Agreement may only be modified in writing signed by an authorized officer of NVIDIA. Developer agrees that it will not ship, transfer or export the Materials into any country, or use the Materials in any manner, prohibited by the United States Bureau of Industry and Security or any export laws, restrictions or regulations.
 */
// -------------------------------------------------------------

/////////////////////////////////////////////////////////////////////////////
// MD5 random generator  (updated from cudpp)
/////////////////////////////////////////////////////////////////////////////


#ifndef __CUDA_MD5_RAND_H_
#define __CUDA_MD5_RAND_H_

#include "vector_types.h"
#include "cuda_defs.h"
#include <cutil.h>
#define RAND_CTA_SIZE 128 //128 chosen, may be changed later



//------------MD5 ROTATING FUNCTIONS------------------------

/**
 * @brief Does a GLSL-style swizzle assigning f->xyzw = f->yzwx
 *
 *  It does the equvalent of f->xyzw = f->yzwx since this functionality is
 *  in shading languages but not exposed in CUDA.
 *  @param[in] f the uint4 data type which will have its elements shifted.  Passed in as pointer.
 *
**/
__device__ void swizzleShift(uint4 *f)
{
	unsigned int temp;
	temp = f->x;
	f->x = f->y;
	f->y = f->z;
	f->z = f->w;
	f->w = temp;
}
/**
 * @brief Rotates the bits in \a x over by \a n bits.
 *
 *  This is the equivalent of the ROTATELEFT operation as described in the MD5 working memo.
 *  It takes the bits in \a x and circular shifts it over by \a n bits.
 *
 *  For more information see: <a href="http://tools.ietf.org/html/rfc1321">The MD5 Message-Digest Algorithm</a>
 *
 *  @param[in] x the variable with the bits
 *  @param[in] n the number of bits to shift left by.
**/
__device__ unsigned int leftRotate(unsigned int x, unsigned int n)
{
	unsigned int t = (((x) << (n)) | ((x) >> (32 - n))) ;
	return t;
}

/**
 * @brief The F scrambling function.
 *
 *  The F function in the MD5 technical memo scrambles three variables
 *  \a x, \a y, and \a z in the following way using bitwise logic:
 *
 *  (x & y) | ((~x) & z)
 *
 *  The resulting value is returned as an unsigned int.
 *
 *  For more information see: <a href="http://tools.ietf.org/html/rfc1321">The MD5 Message-Digest Algorithm</a>
 *
 *  @param[in] x See the above formula
 *  @param[in] y See the above formula
 *  @param[in] z See the above formula
 *
 *  @see FF()
**/
__device__ unsigned int F(unsigned int x, unsigned int y, unsigned int z)
{
	unsigned int t;
	t = ((x & y) | ((~x) & z));
	return t;
}

/**
 * @brief The G scrambling function.
 *
 *  The G function in the MD5 technical memo scrambles three variables
 *  \a x, \a y, and \a z in the following way using bitwise logic:
 *
 *  (x & z) | ((~z) & y)
 *
 *  The resulting value is returned as an unsigned int.
 *
 *  For more information see: <a href="http://tools.ietf.org/html/rfc1321">The MD5 Message-Digest Algorithm</a>
 *
 *  @param[in] x See the above formula
 *  @param[in] y See the above formula
 *  @param[in] z See the above formula
 *
 *  @see GG()
**/
__device__ unsigned int G(unsigned int x, unsigned int y, unsigned int z)
{
	unsigned int t;
	t = ((x & z) | ((~z) & y));
	return t;
}

/**
 * @brief The H scrambling function.
 *
 *  The H function in the MD5 technical memo scrambles three variables
 *  \a x, \a y, and \a z in the following way using bitwise logic:
 *
 *  (x ^ y ^ z)
 *
 *  The resulting value is returned as an unsigned int.
 *
 *  For more information see: <a href="http://tools.ietf.org/html/rfc1321">The MD5 Message-Digest Algorithm</a>
 *
 *  @param[in] x See the above formula
 *  @param[in] y See the above formula
 *  @param[in] z See the above formula
 *
 *  @see HH()
**/
__device__ unsigned int H(unsigned int x, unsigned int y, unsigned int z)
{
	unsigned int t;
	t = (x ^ y ^ z);
	return t;
}

/**
 * @brief The I scrambling function.
 *
 *  The I function in the MD5 technical memo scrambles three variables
 *  \a x, \a y, and \a z in the following way using bitwise logic:
 *
 *  (y ^ (x | ~z))
 *
 *  The resulting value is returned as an unsigned int.
 *
 *  For more information see: <a href="http://tools.ietf.org/html/rfc1321">The MD5 Message-Digest Algorithm</a>
 *
 *  @param[in] x See the above formula
 *  @param[in] y See the above formula
 *  @param[in] z See the above formula
 *
 *  @see II()
**/
__device__ unsigned int I(unsigned int x, unsigned int y, unsigned int z)
{
	unsigned int t;
	t = (y ^(x | ~z));
	return t;
}

/**
 * @brief The FF scrambling function
 *
 *  The FF function in the MD5 technical memo is a wrapper for the F scrambling function
 *  as well as performing its own rotations using LeftRotate and swizzleShift.  The variable
 *  \a td is the current scrambled digest which is passed along and scrambled using the current
 *  iteration \a i, the rotation information \a Fr, and the starting input \a data.  \a p is kept as a
 *  constant of 2^32.
 *  The resulting value is stored in \a td.
 *
 *  For more information see: <a href="http://tools.ietf.org/html/rfc1321">The MD5 Message-Digest Algorithm</a>
 *
 *  @param[in,out] td The current value of the digest stored as an uint4.
 *  @param[in] i  The current iteration of the algorithm.  This affects the values in \a data.
 *  @param[in] Fr The current rotation order.
 *  @param[in] p The constant 2^32.
 *  @param[in] data The starting input to MD5.  Padded from setupInput().
 *
 *  @see F()
 *  @see swizzleShift()
 *  @see leftRotate()
 *  @see setupInput()
**/
__device__ void FF(uint4 * td, int i, uint4 * Fr, float p, unsigned int * data)
{
	unsigned int Ft = F(td->y, td->z, td->w);
	unsigned int r = Fr->x;
	swizzleShift(Fr);
	
	float t = sin(__int_as_float(i)) * p;
	unsigned int trigFunc = __float2uint_rd(t);
	td->x = td->y + leftRotate(td->x + Ft + trigFunc + data[i], r);
	swizzleShift(td);
}

/**
 * @brief The GG scrambling function
 *
 *  The GG function in the MD5 technical memo is a wrapper for the G scrambling function
 *  as well as performing its own rotations using LeftRotate() and swizzleShift().  The variable
 *  \a td is the current scrambled digest which is passed along and scrambled using the current
 *  iteration \a i, the rotation information \a Gr, and the starting input \a data.  \a p is kept as a
 *  constant of 2^32.
 *  The resulting value is stored in \a td.
 *
 *  For more information see: <a href="http://tools.ietf.org/html/rfc1321">The MD5 Message-Digest Algorithm</a>
 *
 *  @param[in,out] td The current value of the digest stored as an uint4.
 *  @param[in] i  The current iteration of the algorithm.  This affects the values in \a data.
 *  @param[in] Gr The current rotation order.
 *  @param[in] p The constant 2^32.
 *  @param[in] data The starting input to MD5.  Padded from setupInput().
 *
 *  @see G()
 *  @see swizzleShift()
 *  @see leftRotate()
 *  @see setupInput()
**/
__device__ void GG(uint4 * td, int i, uint4 * Gr, float p, unsigned int * data)
{
	unsigned int Ft = G(td->y, td->z, td->w);
	i = (5 * i + 1) % 16;
	unsigned int r = Gr->x;
	swizzleShift(Gr);
	
	float t = sin(__int_as_float(i)) * p;
	unsigned int trigFunc = __float2uint_rd(t);
	td->x = td->y + leftRotate(td->x + Ft + trigFunc + data[i], r);
	swizzleShift(td);
}

/**
 * @brief The HH scrambling function
 *
 *  The HH function in the MD5 technical memo is a wrapper for the H scrambling function
 *  as well as performing its own rotations using LeftRotate() and swizzleShift().  The variable
 *  \a td is the current scrambled digest which is passed along and scrambled using the current
 *  iteration \a i, the rotation information \a Hr, and the starting input \a data.  \a p is kept as a
 *  constant of 2^32.
 *  The resulting value is stored in \a td.
 *
 *  For more information see: <a href="http://tools.ietf.org/html/rfc1321">The MD5 Message-Digest Algorithm</a>
 *
 *  @param[in,out] td The current value of the digest stored as an uint4.
 *  @param[in] i  The current iteration of the algorithm.  This affects the values in \a data.
 *  @param[in] Hr The current rotation order.
 *  @param[in] p The constant 2^32.
 *  @param[in] data The starting input to MD5.  Padded from setupInput().
 *
 *  @see H()
 *  @see swizzleShift()
 *  @see leftRotate()
 *  @see setupInput()
**/
__device__ void HH(uint4 * td, int i, uint4 * Hr, float p, unsigned int * data)
{
	unsigned int Ft = H(td->y, td->z, td->w);
	i = (3 * i + 5) % 16;
	unsigned int r = Hr->x;
	swizzleShift(Hr);
	
	float t = sin(__int_as_float(i)) * p;
	unsigned int trigFunc = __float2uint_rd(t);
	td->x = td->y + leftRotate(td->x + Ft + trigFunc + data[i], r);
	swizzleShift(td);
}

/**
 * @brief The II scrambling function
 *
 *  The II function in the MD5 technical memo is a wrapper for the I scrambling function
 *  as well as performing its own rotations using LeftRotate() and swizzleShift().  The variable
 *  \a td is the current scrambled digest which is passed along and scrambled using the current
 *  iteration \a i, the rotation information \a Ir, and the starting input \a data.  \a p is kept as a
 *  constant of 2^32.
 *  The resulting value is stored in \a td.
 *
 *  For more information see: <a href="http://tools.ietf.org/html/rfc1321">The MD5 Message-Digest Algorithm</a>
 *
 *  @param[in,out] td The current value of the digest stored as an uint4.
 *  @param[in] i  The current iteration of the algorithm.  This affects the values in \a data.
 *  @param[in] Ir The current rotation order.
 *  @param[in] p The constant 2^32.
 *  @param[in] data The starting input to MD5.  Padded from setupInput().
 *
 *  @see I()
 *  @see swizzleShift()
 *  @see leftRotate()
 *  @see setupInput()
**/
__device__ void II(uint4 * td, int i, uint4 * Ir, float p, unsigned int * data)
{
	unsigned int Ft = G(td->y, td->z, td->w);
	i = (7 * i) % 16;
	unsigned int r = Ir->x;
	swizzleShift(Ir);
	
	float t = sin(__int_as_float(i)) * p;
	unsigned int trigFunc = __float2uint_rd(t);
	td->x = td->y + leftRotate(td->x + Ft + trigFunc + data[i], r);
	swizzleShift(td);
}

/**
 * @brief Sets up the \a input array using information of \a seed, and \a threadIdx
 *
 *  This function sets up the \a input array using a combination of the current thread's id and the
 *  user supplied \a seed.
 *
 *  For more information see: <a href="http://tools.ietf.org/html/rfc1321">The MD5 Message-Digest Algorithm</a>
 *
 *  @param[out] input The array which will contain the initial values for all the scrambling functions.
 *  @param[in] seed The user supplied seed as an unsigned int.
 *
 *  @see FF()
 *  @see GG()
 *  @see HH()
 *  @see II()
 *  @see gen_randMD5()
**/

extern "C"
__device__ void setupInput(unsigned int * input, const unsigned int seed)
{
	//loop unroll, also do this more intelligently
	input[0] = threadIdx.x ^ seed;
	input[1] = threadIdx.y ^ seed;
	input[2] = threadIdx.z ^ seed;
	input[3] = 0x80000000 ^ seed;
	input[4] = blockIdx.x ^ seed;
	input[5] = seed;
	input[6] = seed;
	input[7] = blockDim.x ^ seed;
	input[8] = seed;
	input[9] = seed;
	input[10] = seed;
	input[11] = seed;
	input[12] = seed;
	input[13] = seed;
	input[14] = seed;
	input[15] = 128 ^ seed;
}


//-------------------END MD5 FUNCTIONS--------------------------------------

/** @} */ // end rand functions
/** @} */ // end cudpp_cta



// -------------------------------------------------------------
// cuDPP -- CUDA Data Parallel Primitives library
// -------------------------------------------------------------
//  $Revision: 4400 $
//  $Date: 2008-08-04 10:58:14 -0700 (Mon, 04 Aug 2008) $
// -------------------------------------------------------------
// This source code is distributed under the terms of license.txt
// in the root directory of this source distribution.
// -------------------------------------------------------------

/**
 * @file
 * rand_kernel.cu
 *
 * @brief CUDPP kernel-level rand routines
 */

/** \addtogroup cudpp_kernel
  * @{
  */
/** @name Rand Functions
 * @{
 */

/**
 * @brief The main MD5 generation algorithm.
 *
 * This function runs the MD5 hashing random number generator.  It generates
 * MD5 hashes, and uses the output as randomized bits.  To repeatedly call this
 * function, always call cudppRandSeed() first to set a new seed or else the output
 * may be the same due to the deterministic nature of hashes.  gen_randMD5 generates
 * 128 random bits per thread.  Therefore, the parameter \a d_out is expected to be
 * an array of type uint4 with \a numElements indicies.
 *
 * @param[out] d_out the output array of type uint4.
 * @param[in] numElements the number of elements in \a d_out
 * @param[in] seed the random seed used to vary the output
 *
 * @see launchRandMD5Kernel()
 */

__device__ void gen_randMD5_per_element(uint4 *d_out, unsigned int* data)
{
	unsigned int h0 = 0x67452301;
	unsigned int h1 = 0xEFCDAB89;
	unsigned int h2 = 0x98BADCFE;
	unsigned int h3 = 0x10325476;
	
	uint4 result = make_uint4(h0, h1, h2, h3);
	uint4 td = result;
	
	float p = pow(2.0, 32.0);
	
	uint4 Fr = make_uint4(7, 12, 17, 22);
	uint4 Gr = make_uint4(5, 9, 14, 20);
	uint4 Hr = make_uint4(4, 11, 16, 23);
	uint4 Ir = make_uint4(6, 10, 15, 21);
	
	//for optimization, this is loop unrolled
	FF(&td, 0, &Fr, p, data);
	FF(&td, 1, &Fr, p, data);
	FF(&td, 2, &Fr, p, data);
	FF(&td, 3, &Fr, p, data);
	FF(&td, 4, &Fr, p, data);
	FF(&td, 5, &Fr, p, data);
	FF(&td, 6, &Fr, p, data);
	FF(&td, 7, &Fr, p, data);
	FF(&td, 8, &Fr, p, data);
	FF(&td, 9, &Fr, p, data);
	FF(&td, 10, &Fr, p, data);
	FF(&td, 11, &Fr, p, data);
	FF(&td, 12, &Fr, p, data);
	FF(&td, 13, &Fr, p, data);
	FF(&td, 14, &Fr, p, data);
	FF(&td, 15, &Fr, p, data);
	
	GG(&td, 16, &Gr, p, data);
	GG(&td, 17, &Gr, p, data);
	GG(&td, 18, &Gr, p, data);
	GG(&td, 19, &Gr, p, data);
	GG(&td, 20, &Gr, p, data);
	GG(&td, 21, &Gr, p, data);
	GG(&td, 22, &Gr, p, data);
	GG(&td, 23, &Gr, p, data);
	GG(&td, 24, &Gr, p, data);
	GG(&td, 25, &Gr, p, data);
	GG(&td, 26, &Gr, p, data);
	GG(&td, 27, &Gr, p, data);
	GG(&td, 28, &Gr, p, data);
	GG(&td, 29, &Gr, p, data);
	GG(&td, 30, &Gr, p, data);
	GG(&td, 31, &Gr, p, data);
	
	HH(&td, 32, &Hr, p, data);
	HH(&td, 33, &Hr, p, data);
	HH(&td, 34, &Hr, p, data);
	HH(&td, 35, &Hr, p, data);
	HH(&td, 36, &Hr, p, data);
	HH(&td, 37, &Hr, p, data);
	HH(&td, 38, &Hr, p, data);
	HH(&td, 39, &Hr, p, data);
	HH(&td, 40, &Hr, p, data);
	HH(&td, 41, &Hr, p, data);
	HH(&td, 42, &Hr, p, data);
	HH(&td, 43, &Hr, p, data);
	HH(&td, 44, &Hr, p, data);
	HH(&td, 45, &Hr, p, data);
	HH(&td, 46, &Hr, p, data);
	HH(&td, 47, &Hr, p, data);
	
	II(&td, 48, &Ir, p, data);
	II(&td, 49, &Ir, p, data);
	II(&td, 50, &Ir, p, data);
	II(&td, 51, &Ir, p, data);
	II(&td, 52, &Ir, p, data);
	II(&td, 53, &Ir, p, data);
	II(&td, 54, &Ir, p, data);
	II(&td, 55, &Ir, p, data);
	II(&td, 56, &Ir, p, data);
	II(&td, 57, &Ir, p, data);
	II(&td, 58, &Ir, p, data);
	II(&td, 59, &Ir, p, data);
	II(&td, 60, &Ir, p, data);
	II(&td, 61, &Ir, p, data);
	II(&td, 62, &Ir, p, data);
	II(&td, 63, &Ir, p, data);
	/*    */
	result.x = result.x + td.x;
	result.y = result.y + td.y;
	result.z = result.z + td.z;
	result.w = result.w + td.w;
	
	__syncthreads();
	
	d_out->x = result.x;
	d_out->y = result.y;
	d_out->z = result.z;
	d_out->w = result.w;
}

__global__ void gen_randMD5(uint4 *d_out, size_t numElements, unsigned int seed)
{
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	
	if(idx < numElements)
	{
		unsigned int data[16];
		setupInput(data, seed);
		uint4 *d_out_ = d_out + idx;
		gen_randMD5_per_element(d_out_, data);
	}
}

__global__ void gen_randMD5_float(float4 *d_out, size_t numElements, unsigned int seed, float2* d_upper_lower = NULL)
{
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx < numElements)
	{
		unsigned int data[16];
		setupInput(data, seed);
		
		uint4 d_out_int;
		gen_randMD5_per_element(&d_out_int, data);
		float p = pow(2.0, 32.0);
		
		if(d_upper_lower)
		{
			float interval = (*d_upper_lower).x - (*d_upper_lower).y;
			float lower_b = (*d_upper_lower).y;
			d_out[idx].x = d_out_int.x / p * interval + lower_b;
			d_out[idx].y = d_out_int.y / p * interval + lower_b;
			d_out[idx].z = d_out_int.z / p * interval + lower_b;
			d_out[idx].w = d_out_int.w / p * interval + lower_b;
		}
		else
		{
			d_out[idx].x = d_out_int.x / p;
			d_out[idx].y = d_out_int.y / p;
			d_out[idx].z = d_out_int.z / p;
			d_out[idx].w = d_out_int.w / p;
		}
	}
}
/** @} */ // end rand functions
/** @} */ // end cudpp_kernel

void launchRandMD5Kernel(unsigned int * d_out, unsigned int seed,
                         size_t numElements)
{
	//first, we need a temporary array of uints
	uint4 * dev_output;
	
	//figure out how many elements are needed in this array
	unsigned int devOutputsize = numElements / 4;
	devOutputsize += (numElements % 4 == 0) ? 0 : 1; //used for overflow
	unsigned int memSize = devOutputsize * sizeof(uint4);
	
	
	//now figure out block size
	unsigned int blockSize = RAND_CTA_SIZE;
	if(devOutputsize < RAND_CTA_SIZE) blockSize = devOutputsize;
	
	unsigned int n_blocks =
	    devOutputsize / blockSize + (devOutputsize % blockSize == 0 ? 0 : 1);
	    
	//now create the memory on the device
	GPUMALLOC((void **) &dev_output, memSize);
	GPUMEMSET(dev_output, 0, memSize);
	gen_randMD5 <<< n_blocks, blockSize>>>(dev_output, devOutputsize, seed);
	
	//here the GPU computation is done
	//here we have all the data on the device, we copy it over into host memory
	
	
	//calculate final memSize
	//@TODO: write a template version of this which calls two different version
	// depending if numElements %4 == 0
	size_t finalMemSize = sizeof(unsigned int) * numElements;
	GPUTOGPU(d_out, dev_output, finalMemSize);
	GPUFREE(dev_output);
}

extern "C"
void launchRandMD5Kernel_float(float *d_out, unsigned int seed,
                               size_t numElements, float2* d_upper_lower = NULL)
{
	//first, we need a temporary array of uints
	float4 * dev_output;
	
	//figure out how many elements are needed in this array
	unsigned int devOutputsize = numElements / 4;
	devOutputsize += (numElements % 4 == 0) ? 0 : 1; //used for overflow
	unsigned int memSize = devOutputsize * sizeof(float4);
	
	
	//now figure out block size
	unsigned int blockSize = RAND_CTA_SIZE;
	if(devOutputsize < RAND_CTA_SIZE) blockSize = devOutputsize;
	
	unsigned int n_blocks =
	    devOutputsize / blockSize + (devOutputsize % blockSize == 0 ? 0 : 1);
	    
	    
	//now create the memory on the device
	GPUMALLOC((void **) &dev_output, memSize);
	GPUMEMSET(dev_output, 0, memSize);
	gen_randMD5_float <<< n_blocks, blockSize>>>(dev_output, devOutputsize, seed, d_upper_lower);
	
	//here the GPU computation is done
	//here we have all the data on the device, we copy it over into host memory
	
	
	//calculate final memSize
	//@TODO: write a template version of this which calls two different version
	// depending if numElements %4 == 0
	size_t finalMemSize = sizeof(float) * numElements;
	GPUTOGPU(d_out, dev_output, finalMemSize);
	GPUFREE(dev_output));
}


extern "C"
void generate_uniform_MD5_float(int nRand, int seed,  float upper_bound, float lower_bound)
{
	float* phOutput = new float[nRand];
	float* pdOutput;
	float2 upper_lower = make_float2(upper_bound, lower_bound);
	float2* pdBound;
	
	GPUMALLOC((void**)&pdOutput, sizeof(float) * nRand);
	GPUMALLOC((void**)&pdBound, sizeof(float2));
	GPUMEMSET(pdOutput, 0, sizeof(float) * nRand);
	
	TOGPU(pdBound, &upper_lower, sizeof(float2));
	
	launchRandMD5Kernel_float(pdOutput, seed, nRand, pdBound);
	
	FROMGPU(phOutput, pdOutput, sizeof(float) * nRand);
	
	for(int i = 0; i < nRand; ++i)
	{
		printf("%f ", phOutput[i]);
	}
	printf("\n");
	
	GPUFREE(pdOutput);
	
	delete [] phOutput;
	phOutput = NULL;
}

extern "C"
void generate_uniform_MD5(int nRand, int seed)
{
	unsigned int* phOutput = new unsigned int[nRand];
	unsigned int* pdOutput;
	
	GPUMALLOC((void**)&pdOutput, sizeof(unsigned int) * nRand);
	GPUMEMSET(pdOutput, 0, sizeof(unsigned int) * nRand);
	
	launchRandMD5Kernel(pdOutput, seed, nRand);
	
	FROMGPU(phOutput, pdOutput, sizeof(unsigned int) * nRand);
	
	for(int i = 0; i < nRand; ++i)
	{
		printf("%d ", phOutput[i]);
	}
	printf("\n");
	
	GPUFREE(pdOutput);
	
	delete [] phOutput;
	phOutput = NULL;
}


#endif