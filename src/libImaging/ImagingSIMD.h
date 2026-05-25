#ifndef __IMAGING_SIMD_H__
#define __IMAGING_SIMD_H__

/* Microsoft compiler doesn't limit intrinsics for an architecture.
   This macro is set only on x86 and means SSE2 and above including AVX2. */
#if defined(_M_X64) || _M_IX86_FP == 2
    #define __SSE4_2__
#endif

#if defined(__SSE4_2__)
    #include <emmintrin.h>
    #include <mmintrin.h>
    #include <smmintrin.h>
#endif

#if defined(__AVX2__)
    #include <immintrin.h>
#endif

#if defined(__riscv_vector)
    #include <riscv_vector.h>

    /*
     * RVV intrinsic API compatibility between GCC 13 (v0.11) and GCC 14+ (v1.0).
     *
     * GCC 13: vnclipu takes 3 args (src, shift, vl), no VXRM parameter.
     * GCC 14+: vnclipu takes 4 args (src, shift, vxrm, vl).
     *
     * GCC 13: vlseg/vsseg use pointer-based API.
     * GCC 14+: vlseg/vsseg use tuple-based API (vuint8m1x4_t etc).
     */
    /*
     * Detect vnclipu API version:
     * - GCC 13 (rvv intrinsic v0.11): 3 args (src, shift, vl)
     * - GCC 14+/15 (rvv intrinsic v0.12+): 4 args (src, shift, vxrm, vl)
     * - Clang 17+: 4 args with __RISCV_VXRM_RDN enum
     *
     * We detect by checking __riscv_v_intrinsic version or GCC version.
     * RDN (round-down/truncate) = 2 as integer constant.
     */
    #if (defined(__GNUC__) && !defined(__clang__) && __GNUC__ <= 13)
        /* GCC 13 and earlier: 3-argument vnclipu (no VXRM) */
        #define RVV_VNCLIPU(type, src, shift, vl) \
            __riscv_vnclipu_wx_##type((src), (shift), (vl))
    #elif defined(__RISCV_VXRM_RDN)
        /* Clang 17+: uses enum constant */
        #define RVV_VNCLIPU(type, src, shift, vl) \
            __riscv_vnclipu_wx_##type((src), (shift), __RISCV_VXRM_RDN, (vl))
    #else
        /* GCC 14+/15: 4-argument with integer constant (2 = RDN) */
        #define RVV_VNCLIPU(type, src, shift, vl) \
            __riscv_vnclipu_wx_##type((src), (shift), 2, (vl))
    #endif

    /*
     * Segment load/store compatibility.
     * GCC 14+: tuple types (vuint8m1x4_t), __riscv_vlseg4e8_v_u8m1x4 etc.
     * GCC 13: pointer-based (__riscv_vlseg4e8_v_u8m1(&r, &g, &b, &a, ptr, vl)).
     */
    #ifdef __riscv_v_intrinsic
        /* GCC 14+ / Clang: has __riscv_v_intrinsic >= 12000 for tuple types */
        #if __riscv_v_intrinsic >= 12000
            #define RVV_HAS_TUPLE_API 1
        #endif
    #endif

    /* Fallback: detect tuple types by checking if GCC >= 14 or Clang >= 17 */
    #ifndef RVV_HAS_TUPLE_API
        #if (defined(__GNUC__) && __GNUC__ >= 14) || \
            (defined(__clang__) && __clang_major__ >= 17)
            #define RVV_HAS_TUPLE_API 1
        #else
            #define RVV_HAS_TUPLE_API 0
        #endif
    #endif

    /*
     * vlseg4e8 / vsseg4e8 wrappers
     */
    #if RVV_HAS_TUPLE_API
        #define RVV_VLSEG4E8_U8M1(r, g, b, a, ptr, vl) do { \
            vuint8m1x4_t _seg = __riscv_vlseg4e8_v_u8m1x4((const uint8_t*)(ptr), (vl)); \
            (r) = __riscv_vget_v_u8m1x4_u8m1(_seg, 0); \
            (g) = __riscv_vget_v_u8m1x4_u8m1(_seg, 1); \
            (b) = __riscv_vget_v_u8m1x4_u8m1(_seg, 2); \
            (a) = __riscv_vget_v_u8m1x4_u8m1(_seg, 3); \
        } while(0)

        #define RVV_VSSEG4E8_U8M1(ptr, r, g, b, a, vl) do { \
            vuint8m1x4_t _oseg = __riscv_vcreate_v_u8m1x4((r), (g), (b), (a)); \
            __riscv_vsseg4e8_v_u8m1x4((uint8_t*)(ptr), _oseg, (vl)); \
        } while(0)

        #define RVV_VLSEG2E32_U32M2(a, b, ptr, vl) do { \
            vuint32m2x2_t _s = __riscv_vlseg2e32_v_u32m2x2((const uint32_t*)(ptr), (vl)); \
            (a) = __riscv_vget_v_u32m2x2_u32m2(_s, 0); \
            (b) = __riscv_vget_v_u32m2x2_u32m2(_s, 1); \
        } while(0)

        #define RVV_VLSEG3E32_U32M2(a, b, c, ptr, vl) do { \
            vuint32m2x3_t _s = __riscv_vlseg3e32_v_u32m2x3((const uint32_t*)(ptr), (vl)); \
            (a) = __riscv_vget_v_u32m2x3_u32m2(_s, 0); \
            (b) = __riscv_vget_v_u32m2x3_u32m2(_s, 1); \
            (c) = __riscv_vget_v_u32m2x3_u32m2(_s, 2); \
        } while(0)

        #define RVV_VLSEG3E32_U32M1(a, b, c, ptr, vl) do { \
            vuint32m1x3_t _s = __riscv_vlseg3e32_v_u32m1x3((const uint32_t*)(ptr), (vl)); \
            (a) = __riscv_vget_v_u32m1x3_u32m1(_s, 0); \
            (b) = __riscv_vget_v_u32m1x3_u32m1(_s, 1); \
            (c) = __riscv_vget_v_u32m1x3_u32m1(_s, 2); \
        } while(0)
    #else
        /* GCC 13: pointer-based API */
        #define RVV_VLSEG4E8_U8M1(r, g, b, a, ptr, vl) \
            __riscv_vlseg4e8_v_u8m1(&(r), &(g), &(b), &(a), (const uint8_t*)(ptr), (vl))

        #define RVV_VSSEG4E8_U8M1(ptr, r, g, b, a, vl) \
            __riscv_vsseg4e8_v_u8m1((uint8_t*)(ptr), (r), (g), (b), (a), (vl))

        #define RVV_VLSEG2E32_U32M2(a, b, ptr, vl) \
            __riscv_vlseg2e32_v_u32m2(&(a), &(b), (const uint32_t*)(ptr), (vl))

        #define RVV_VLSEG3E32_U32M2(a, b, c, ptr, vl) \
            __riscv_vlseg3e32_v_u32m2(&(a), &(b), &(c), (const uint32_t*)(ptr), (vl))

        #define RVV_VLSEG3E32_U32M1(a, b, c, ptr, vl) \
            __riscv_vlseg3e32_v_u32m1(&(a), &(b), &(c), (const uint32_t*)(ptr), (vl))
    #endif

#endif /* __riscv_vector */

#if defined(__SSE4_2__)
static __m128i inline
mm_cvtepu8_epi32(void *ptr) {
    return _mm_cvtepu8_epi32(_mm_cvtsi32_si128(*(INT32 *) ptr));
}
#endif

#if defined(__AVX2__)
static __m256i inline
mm256_cvtepu8_epi32(void *ptr) {
    return _mm256_cvtepu8_epi32(_mm_loadl_epi64((__m128i *) ptr));
}
#endif

/*
 * RVV helpers
 */
#if defined(__riscv_vector)

/* Load 4 x uint8 and zero-extend to 4 x uint32 */
static inline vuint32m1_t
rvv_cvtepu8_epi32(const void *ptr) {
    vuint8mf4_t v8 = __riscv_vle8_v_u8mf4((const uint8_t *)ptr, 4);
    return __riscv_vzext_vf4_u32m1(v8, 4);
}

/* Pack 4 x uint32 -> 4 x uint8 with saturation, store as UINT32 */
static inline void
rvv_pack_u32_to_u8x4(void *dst, vuint32m1_t v) {
    vuint16mf2_t v16 = RVV_VNCLIPU(u16mf2, v, 0, 4);
    vuint8mf4_t v8 = RVV_VNCLIPU(u8mf4, v16, 0, 4);
    __riscv_vse8_v_u8mf4((uint8_t *)dst, v8, 4);
}

/* Pack 4 x int32 -> 4 x uint8 with clamping to [0,255], store as UINT32 */
static inline void
rvv_pack_i32_to_u8x4(void *dst, vint32m1_t v) {
    v = __riscv_vmax_vx_i32m1(v, 0, 4);
    v = __riscv_vmin_vx_i32m1(v, 255, 4);
    vuint32m1_t vu = __riscv_vreinterpret_v_i32m1_u32m1(v);
    rvv_pack_u32_to_u8x4(dst, vu);
}

/* Load 8 x uint8 and zero-extend to 8 x uint16 */
static inline vuint16m1_t
rvv_cvtepu8_epu16(const void *ptr) {
    vuint8mf2_t v8 = __riscv_vle8_v_u8mf2((const uint8_t *)ptr, 8);
    return __riscv_vzext_vf2_u16m1(v8, 8);
}

#endif /* __riscv_vector */

#endif /* __IMAGING_SIMD_H__ */
