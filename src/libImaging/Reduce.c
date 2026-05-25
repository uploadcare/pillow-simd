#include "Imaging.h"
#include "ImagingSIMD.h"

#include <math.h>

#define ROUND_UP(f) ((int)((f) >= 0.0 ? (f) + 0.5F : (f)-0.5F))

UINT32
division_UINT32(int divider, int result_bits) {
    UINT32 max_dividend = (1 << result_bits) * divider;
    float max_int = (1 << 30) * 4.0;
    return (UINT32)(max_int / max_dividend);
}

/* ----------------------------------------------------------------
 * Scalar helpers used when neither SSE4 nor RVV is available, or
 * as a reference fallback.
 * ---------------------------------------------------------------- */

/* Pack 4 UINT32 channel values (already in [0,255]) into one UINT32 pixel. */
static inline UINT32
scalar_pack_u32(UINT32 r, UINT32 g, UINT32 b, UINT32 a) {
    return (a << 24) | (b << 16) | (g << 8) | r;
}

/* ----------------------------------------------------------------
 * Two-pixel load helpers for RVV.
 * Load 8 bytes (2 adjacent RGBA pixels), zero-extend to u16,
 * then widen the low 4 elements and the high 4 elements to u32.
 * ---------------------------------------------------------------- */
#if defined(__riscv_vector)
static inline void
rvv_load_2px(void *ptr, vuint32m1_t *lo, vuint32m1_t *hi) {
    vuint8mf2_t v8 = __riscv_vle8_v_u8mf2((const uint8_t *)ptr, 8);
    vuint16m1_t v16 = __riscv_vzext_vf2_u16m1(v8, 8);
    /* low 4 elements -> u32 */
    vuint16mf2_t v16_lo = __riscv_vlmul_trunc_v_u16m1_u16mf2(v16);
    *lo = __riscv_vzext_vf2_u32m1(v16_lo, 4);
    /* high 4 elements -> u32: slidedown by 4 then truncate */
    vuint16m1_t v16_hi_full = __riscv_vslidedown_vx_u16m1(v16, 4, 8);
    vuint16mf2_t v16_hi = __riscv_vlmul_trunc_v_u16m1_u16mf2(v16_hi_full);
    *hi = __riscv_vzext_vf2_u32m1(v16_hi, 4);
}

/* Load 8 bytes (2 RGBA pixels), widen to 8xu16, add two rows, then
 * split into low/high u32 halves.  Used in 2-pixel-per-iteration loops.
 */
static inline vuint16m1_t
rvv_load_2px_u16(void *ptr) {
    vuint8mf2_t v8 = __riscv_vle8_v_u8mf2((const uint8_t *)ptr, 8);
    return __riscv_vzext_vf2_u16m1(v8, 8);
}

/* Split a u16m1 (8 elements) into two u32m1 (4 elements each): lo and hi. */
static inline void
rvv_u16_to_2xu32(vuint16m1_t v16, vuint32m1_t *lo, vuint32m1_t *hi) {
    vuint16mf2_t v16_lo = __riscv_vlmul_trunc_v_u16m1_u16mf2(v16);
    *lo = __riscv_vzext_vf2_u32m1(v16_lo, 4);
    vuint16m1_t v16_hi_full = __riscv_vslidedown_vx_u16m1(v16, 4, 8);
    vuint16mf2_t v16_hi = __riscv_vlmul_trunc_v_u16m1_u16mf2(v16_hi_full);
    *hi = __riscv_vzext_vf2_u32m1(v16_hi, 4);
}
#endif

void
ImagingReduceNxN(Imaging imOut, Imaging imIn, int box[4], int xscale, int yscale) {
    /* The most general implementation for any xscale and yscale
     */
    int x, y, xx, yy;
    UINT32 multiplier = division_UINT32(yscale * xscale, 8);
    UINT32 amend = yscale * xscale / 2;

    if (imIn->image8) {
        for (y = 0; y < box[3] / yscale; y++) {
            int yy_from = box[1] + y * yscale;
            for (x = 0; x < box[2] / xscale; x++) {
                int xx_from = box[0] + x * xscale;
                UINT32 ss = amend;
                for (yy = yy_from; yy < yy_from + yscale - 1; yy += 2) {
                    UINT8 *line0 = (UINT8 *)imIn->image8[yy];
                    UINT8 *line1 = (UINT8 *)imIn->image8[yy + 1];
                    for (xx = xx_from; xx < xx_from + xscale - 1; xx += 2) {
                        ss += line0[xx + 0] + line0[xx + 1] + line1[xx + 0] +
                              line1[xx + 1];
                    }
                    if (xscale & 0x01) {
                        ss += line0[xx + 0] + line1[xx + 0];
                    }
                }
                if (yscale & 0x01) {
                    UINT8 *line = (UINT8 *)imIn->image8[yy];
                    for (xx = xx_from; xx < xx_from + xscale - 1; xx += 2) {
                        ss += line[xx + 0] + line[xx + 1];
                    }
                    if (xscale & 0x01) {
                        ss += line[xx + 0];
                    }
                }
                imOut->image8[y][x] = (ss * multiplier) >> 24;
            }
        }
    } else {
#if defined(__SSE4_2__)
        __m128i mm_multiplier = _mm_set1_epi32(multiplier);
        __m128i mm_amend = _mm_set1_epi32(amend);
        for (y = 0; y < box[3] / yscale; y++) {
            int yy_from = box[1] + y*yscale;
            for (x = 0; x < box[2] / xscale; x++) {
                int xx_from = box[0] + x*xscale;
                __m128i mm_ss0 = mm_amend;
                for (yy = yy_from; yy < yy_from + yscale - 1; yy += 2) {
                    UINT8 *line0 = (UINT8 *)imIn->image[yy];
                    UINT8 *line1 = (UINT8 *)imIn->image[yy + 1];
                    for (xx = xx_from; xx < xx_from + xscale - 1; xx += 2) {
                        __m128i pix0 = _mm_unpacklo_epi8(
                            _mm_loadl_epi64((__m128i *) &line0[xx*4]),
                            _mm_setzero_si128());
                        __m128i pix1 = _mm_unpacklo_epi8(
                            _mm_loadl_epi64((__m128i *) &line1[xx*4]),
                            _mm_setzero_si128());
                        pix0 = _mm_add_epi16(pix0, pix1);
                        pix0 = _mm_add_epi32(
                            _mm_unpackhi_epi16(pix0, _mm_setzero_si128()),
                            _mm_unpacklo_epi16(pix0, _mm_setzero_si128()));
                        mm_ss0 = _mm_add_epi32(mm_ss0, pix0);
                    }
                    if (xscale & 0x01) {
                        __m128i pix0 = mm_cvtepu8_epi32(&line0[xx*4]);
                        __m128i pix1 = mm_cvtepu8_epi32(&line1[xx*4]);
                        mm_ss0 = _mm_add_epi32(mm_ss0, pix0);
                        mm_ss0 = _mm_add_epi32(mm_ss0, pix1);
                    }
                }
                if (yscale & 0x01) {
                    UINT8 *line = (UINT8 *)imIn->image[yy];
                    for (xx = xx_from; xx < xx_from + xscale - 1; xx += 2) {
                        __m128i pix0 = _mm_unpacklo_epi8(
                            _mm_loadl_epi64((__m128i *) &line[xx*4]),
                            _mm_setzero_si128());
                        pix0 = _mm_add_epi32(
                            _mm_unpackhi_epi16(pix0, _mm_setzero_si128()),
                            _mm_unpacklo_epi16(pix0, _mm_setzero_si128()));
                        mm_ss0 = _mm_add_epi32(mm_ss0, pix0);
                    }
                    if (xscale & 0x01) {
                        __m128i pix0 = mm_cvtepu8_epi32(&line[xx*4]);
                        mm_ss0 = _mm_add_epi32(mm_ss0, pix0);
                    }
                }

                mm_ss0 = _mm_mullo_epi32(mm_ss0, mm_multiplier);
                mm_ss0 = _mm_srli_epi32(mm_ss0, 24);
                mm_ss0 = _mm_packs_epi32(mm_ss0, mm_ss0);
                ((UINT32 *)imOut->image[y])[x] = _mm_cvtsi128_si32(_mm_packus_epi16(mm_ss0, mm_ss0));
            }
        }
#elif defined(__riscv_vector)
        vuint32m1_t rv_multiplier = __riscv_vmv_v_x_u32m1(multiplier, 4);
        vuint32m1_t rv_amend = __riscv_vmv_v_x_u32m1(amend, 4);
        for (y = 0; y < box[3] / yscale; y++) {
            int yy_from = box[1] + y*yscale;
            for (x = 0; x < box[2] / xscale; x++) {
                int xx_from = box[0] + x*xscale;
                vuint32m1_t rv_ss0 = rv_amend;
                for (yy = yy_from; yy < yy_from + yscale - 1; yy += 2) {
                    UINT8 *line0 = (UINT8 *)imIn->image[yy];
                    UINT8 *line1 = (UINT8 *)imIn->image[yy + 1];
                    for (xx = xx_from; xx < xx_from + xscale - 1; xx += 2) {
                        vuint32m1_t lo0, hi0, lo1, hi1;
                        rvv_load_2px(&line0[xx*4], &lo0, &hi0);
                        rvv_load_2px(&line1[xx*4], &lo1, &hi1);
                        vuint32m1_t sum01 = __riscv_vadd_vv_u32m1(lo0, hi0, 4);
                        vuint32m1_t sum23 = __riscv_vadd_vv_u32m1(lo1, hi1, 4);
                        rv_ss0 = __riscv_vadd_vv_u32m1(rv_ss0,
                            __riscv_vadd_vv_u32m1(sum01, sum23, 4), 4);
                    }
                    if (xscale & 0x01) {
                        vuint32m1_t pix0 = rvv_cvtepu8_epi32(&line0[xx*4]);
                        vuint32m1_t pix1 = rvv_cvtepu8_epi32(&line1[xx*4]);
                        rv_ss0 = __riscv_vadd_vv_u32m1(rv_ss0, pix0, 4);
                        rv_ss0 = __riscv_vadd_vv_u32m1(rv_ss0, pix1, 4);
                    }
                }
                if (yscale & 0x01) {
                    UINT8 *line = (UINT8 *)imIn->image[yy];
                    for (xx = xx_from; xx < xx_from + xscale - 1; xx += 2) {
                        vuint32m1_t lo0, hi0;
                        rvv_load_2px(&line[xx*4], &lo0, &hi0);
                        rv_ss0 = __riscv_vadd_vv_u32m1(rv_ss0,
                            __riscv_vadd_vv_u32m1(lo0, hi0, 4), 4);
                    }
                    if (xscale & 0x01) {
                        vuint32m1_t pix0 = rvv_cvtepu8_epi32(&line[xx*4]);
                        rv_ss0 = __riscv_vadd_vv_u32m1(rv_ss0, pix0, 4);
                    }
                }

                rv_ss0 = __riscv_vmul_vv_u32m1(rv_ss0, rv_multiplier, 4);
                rv_ss0 = __riscv_vsrl_vx_u32m1(rv_ss0, 24, 4);
                rvv_pack_u32_to_u8x4(&((UINT32 *)imOut->image[y])[x], rv_ss0);
            }
        }
#else
        /* Scalar fallback */
        for (y = 0; y < box[3] / yscale; y++) {
            int yy_from = box[1] + y*yscale;
            for (x = 0; x < box[2] / xscale; x++) {
                int xx_from = box[0] + x*xscale;
                UINT32 ss0 = amend, ss1 = amend, ss2 = amend, ss3 = amend;
                for (yy = yy_from; yy < yy_from + yscale; yy++) {
                    UINT8 *line = (UINT8 *)imIn->image[yy];
                    for (xx = xx_from; xx < xx_from + xscale; xx++) {
                        ss0 += line[xx*4 + 0];
                        ss1 += line[xx*4 + 1];
                        ss2 += line[xx*4 + 2];
                        ss3 += line[xx*4 + 3];
                    }
                }
                ss0 = (ss0 * multiplier) >> 24;
                ss1 = (ss1 * multiplier) >> 24;
                ss2 = (ss2 * multiplier) >> 24;
                ss3 = (ss3 * multiplier) >> 24;
                ((UINT32 *)imOut->image[y])[x] = scalar_pack_u32(ss0, ss1, ss2, ss3);
            }
        }
#endif
    }
}

void
ImagingReduce1xN(Imaging imOut, Imaging imIn, int box[4], int yscale) {
    /* Optimized implementation for xscale = 1.
     */
    int x, y, yy;
    int xscale = 1;
    UINT32 multiplier = division_UINT32(yscale * xscale, 8);
    UINT32 amend = yscale * xscale / 2;

    if (imIn->image8) {
        for (y = 0; y < box[3] / yscale; y++) {
            int yy_from = box[1] + y * yscale;
            for (x = 0; x < box[2] / xscale; x++) {
                int xx = box[0] + x * xscale;
                UINT32 ss = amend;
                for (yy = yy_from; yy < yy_from + yscale - 1; yy += 2) {
                    UINT8 *line0 = (UINT8 *)imIn->image8[yy];
                    UINT8 *line1 = (UINT8 *)imIn->image8[yy + 1];
                    ss += line0[xx + 0] + line1[xx + 0];
                }
                if (yscale & 0x01) {
                    UINT8 *line = (UINT8 *)imIn->image8[yy];
                    ss += line[xx + 0];
                }
                imOut->image8[y][x] = (ss * multiplier) >> 24;
            }
        }
    } else {
#if defined(__SSE4_2__)
        __m128i mm_multiplier = _mm_set1_epi32(multiplier);
        __m128i mm_amend = _mm_set1_epi32(amend);
        for (y = 0; y < box[3] / yscale; y++) {
            int yy_from = box[1] + y*yscale;
            for (x = 0; x < box[2] / xscale - 1; x += 2) {
                int xx = box[0] + x*xscale;
                __m128i mm_ss0 = mm_amend;
                __m128i mm_ss1 = mm_amend;
                for (yy = yy_from; yy < yy_from + yscale - 1; yy += 2) {
                    UINT8 *line0 = (UINT8 *)imIn->image[yy + 0];
                    UINT8 *line1 = (UINT8 *)imIn->image[yy + 1];
                    __m128i pix0 = _mm_unpacklo_epi8(
                            _mm_loadl_epi64((__m128i *) &line0[xx*4]),
                            _mm_setzero_si128());
                    __m128i pix1 = _mm_unpacklo_epi8(
                            _mm_loadl_epi64((__m128i *) &line1[xx*4]),
                            _mm_setzero_si128());
                    pix0 = _mm_add_epi16(pix0, pix1);
                    mm_ss0 = _mm_add_epi32(mm_ss0, _mm_unpacklo_epi16(pix0, _mm_setzero_si128()));
                    mm_ss1 = _mm_add_epi32(mm_ss1, _mm_unpackhi_epi16(pix0, _mm_setzero_si128()));
                }
                if (yscale & 0x01) {
                    UINT8 *line = (UINT8 *)imIn->image[yy];
                    __m128i pix0 = _mm_unpacklo_epi8(
                            _mm_loadl_epi64((__m128i *) &line[xx*4]),
                            _mm_setzero_si128());
                    mm_ss0 = _mm_add_epi32(mm_ss0, _mm_unpacklo_epi16(pix0, _mm_setzero_si128()));
                    mm_ss1 = _mm_add_epi32(mm_ss1, _mm_unpackhi_epi16(pix0, _mm_setzero_si128()));
                }

                mm_ss0 = _mm_mullo_epi32(mm_ss0, mm_multiplier);
                mm_ss1 = _mm_mullo_epi32(mm_ss1, mm_multiplier);
                mm_ss0 = _mm_srli_epi32(mm_ss0, 24);
                mm_ss1 = _mm_srli_epi32(mm_ss1, 24);
                mm_ss0 = _mm_packs_epi32(mm_ss0, mm_ss1);
                _mm_storel_epi64((__m128i *)(imOut->image32[y] + x), _mm_packus_epi16(mm_ss0, mm_ss0));
            }
            for (; x < box[2] / xscale; x++) {
                int xx = box[0] + x*xscale;
                __m128i mm_ss0 = mm_amend;
                for (yy = yy_from; yy < yy_from + yscale - 1; yy += 2) {
                    UINT8 *line0 = (UINT8 *)imIn->image[yy + 0];
                    UINT8 *line1 = (UINT8 *)imIn->image[yy + 1];
                    __m128i pix0 = mm_cvtepu8_epi32(&line0[xx*4]);
                    __m128i pix1 = mm_cvtepu8_epi32(&line1[xx*4]);
                    mm_ss0 = _mm_add_epi32(mm_ss0, _mm_add_epi32(pix0, pix1));
                }
                if (yscale & 0x01) {
                    UINT8 *line = (UINT8 *)imIn->image[yy];
                    __m128i pix0 = mm_cvtepu8_epi32(&line[xx*4]);
                    mm_ss0 = _mm_add_epi32(mm_ss0, pix0);
                }

                mm_ss0 = _mm_mullo_epi32(mm_ss0, mm_multiplier);
                mm_ss0 = _mm_srli_epi32(mm_ss0, 24);
                mm_ss0 = _mm_packs_epi32(mm_ss0, mm_ss0);
                ((UINT32 *)imOut->image[y])[x] = _mm_cvtsi128_si32(_mm_packus_epi16(mm_ss0, mm_ss0));
            }
        }
#elif defined(__riscv_vector)
        /* VL-agnostic: xscale=1 so output pixels are contiguous.
         * Process multiple output pixels per iteration using u32m4 accumulators.
         */
        for (y = 0; y < box[3] / yscale; y++) {
            int yy_from = box[1] + y*yscale;
            int x_base = box[0];
            int x_count = box[2] / xscale;
            for (x = 0; x < x_count; ) {
                size_t n_bytes = (size_t)(x_count - x) * 4;
                size_t vl = __riscv_vsetvl_e32m4(n_bytes);

                vuint32m4_t rv_ss = __riscv_vmv_v_x_u32m4(amend, vl);

                for (yy = yy_from; yy < yy_from + yscale - 1; yy += 2) {
                    UINT8 *line0 = (UINT8 *)imIn->image[yy + 0] + (x_base + x) * 4;
                    UINT8 *line1 = (UINT8 *)imIn->image[yy + 1] + (x_base + x) * 4;

                    vuint8m1_t p0 = __riscv_vle8_v_u8m1(line0, vl);
                    vuint8m1_t p1 = __riscv_vle8_v_u8m1(line1, vl);
                    /* Widen and add pair of rows */
                    vuint16m2_t sum16 = __riscv_vwaddu_vv_u16m2(p0, p1, vl);
                    vuint32m4_t sum32 = __riscv_vzext_vf2_u32m4(sum16, vl);
                    rv_ss = __riscv_vadd_vv_u32m4(rv_ss, sum32, vl);
                }
                if (yscale & 0x01) {
                    UINT8 *line = (UINT8 *)imIn->image[yy] + (x_base + x) * 4;
                    vuint8m1_t p = __riscv_vle8_v_u8m1(line, vl);
                    vuint32m4_t pw = __riscv_vzext_vf4_u32m4(p, vl);
                    rv_ss = __riscv_vadd_vv_u32m4(rv_ss, pw, vl);
                }

                rv_ss = __riscv_vmul_vx_u32m4(rv_ss, multiplier, vl);
                rv_ss = __riscv_vsrl_vx_u32m4(rv_ss, 24, vl);

                vuint16m2_t n16 = RVV_VNCLIPU(u16m2, rv_ss, 0, vl);
                vuint8m1_t n8 = RVV_VNCLIPU(u8m1, n16, 0, vl);
                __riscv_vse8_v_u8m1((UINT8 *)&((UINT32 *)imOut->image[y])[x], n8, vl);
                x += vl / 4;
            }
        }
#else
        /* Scalar fallback */
        for (y = 0; y < box[3] / yscale; y++) {
            int yy_from = box[1] + y*yscale;
            for (x = 0; x < box[2] / xscale; x++) {
                int xx = box[0] + x*xscale;
                UINT32 ss0 = amend, ss1 = amend, ss2 = amend, ss3 = amend;
                for (yy = yy_from; yy < yy_from + yscale; yy++) {
                    UINT8 *line = (UINT8 *)imIn->image[yy];
                    ss0 += line[xx*4 + 0];
                    ss1 += line[xx*4 + 1];
                    ss2 += line[xx*4 + 2];
                    ss3 += line[xx*4 + 3];
                }
                ss0 = (ss0 * multiplier) >> 24;
                ss1 = (ss1 * multiplier) >> 24;
                ss2 = (ss2 * multiplier) >> 24;
                ss3 = (ss3 * multiplier) >> 24;
                ((UINT32 *)imOut->image[y])[x] = scalar_pack_u32(ss0, ss1, ss2, ss3);
            }
        }
#endif
    }
}

void
ImagingReduceNx1(Imaging imOut, Imaging imIn, int box[4], int xscale) {
    /* Optimized implementation for yscale = 1.
     */
    int x, y, xx;
    int yscale = 1;
    UINT32 multiplier = division_UINT32(yscale * xscale, 8);
    UINT32 amend = yscale * xscale / 2;

    if (imIn->image8) {
        for (y = 0; y < box[3] / yscale; y++) {
            int yy = box[1] + y*yscale;
            UINT8 *line = (UINT8 *)imIn->image8[yy];
            for (x = 0; x < box[2] / xscale; x++) {
                int xx_from = box[0] + x * xscale;
                UINT32 ss = amend;
                for (xx = xx_from; xx < xx_from + xscale - 1; xx += 2) {
                    ss += line[xx + 0] + line[xx + 1];
                }
                if (xscale & 0x01) {
                    ss += line[xx + 0];
                }
                imOut->image8[y][x] = (ss * multiplier) >> 24;
            }
        }
    } else {
#if defined(__SSE4_2__)
        __m128i mm_multiplier = _mm_set1_epi32(multiplier);
        __m128i mm_amend = _mm_set1_epi32(amend);
        for (y = 0; y < box[3] / yscale - 1; y += 2) {
            int yy = box[1] + y*yscale;
            UINT8 *line0 = (UINT8 *)imIn->image[yy + 0];
            UINT8 *line1 = (UINT8 *)imIn->image[yy + 1];
            for (x = 0; x < box[2] / xscale; x++) {
                int xx_from = box[0] + x*xscale;
                __m128i mm_ss0 = mm_amend;
                __m128i mm_ss1 = mm_amend;
                for (xx = xx_from; xx < xx_from + xscale - 1; xx += 2) {
                    __m128i pix0 = _mm_unpacklo_epi8(
                        _mm_loadl_epi64((__m128i *) &line0[xx*4]),
                        _mm_setzero_si128());
                    __m128i pix1 = _mm_unpacklo_epi8(
                        _mm_loadl_epi64((__m128i *) &line1[xx*4]),
                        _mm_setzero_si128());
                    pix0 = _mm_add_epi32(
                        _mm_unpacklo_epi16(pix0, _mm_setzero_si128()),
                        _mm_unpackhi_epi16(pix0, _mm_setzero_si128()));
                    pix1 = _mm_add_epi32(
                        _mm_unpacklo_epi16(pix1, _mm_setzero_si128()),
                        _mm_unpackhi_epi16(pix1, _mm_setzero_si128()));
                    mm_ss0 = _mm_add_epi32(mm_ss0, pix0);
                    mm_ss1 = _mm_add_epi32(mm_ss1, pix1);
                }
                if (xscale & 0x01) {
                    __m128i pix0 = mm_cvtepu8_epi32(&line0[xx*4]);
                    __m128i pix1 = mm_cvtepu8_epi32(&line1[xx*4]);
                    mm_ss0 = _mm_add_epi32(mm_ss0, pix0);
                    mm_ss1 = _mm_add_epi32(mm_ss1, pix1);
                }

                mm_ss0 = _mm_mullo_epi32(mm_ss0, mm_multiplier);
                mm_ss1 = _mm_mullo_epi32(mm_ss1, mm_multiplier);
                mm_ss0 = _mm_srli_epi32(mm_ss0, 24);
                mm_ss1 = _mm_srli_epi32(mm_ss1, 24);
                mm_ss0 = _mm_packs_epi32(mm_ss0, mm_ss0);
                mm_ss1 = _mm_packs_epi32(mm_ss1, mm_ss1);
                ((UINT32 *)imOut->image[y + 0])[x] = _mm_cvtsi128_si32(_mm_packus_epi16(mm_ss0, mm_ss0));
                ((UINT32 *)imOut->image[y + 1])[x] = _mm_cvtsi128_si32(_mm_packus_epi16(mm_ss1, mm_ss1));
            }
        }
        for (; y < box[3] / yscale; y++) {
            int yy = box[1] + y*yscale;
            UINT8 *line = (UINT8 *)imIn->image[yy];
            for (x = 0; x < box[2] / xscale; x++) {
                int xx_from = box[0] + x*xscale;
                __m128i mm_ss0 = mm_amend;
                for (xx = xx_from; xx < xx_from + xscale - 1; xx += 2) {
                    __m128i pix0 = _mm_unpacklo_epi8(
                        _mm_loadl_epi64((__m128i *) &line[xx*4]),
                        _mm_setzero_si128());
                    pix0 = _mm_add_epi32(
                        _mm_unpacklo_epi16(pix0, _mm_setzero_si128()),
                        _mm_unpackhi_epi16(pix0, _mm_setzero_si128()));
                    mm_ss0 = _mm_add_epi32(mm_ss0, pix0);
                }
                if (xscale & 0x01) {
                    __m128i pix0 = mm_cvtepu8_epi32(&line[xx*4]);
                    mm_ss0 = _mm_add_epi32(mm_ss0, pix0);
                }

                mm_ss0 = _mm_mullo_epi32(mm_ss0, mm_multiplier);
                mm_ss0 = _mm_srli_epi32(mm_ss0, 24);
                mm_ss0 = _mm_packs_epi32(mm_ss0, mm_ss0);
                ((UINT32 *)imOut->image[y])[x] = _mm_cvtsi128_si32(_mm_packus_epi16(mm_ss0, mm_ss0));
            }
        }
#elif defined(__riscv_vector)
        vuint32m1_t rv_multiplier = __riscv_vmv_v_x_u32m1(multiplier, 4);
        vuint32m1_t rv_amend = __riscv_vmv_v_x_u32m1(amend, 4);
        /* Process 2 output rows at a time */
        for (y = 0; y < box[3] / yscale - 1; y += 2) {
            int yy = box[1] + y*yscale;
            UINT8 *line0 = (UINT8 *)imIn->image[yy + 0];
            UINT8 *line1 = (UINT8 *)imIn->image[yy + 1];
            for (x = 0; x < box[2] / xscale; x++) {
                int xx_from = box[0] + x*xscale;
                vuint32m1_t rv_ss0 = rv_amend;
                vuint32m1_t rv_ss1 = rv_amend;
                for (xx = xx_from; xx < xx_from + xscale - 1; xx += 2) {
                    vuint32m1_t lo0, hi0, lo1, hi1;
                    rvv_load_2px(&line0[xx*4], &lo0, &hi0);
                    rvv_load_2px(&line1[xx*4], &lo1, &hi1);
                    rv_ss0 = __riscv_vadd_vv_u32m1(rv_ss0,
                        __riscv_vadd_vv_u32m1(lo0, hi0, 4), 4);
                    rv_ss1 = __riscv_vadd_vv_u32m1(rv_ss1,
                        __riscv_vadd_vv_u32m1(lo1, hi1, 4), 4);
                }
                if (xscale & 0x01) {
                    vuint32m1_t pix0 = rvv_cvtepu8_epi32(&line0[xx*4]);
                    vuint32m1_t pix1 = rvv_cvtepu8_epi32(&line1[xx*4]);
                    rv_ss0 = __riscv_vadd_vv_u32m1(rv_ss0, pix0, 4);
                    rv_ss1 = __riscv_vadd_vv_u32m1(rv_ss1, pix1, 4);
                }

                rv_ss0 = __riscv_vmul_vv_u32m1(rv_ss0, rv_multiplier, 4);
                rv_ss1 = __riscv_vmul_vv_u32m1(rv_ss1, rv_multiplier, 4);
                rv_ss0 = __riscv_vsrl_vx_u32m1(rv_ss0, 24, 4);
                rv_ss1 = __riscv_vsrl_vx_u32m1(rv_ss1, 24, 4);
                rvv_pack_u32_to_u8x4(&((UINT32 *)imOut->image[y + 0])[x], rv_ss0);
                rvv_pack_u32_to_u8x4(&((UINT32 *)imOut->image[y + 1])[x], rv_ss1);
            }
        }
        /* Remaining single row */
        for (; y < box[3] / yscale; y++) {
            int yy = box[1] + y*yscale;
            UINT8 *line = (UINT8 *)imIn->image[yy];
            for (x = 0; x < box[2] / xscale; x++) {
                int xx_from = box[0] + x*xscale;
                vuint32m1_t rv_ss0 = rv_amend;
                for (xx = xx_from; xx < xx_from + xscale - 1; xx += 2) {
                    vuint32m1_t lo0, hi0;
                    rvv_load_2px(&line[xx*4], &lo0, &hi0);
                    rv_ss0 = __riscv_vadd_vv_u32m1(rv_ss0,
                        __riscv_vadd_vv_u32m1(lo0, hi0, 4), 4);
                }
                if (xscale & 0x01) {
                    vuint32m1_t pix0 = rvv_cvtepu8_epi32(&line[xx*4]);
                    rv_ss0 = __riscv_vadd_vv_u32m1(rv_ss0, pix0, 4);
                }

                rv_ss0 = __riscv_vmul_vv_u32m1(rv_ss0, rv_multiplier, 4);
                rv_ss0 = __riscv_vsrl_vx_u32m1(rv_ss0, 24, 4);
                rvv_pack_u32_to_u8x4(&((UINT32 *)imOut->image[y])[x], rv_ss0);
            }
        }
#else
        /* Scalar fallback */
        for (y = 0; y < box[3] / yscale; y++) {
            int yy = box[1] + y*yscale;
            UINT8 *line = (UINT8 *)imIn->image[yy];
            for (x = 0; x < box[2] / xscale; x++) {
                int xx_from = box[0] + x*xscale;
                UINT32 ss0 = amend, ss1 = amend, ss2 = amend, ss3 = amend;
                for (xx = xx_from; xx < xx_from + xscale; xx++) {
                    ss0 += line[xx*4 + 0];
                    ss1 += line[xx*4 + 1];
                    ss2 += line[xx*4 + 2];
                    ss3 += line[xx*4 + 3];
                }
                ss0 = (ss0 * multiplier) >> 24;
                ss1 = (ss1 * multiplier) >> 24;
                ss2 = (ss2 * multiplier) >> 24;
                ss3 = (ss3 * multiplier) >> 24;
                ((UINT32 *)imOut->image[y])[x] = scalar_pack_u32(ss0, ss1, ss2, ss3);
            }
        }
#endif
    }
}

void
ImagingReduce1x2(Imaging imOut, Imaging imIn, int box[4]) {
    /* Optimized implementation for xscale = 1 and yscale = 2.
     */
    int xscale = 1, yscale = 2;
    int x, y;
    UINT32 amend = yscale * xscale / 2;

    if (imIn->image8) {
        for (y = 0; y < box[3] / yscale; y++) {
            int yy = box[1] + y * yscale;
            UINT8 *line0 = (UINT8 *)imIn->image8[yy + 0];
            UINT8 *line1 = (UINT8 *)imIn->image8[yy + 1];
            for (x = 0; x < box[2] / xscale; x++) {
                int xx = box[0] + x*xscale;
                UINT32 ss0 = line0[xx + 0] +
                      line1[xx + 0];
                imOut->image8[y][x] = (ss0 + amend) >> 1;
            }
        }
    } else {
#if defined(__SSE4_2__)
        __m128i mm_amend = _mm_set1_epi32(amend);
        for (y = 0; y < box[3] / yscale; y++) {
            int yy = box[1] + y * yscale;
            UINT8 *line0 = (UINT8 *)imIn->image[yy + 0];
            UINT8 *line1 = (UINT8 *)imIn->image[yy + 1];
            for (x = 0; x < box[2] / xscale; x++) {
                int xx = box[0] + x*xscale;
                __m128i mm_ss0;

                __m128i pix0 = mm_cvtepu8_epi32(&line0[xx*4]);
                __m128i pix1 = mm_cvtepu8_epi32(&line1[xx*4]);
                mm_ss0 = _mm_add_epi32(pix0, pix1);
                mm_ss0 = _mm_add_epi32(mm_ss0, mm_amend);
                mm_ss0 = _mm_srli_epi32(mm_ss0, 1);
                mm_ss0 = _mm_packs_epi32(mm_ss0, mm_ss0);
                ((UINT32 *)imOut->image[y])[x] = _mm_cvtsi128_si32(_mm_packus_epi16(mm_ss0, mm_ss0));
            }
        }
#elif defined(__riscv_vector)
        /* VL-agnostic: process multiple pixels per iteration.
         * xscale=1 so each output pixel reads 4 bytes from each row.
         * We treat the row as a flat byte array and process N*4 channels at once.
         * Use u16 arithmetic: u8+u8 fits in u16, then add amend and shift.
         */
        for (y = 0; y < box[3] / yscale; y++) {
            int yy = box[1] + y * yscale;
            UINT8 *line0 = (UINT8 *)imIn->image[yy + 0];
            UINT8 *line1 = (UINT8 *)imIn->image[yy + 1];
            int x_base = box[0];
            int x_count = box[2] / xscale;
            for (x = 0; x < x_count; ) {
                size_t n_bytes = (size_t)(x_count - x) * 4;
                size_t vl = __riscv_vsetvl_e8m2(n_bytes);
                vuint8m2_t p0 = __riscv_vle8_v_u8m2(&line0[(x_base + x) * 4], vl);
                vuint8m2_t p1 = __riscv_vle8_v_u8m2(&line1[(x_base + x) * 4], vl);
                vuint16m4_t sum = __riscv_vwaddu_vv_u16m4(p0, p1, vl);
                sum = __riscv_vadd_vx_u16m4(sum, (uint16_t)amend, vl);
                vuint8m2_t avg = RVV_VNCLIPU(u8m2, sum, 1, vl);
                __riscv_vse8_v_u8m2((UINT8 *)&((UINT32 *)imOut->image[y])[x], avg, vl);
                x += vl / 4;
            }
        }
#else
        /* Scalar fallback */
        for (y = 0; y < box[3] / yscale; y++) {
            int yy = box[1] + y * yscale;
            UINT8 *line0 = (UINT8 *)imIn->image[yy + 0];
            UINT8 *line1 = (UINT8 *)imIn->image[yy + 1];
            for (x = 0; x < box[2] / xscale; x++) {
                int xx = box[0] + x*xscale;
                UINT32 ss0 = (line0[xx*4+0] + line1[xx*4+0] + amend) >> 1;
                UINT32 ss1 = (line0[xx*4+1] + line1[xx*4+1] + amend) >> 1;
                UINT32 ss2 = (line0[xx*4+2] + line1[xx*4+2] + amend) >> 1;
                UINT32 ss3 = (line0[xx*4+3] + line1[xx*4+3] + amend) >> 1;
                ((UINT32 *)imOut->image[y])[x] = scalar_pack_u32(ss0, ss1, ss2, ss3);
            }
        }
#endif
    }
}

void
ImagingReduce2x1(Imaging imOut, Imaging imIn, int box[4]) {
    /* Optimized implementation for xscale = 2 and yscale = 1.
     */
    int xscale = 2, yscale = 1;
    int x, y;
    UINT32 amend = yscale * xscale / 2;

    if (imIn->image8) {
        for (y = 0; y < box[3] / yscale; y++) {
            int yy = box[1] + y * yscale;
            UINT8 *line0 = (UINT8 *)imIn->image8[yy + 0];
            for (x = 0; x < box[2] / xscale; x++) {
                int xx = box[0] + x*xscale;
                UINT32 ss0 = line0[xx + 0] + line0[xx + 1];
                imOut->image8[y][x] = (ss0 + amend) >> 1;
            }
        }
    } else {
#if defined(__SSE4_2__)
        __m128i mm_amend = _mm_set1_epi16(amend);
        for (y = 0; y < box[3] / yscale; y++) {
            int yy = box[1] + y * yscale;
            UINT8 *line0 = (UINT8 *)imIn->image[yy + 0];
            for (x = 0; x < box[2] / xscale; x++) {
                int xx = box[0] + x*xscale;
                __m128i mm_ss0;

                __m128i pix0 = _mm_unpacklo_epi8(
                    _mm_loadl_epi64((__m128i *) &line0[xx*4]),
                    _mm_setzero_si128());
                pix0 = _mm_add_epi16(_mm_srli_si128(pix0, 8), pix0);
                mm_ss0 = _mm_add_epi16(pix0, mm_amend);
                mm_ss0 = _mm_srli_epi16(mm_ss0, 1);
                ((UINT32 *)imOut->image[y])[x] = _mm_cvtsi128_si32(_mm_packus_epi16(mm_ss0, mm_ss0));
            }
        }
#elif defined(__riscv_vector)
        /* VL-agnostic: use vlseg2e32 to deinterleave pairs of adjacent pixels.
         * For xscale=2, each output pixel is the average of 2 adjacent input pixels.
         */
        for (y = 0; y < box[3] / yscale; y++) {
            int yy = box[1] + y * yscale;
            UINT8 *line0 = (UINT8 *)imIn->image[yy + 0];
            int x_count = box[2] / xscale;
            for (x = 0; x < x_count; ) {
                size_t n_out = (size_t)(x_count - x);
                size_t vl = __riscv_vsetvl_e32m2(n_out);
                int xx = box[0] + x * xscale;

                /* Deinterleave pairs of u32 pixels */
                vuint32m2_t px_even, px_odd;
                RVV_VLSEG2E32_U32M2(px_even, px_odd, (const uint32_t *)&line0[xx * 4], vl);

                /* Add per-channel via byte reinterpret + widening add */
                vuint8m2_t be = __riscv_vreinterpret_v_u32m2_u8m2(px_even);
                vuint8m2_t bo = __riscv_vreinterpret_v_u32m2_u8m2(px_odd);
                size_t bvl = vl * 4;
                vuint16m4_t sum = __riscv_vwaddu_vv_u16m4(be, bo, bvl);
                sum = __riscv_vadd_vx_u16m4(sum, (uint16_t)amend, bvl);
                vuint8m2_t result = RVV_VNCLIPU(u8m2, sum, 1, bvl);
                __riscv_vse8_v_u8m2((UINT8 *)&((UINT32 *)imOut->image[y])[x], result, bvl);
                x += vl;
            }
        }
#else
        /* Scalar fallback */
        for (y = 0; y < box[3] / yscale; y++) {
            int yy = box[1] + y * yscale;
            UINT8 *line0 = (UINT8 *)imIn->image[yy + 0];
            for (x = 0; x < box[2] / xscale; x++) {
                int xx = box[0] + x*xscale;
                UINT32 ss0 = (line0[xx*4+0] + line0[xx*4+4] + amend) >> 1;
                UINT32 ss1 = (line0[xx*4+1] + line0[xx*4+5] + amend) >> 1;
                UINT32 ss2 = (line0[xx*4+2] + line0[xx*4+6] + amend) >> 1;
                UINT32 ss3 = (line0[xx*4+3] + line0[xx*4+7] + amend) >> 1;
                ((UINT32 *)imOut->image[y])[x] = scalar_pack_u32(ss0, ss1, ss2, ss3);
            }
        }
#endif
    }
}

void
ImagingReduce2x2(Imaging imOut, Imaging imIn, int box[4]) {
    /* Optimized implementation for xscale = 2 and yscale = 2.
     */
    int xscale = 2, yscale = 2;
    int x, y;
    UINT32 amend = yscale * xscale / 2;

    if (imIn->image8) {
        for (y = 0; y < box[3] / yscale; y++) {
            int yy = box[1] + y * yscale;
            UINT8 *line0 = (UINT8 *)imIn->image8[yy + 0];
            UINT8 *line1 = (UINT8 *)imIn->image8[yy + 1];
            for (x = 0; x < box[2] / xscale; x++) {
                int xx = box[0] + x*xscale;
                UINT32 ss0 = line0[xx + 0] + line0[xx + 1] +
                      line1[xx + 0] + line1[xx + 1];
                imOut->image8[y][x] = (ss0 + amend) >> 2;
            }
        }
    } else {
#if defined(__SSE4_2__)
        __m128i mm_amend = _mm_set1_epi16(amend);
        for (y = 0; y < box[3] / yscale; y++) {
            int yy = box[1] + y * yscale;
            UINT8 *line0 = (UINT8 *)imIn->image[yy + 0];
            UINT8 *line1 = (UINT8 *)imIn->image[yy + 1];
            for (x = 0; x < box[2] / xscale; x++) {
                int xx = box[0] + x*xscale;
                __m128i mm_ss0;

                __m128i pix0 = _mm_unpacklo_epi8(
                    _mm_loadl_epi64((__m128i *) &line0[xx*4]),
                    _mm_setzero_si128());
                __m128i pix1 = _mm_unpacklo_epi8(
                    _mm_loadl_epi64((__m128i *) &line1[xx*4]),
                    _mm_setzero_si128());
                pix0 = _mm_add_epi16(pix0, pix1);
                pix0 = _mm_add_epi16(_mm_srli_si128(pix0, 8), pix0);
                mm_ss0 = _mm_add_epi16(pix0, mm_amend);
                mm_ss0 = _mm_srli_epi16(mm_ss0, 2);
                ((UINT32 *)imOut->image[y])[x] = _mm_cvtsi128_si32(_mm_packus_epi16(mm_ss0, mm_ss0));
            }
        }
#elif defined(__riscv_vector)
        /* VL-agnostic: use vlseg2e32 to deinterleave pairs of adjacent pixels.
         * For each output pixel, we need pixel[2*i] + pixel[2*i+1] from each row.
         * vlseg2e32 loads even-indexed and odd-indexed u32 values into separate registers.
         */
        for (y = 0; y < box[3] / yscale; y++) {
            int yy = box[1] + y * yscale;
            UINT8 *line0 = (UINT8 *)imIn->image[yy + 0];
            UINT8 *line1 = (UINT8 *)imIn->image[yy + 1];
            int x_count = box[2] / xscale;
            for (x = 0; x < x_count; ) {
                size_t n_out = (size_t)(x_count - x);
                size_t vl = __riscv_vsetvl_e32m2(n_out);
                int xx = box[0] + x * xscale;

                /* Load pairs: vlseg2 loads pixel[0],pixel[2],... into v0
                 * and pixel[1],pixel[3],... into v1 (as u32 RGBA words) */
                vuint32m2_t r0_even, r0_odd, r1_even, r1_odd;
                RVV_VLSEG2E32_U32M2(r0_even, r0_odd, (const uint32_t *)&line0[xx * 4], vl);
                RVV_VLSEG2E32_U32M2(r1_even, r1_odd, (const uint32_t *)&line1[xx * 4], vl);

                /* We need per-channel addition, so reinterpret as bytes,
                 * widen to u16, and add all four sources. */
                vuint8m2_t b0e = __riscv_vreinterpret_v_u32m2_u8m2(r0_even);
                vuint8m2_t b0o = __riscv_vreinterpret_v_u32m2_u8m2(r0_odd);
                vuint8m2_t b1e = __riscv_vreinterpret_v_u32m2_u8m2(r1_even);
                vuint8m2_t b1o = __riscv_vreinterpret_v_u32m2_u8m2(r1_odd);

                size_t bvl = vl * 4; /* byte-level vl */
                vuint16m4_t sum = __riscv_vwaddu_vv_u16m4(b0e, b0o, bvl);
                vuint16m4_t sum1 = __riscv_vwaddu_vv_u16m4(b1e, b1o, bvl);
                sum = __riscv_vadd_vv_u16m4(sum, sum1, bvl);
                sum = __riscv_vadd_vx_u16m4(sum, (uint16_t)amend, bvl);
                vuint8m2_t result = RVV_VNCLIPU(u8m2, sum, 2, bvl);
                __riscv_vse8_v_u8m2((UINT8 *)&((UINT32 *)imOut->image[y])[x], result, bvl);
                x += vl;
            }
        }
#else
        /* Scalar fallback */
        for (y = 0; y < box[3] / yscale; y++) {
            int yy = box[1] + y * yscale;
            UINT8 *line0 = (UINT8 *)imIn->image[yy + 0];
            UINT8 *line1 = (UINT8 *)imIn->image[yy + 1];
            for (x = 0; x < box[2] / xscale; x++) {
                int xx = box[0] + x*xscale;
                UINT32 ss0 = (line0[xx*4+0] + line0[xx*4+4] + line1[xx*4+0] + line1[xx*4+4] + amend) >> 2;
                UINT32 ss1 = (line0[xx*4+1] + line0[xx*4+5] + line1[xx*4+1] + line1[xx*4+5] + amend) >> 2;
                UINT32 ss2 = (line0[xx*4+2] + line0[xx*4+6] + line1[xx*4+2] + line1[xx*4+6] + amend) >> 2;
                UINT32 ss3 = (line0[xx*4+3] + line0[xx*4+7] + line1[xx*4+3] + line1[xx*4+7] + amend) >> 2;
                ((UINT32 *)imOut->image[y])[x] = scalar_pack_u32(ss0, ss1, ss2, ss3);
            }
        }
#endif
    }
}

void
ImagingReduce1x3(Imaging imOut, Imaging imIn, int box[4]) {
    /* Optimized implementation for xscale = 1 and yscale = 3.
     */
    int xscale = 1, yscale = 3;
    int x, y;
    UINT32 multiplier = division_UINT32(yscale * xscale, 8);
    UINT32 amend = yscale * xscale / 2;

    if (imIn->image8) {
        for (y = 0; y < box[3] / yscale; y++) {
            int yy = box[1] + y * yscale;
            UINT8 *line0 = (UINT8 *)imIn->image8[yy + 0];
            UINT8 *line1 = (UINT8 *)imIn->image8[yy + 1];
            UINT8 *line2 = (UINT8 *)imIn->image8[yy + 2];
            for (x = 0; x < box[2] / xscale; x++) {
                int xx = box[0] + x*xscale;
                UINT32 ss0 = line0[xx + 0] +
                      line1[xx + 0] +
                      line2[xx + 0];
                imOut->image8[y][x] = ((ss0 + amend) * multiplier) >> 24;
            }
        }
    } else {
#if defined(__SSE4_2__)
        __m128i mm_multiplier = _mm_set1_epi32(multiplier);
        __m128i mm_amend = _mm_set1_epi32(amend);
        for (y = 0; y < box[3] / yscale; y++) {
            int yy = box[1] + y * yscale;
            UINT8 *line0 = (UINT8 *)imIn->image[yy + 0];
            UINT8 *line1 = (UINT8 *)imIn->image[yy + 1];
            UINT8 *line2 = (UINT8 *)imIn->image[yy + 2];
            for (x = 0; x < box[2] / xscale; x++) {
                int xx = box[0] + x*xscale;
                __m128i mm_ss0 = _mm_add_epi32(
                    _mm_add_epi32(
                        mm_cvtepu8_epi32(&line0[xx*4]),
                        mm_cvtepu8_epi32(&line1[xx*4])),
                    _mm_add_epi32(
                        mm_cvtepu8_epi32(&line2[xx*4]),
                        mm_amend));

                mm_ss0 = _mm_add_epi32(mm_ss0, mm_amend);
                mm_ss0 = _mm_mullo_epi32(mm_ss0, mm_multiplier);
                mm_ss0 = _mm_srli_epi32(mm_ss0, 24);
                mm_ss0 = _mm_packs_epi32(mm_ss0, mm_ss0);
                ((UINT32 *)imOut->image[y])[x] = _mm_cvtsi128_si32(_mm_packus_epi16(mm_ss0, mm_ss0));
            }
        }
#elif defined(__riscv_vector)
        /* VL-agnostic: process multiple pixels per iteration.
         * xscale=1 so output pixels are contiguous. Sum 3 rows in u16 (max 255*3=765, fits u16),
         * then widen to u32 for multiply+shift.
         */
        for (y = 0; y < box[3] / yscale; y++) {
            int yy = box[1] + y * yscale;
            UINT8 *line0 = (UINT8 *)imIn->image[yy + 0];
            UINT8 *line1 = (UINT8 *)imIn->image[yy + 1];
            UINT8 *line2 = (UINT8 *)imIn->image[yy + 2];
            int x_base = box[0];
            int x_count = box[2] / xscale;
            for (x = 0; x < x_count; ) {
                size_t n_bytes = (size_t)(x_count - x) * 4;
                size_t vl = __riscv_vsetvl_e16m2(n_bytes);

                vuint8m1_t p0 = __riscv_vle8_v_u8m1(&line0[(x_base + x) * 4], vl);
                vuint8m1_t p1 = __riscv_vle8_v_u8m1(&line1[(x_base + x) * 4], vl);
                vuint8m1_t p2 = __riscv_vle8_v_u8m1(&line2[(x_base + x) * 4], vl);

                vuint16m2_t sum = __riscv_vwaddu_vv_u16m2(p0, p1, vl);
                vuint16m2_t p2w = __riscv_vzext_vf2_u16m2(p2, vl);
                sum = __riscv_vadd_vv_u16m2(sum, p2w, vl);
                sum = __riscv_vadd_vx_u16m2(sum, (uint16_t)amend, vl);

                /* Widen to u32 for multiply */
                vuint32m4_t sum32 = __riscv_vzext_vf2_u32m4(sum, vl);
                sum32 = __riscv_vmul_vx_u32m4(sum32, multiplier, vl);
                sum32 = __riscv_vsrl_vx_u32m4(sum32, 24, vl);

                /* Narrow u32 -> u16 -> u8 */
                vuint16m2_t n16 = RVV_VNCLIPU(u16m2, sum32, 0, vl);
                vuint8m1_t n8 = RVV_VNCLIPU(u8m1, n16, 0, vl);
                __riscv_vse8_v_u8m1((UINT8 *)&((UINT32 *)imOut->image[y])[x], n8, vl);
                x += vl / 4;
            }
        }
#else
        /* Scalar fallback */
        for (y = 0; y < box[3] / yscale; y++) {
            int yy = box[1] + y * yscale;
            UINT8 *line0 = (UINT8 *)imIn->image[yy + 0];
            UINT8 *line1 = (UINT8 *)imIn->image[yy + 1];
            UINT8 *line2 = (UINT8 *)imIn->image[yy + 2];
            for (x = 0; x < box[2] / xscale; x++) {
                int xx = box[0] + x*xscale;
                UINT32 ss0 = ((line0[xx*4+0] + line1[xx*4+0] + line2[xx*4+0] + amend) * multiplier) >> 24;
                UINT32 ss1 = ((line0[xx*4+1] + line1[xx*4+1] + line2[xx*4+1] + amend) * multiplier) >> 24;
                UINT32 ss2 = ((line0[xx*4+2] + line1[xx*4+2] + line2[xx*4+2] + amend) * multiplier) >> 24;
                UINT32 ss3 = ((line0[xx*4+3] + line1[xx*4+3] + line2[xx*4+3] + amend) * multiplier) >> 24;
                ((UINT32 *)imOut->image[y])[x] = scalar_pack_u32(ss0, ss1, ss2, ss3);
            }
        }
#endif
    }
}

void
ImagingReduce3x1(Imaging imOut, Imaging imIn, int box[4]) {
    /* Optimized implementation for xscale = 3 and yscale = 1.
     */
    int xscale = 3, yscale = 1;
    int x, y;
    UINT32 multiplier = division_UINT32(yscale * xscale, 8);
    UINT32 amend = yscale * xscale / 2;

    if (imIn->image8) {
        for (y = 0; y < box[3] / yscale; y++) {
            int yy = box[1] + y * yscale;
            UINT8 *line0 = (UINT8 *)imIn->image8[yy + 0];
            for (x = 0; x < box[2] / xscale; x++) {
                int xx = box[0] + x*xscale;
                UINT32 ss0 = line0[xx + 0] + line0[xx + 1] + line0[xx + 2];
                imOut->image8[y][x] = ((ss0 + amend) * multiplier) >> 24;
            }
        }
    } else {
#if defined(__SSE4_2__)
        __m128i mm_multiplier = _mm_set1_epi32(multiplier);
        __m128i mm_amend = _mm_set1_epi32(amend);
        for (y = 0; y < box[3] / yscale; y++) {
            int yy = box[1] + y * yscale;
            UINT8 *line0 = (UINT8 *)imIn->image[yy + 0];
            for (x = 0; x < box[2] / xscale; x++) {
                int xx = box[0] + x*xscale;
                __m128i mm_ss0;

                __m128i pix0 = _mm_unpacklo_epi8(
                    _mm_loadl_epi64((__m128i *) &line0[xx*4]),
                    _mm_setzero_si128());
                pix0 = _mm_add_epi32(
                    _mm_unpacklo_epi16(pix0, _mm_setzero_si128()),
                    _mm_unpackhi_epi16(pix0, _mm_setzero_si128()));

                mm_ss0 = _mm_add_epi32(
                    mm_cvtepu8_epi32(&line0[xx*4 + 8]),
                    pix0);

                mm_ss0 = _mm_add_epi32(mm_ss0, mm_amend);
                mm_ss0 = _mm_mullo_epi32(mm_ss0, mm_multiplier);
                mm_ss0 = _mm_srli_epi32(mm_ss0, 24);
                mm_ss0 = _mm_packs_epi32(mm_ss0, mm_ss0);
                ((UINT32 *)imOut->image[y])[x] = _mm_cvtsi128_si32(_mm_packus_epi16(mm_ss0, mm_ss0));
            }
        }
#elif defined(__riscv_vector)
        /* VL-agnostic: use vlseg3e32 to deinterleave triplets of adjacent pixels. */
        for (y = 0; y < box[3] / yscale; y++) {
            int yy = box[1] + y * yscale;
            UINT8 *line0 = (UINT8 *)imIn->image[yy + 0];
            int x_count = box[2] / xscale;
            for (x = 0; x < x_count; ) {
                size_t n_out = (size_t)(x_count - x);
                size_t vl = __riscv_vsetvl_e32m2(n_out);
                int xx = box[0] + x * xscale;

                vuint32m2_t px0, px1, px2;
                RVV_VLSEG3E32_U32M2(px0, px1, px2, (const uint32_t *)&line0[xx * 4], vl);

                /* Per-channel sum via byte reinterpret + widening */
                vuint8m2_t b0 = __riscv_vreinterpret_v_u32m2_u8m2(px0);
                vuint8m2_t b1 = __riscv_vreinterpret_v_u32m2_u8m2(px1);
                vuint8m2_t b2 = __riscv_vreinterpret_v_u32m2_u8m2(px2);

                size_t bvl = vl * 4;
                vuint16m4_t sum = __riscv_vwaddu_vv_u16m4(b0, b1, bvl);
                vuint16m4_t b2w = __riscv_vzext_vf2_u16m4(b2, bvl);
                sum = __riscv_vadd_vv_u16m4(sum, b2w, bvl);
                sum = __riscv_vadd_vx_u16m4(sum, (uint16_t)amend, bvl);

                /* Widen to u32 for fixed-point multiply */
                vuint32m8_t sum32 = __riscv_vzext_vf2_u32m8(sum, bvl);
                sum32 = __riscv_vmul_vx_u32m8(sum32, multiplier, bvl);
                sum32 = __riscv_vsrl_vx_u32m8(sum32, 24, bvl);

                vuint16m4_t n16 = RVV_VNCLIPU(u16m4, sum32, 0, bvl);
                vuint8m2_t n8 = RVV_VNCLIPU(u8m2, n16, 0, bvl);
                __riscv_vse8_v_u8m2((UINT8 *)&((UINT32 *)imOut->image[y])[x], n8, bvl);
                x += vl;
            }
        }
#else
        /* Scalar fallback */
        for (y = 0; y < box[3] / yscale; y++) {
            int yy = box[1] + y * yscale;
            UINT8 *line0 = (UINT8 *)imIn->image[yy + 0];
            for (x = 0; x < box[2] / xscale; x++) {
                int xx = box[0] + x*xscale;
                UINT32 ss0 = ((line0[xx*4+0] + line0[xx*4+4] + line0[xx*4+8] + amend) * multiplier) >> 24;
                UINT32 ss1 = ((line0[xx*4+1] + line0[xx*4+5] + line0[xx*4+9] + amend) * multiplier) >> 24;
                UINT32 ss2 = ((line0[xx*4+2] + line0[xx*4+6] + line0[xx*4+10] + amend) * multiplier) >> 24;
                UINT32 ss3 = ((line0[xx*4+3] + line0[xx*4+7] + line0[xx*4+11] + amend) * multiplier) >> 24;
                ((UINT32 *)imOut->image[y])[x] = scalar_pack_u32(ss0, ss1, ss2, ss3);
            }
        }
#endif
    }
}

void
ImagingReduce3x3(Imaging imOut, Imaging imIn, int box[4]) {
    /* Optimized implementation for xscale = 3 and yscale = 3.
     */
    int xscale = 3, yscale = 3;
    int x, y;
    UINT32 multiplier = division_UINT32(yscale * xscale, 8);
    UINT32 amend = yscale * xscale / 2;

    if (imIn->image8) {
        for (y = 0; y < box[3] / yscale; y++) {
            int yy = box[1] + y * yscale;
            UINT8 *line0 = (UINT8 *)imIn->image8[yy + 0];
            UINT8 *line1 = (UINT8 *)imIn->image8[yy + 1];
            UINT8 *line2 = (UINT8 *)imIn->image8[yy + 2];
            for (x = 0; x < box[2] / xscale; x++) {
                int xx = box[0] + x*xscale;
                UINT32 ss0 = line0[xx + 0] + line0[xx + 1] + line0[xx + 2] +
                      line1[xx + 0] + line1[xx + 1] + line1[xx + 2] +
                      line2[xx + 0] + line2[xx + 1] + line2[xx + 2];
                imOut->image8[y][x] = ((ss0 + amend) * multiplier) >> 24;
            }
        }
    } else {
#if defined(__SSE4_2__)
        __m128i mm_multiplier = _mm_set1_epi32(multiplier);
        __m128i mm_amend = _mm_set1_epi32(amend);
        for (y = 0; y < box[3] / yscale; y++) {
            int yy = box[1] + y * yscale;
            UINT8 *line0 = (UINT8 *)imIn->image[yy + 0];
            UINT8 *line1 = (UINT8 *)imIn->image[yy + 1];
            UINT8 *line2 = (UINT8 *)imIn->image[yy + 2];
            for (x = 0; x < box[2] / xscale; x++) {
                int xx = box[0] + x*xscale;
                __m128i mm_ss0;

                __m128i pix0 = _mm_unpacklo_epi8(
                    _mm_loadl_epi64((__m128i *) &line0[xx*4]),
                    _mm_setzero_si128());
                __m128i pix1 = _mm_unpacklo_epi8(
                    _mm_loadl_epi64((__m128i *) &line1[xx*4]),
                    _mm_setzero_si128());
                __m128i pix2 = _mm_unpacklo_epi8(
                    _mm_loadl_epi64((__m128i *) &line2[xx*4]),
                    _mm_setzero_si128());

                pix0 = _mm_add_epi16(_mm_add_epi16(pix0, pix1), pix2);
                pix0 = _mm_add_epi32(
                    _mm_unpacklo_epi16(pix0, _mm_setzero_si128()),
                    _mm_unpackhi_epi16(pix0, _mm_setzero_si128()));

                mm_ss0 = _mm_add_epi32(
                    _mm_add_epi32(
                        mm_cvtepu8_epi32(&line0[xx*4 + 8]),
                        mm_cvtepu8_epi32(&line1[xx*4 + 8])),
                    _mm_add_epi32(
                        mm_cvtepu8_epi32(&line2[xx*4 + 8]),
                        pix0));

                mm_ss0 = _mm_add_epi32(mm_ss0, mm_amend);
                mm_ss0 = _mm_mullo_epi32(mm_ss0, mm_multiplier);
                mm_ss0 = _mm_srli_epi32(mm_ss0, 24);
                mm_ss0 = _mm_packs_epi32(mm_ss0, mm_ss0);
                ((UINT32 *)imOut->image[y])[x] = _mm_cvtsi128_si32(_mm_packus_epi16(mm_ss0, mm_ss0));
            }
        }
#elif defined(__riscv_vector)
        /* VL-agnostic: use vlseg3e32 to deinterleave triplets of adjacent pixels,
         * then sum across 3 rows and 3 columns.
         */
        for (y = 0; y < box[3] / yscale; y++) {
            int yy = box[1] + y * yscale;
            UINT8 *line0 = (UINT8 *)imIn->image[yy + 0];
            UINT8 *line1 = (UINT8 *)imIn->image[yy + 1];
            UINT8 *line2 = (UINT8 *)imIn->image[yy + 2];
            int x_count = box[2] / xscale;
            for (x = 0; x < x_count; ) {
                size_t n_out = (size_t)(x_count - x);
                size_t vl = __riscv_vsetvl_e32m1(n_out);
                int xx = box[0] + x * xscale;

                /* Deinterleave 3 pixel columns from each row */
                vuint32m1_t r0p0, r0p1, r0p2;
                vuint32m1_t r1p0, r1p1, r1p2;
                vuint32m1_t r2p0, r2p1, r2p2;
                RVV_VLSEG3E32_U32M1(r0p0, r0p1, r0p2, (const uint32_t *)&line0[xx * 4], vl);
                RVV_VLSEG3E32_U32M1(r1p0, r1p1, r1p2, (const uint32_t *)&line1[xx * 4], vl);
                RVV_VLSEG3E32_U32M1(r2p0, r2p1, r2p2, (const uint32_t *)&line2[xx * 4], vl);

                /* Sum all 9 sources per-channel via byte widening.
                 * First sum the 3 rows for each column position in u16,
                 * then sum the 3 column positions in u32.
                 */
                size_t bvl = vl * 4;

                /* Column 0: rows 0+1+2 */
                vuint16m2_t c0 = __riscv_vwaddu_vv_u16m2(
                    __riscv_vreinterpret_v_u32m1_u8m1(r0p0),
                    __riscv_vreinterpret_v_u32m1_u8m1(r1p0), bvl);
                c0 = __riscv_vadd_vv_u16m2(c0,
                    __riscv_vzext_vf2_u16m2(__riscv_vreinterpret_v_u32m1_u8m1(r2p0), bvl), bvl);

                /* Column 1: rows 0+1+2 */
                vuint16m2_t c1 = __riscv_vwaddu_vv_u16m2(
                    __riscv_vreinterpret_v_u32m1_u8m1(r0p1),
                    __riscv_vreinterpret_v_u32m1_u8m1(r1p1), bvl);
                c1 = __riscv_vadd_vv_u16m2(c1,
                    __riscv_vzext_vf2_u16m2(__riscv_vreinterpret_v_u32m1_u8m1(r2p1), bvl), bvl);

                /* Column 2: rows 0+1+2 */
                vuint16m2_t c2 = __riscv_vwaddu_vv_u16m2(
                    __riscv_vreinterpret_v_u32m1_u8m1(r0p2),
                    __riscv_vreinterpret_v_u32m1_u8m1(r1p2), bvl);
                c2 = __riscv_vadd_vv_u16m2(c2,
                    __riscv_vzext_vf2_u16m2(__riscv_vreinterpret_v_u32m1_u8m1(r2p2), bvl), bvl);

                /* Sum columns in u16 (max 255*9=2295, fits u16) */
                vuint16m2_t total = __riscv_vadd_vv_u16m2(__riscv_vadd_vv_u16m2(c0, c1, bvl), c2, bvl);
                total = __riscv_vadd_vx_u16m2(total, (uint16_t)amend, bvl);

                /* Widen to u32 for fixed-point multiply */
                vuint32m4_t sum32 = __riscv_vzext_vf2_u32m4(total, bvl);
                sum32 = __riscv_vmul_vx_u32m4(sum32, multiplier, bvl);
                sum32 = __riscv_vsrl_vx_u32m4(sum32, 24, bvl);

                vuint16m2_t n16 = RVV_VNCLIPU(u16m2, sum32, 0, bvl);
                vuint8m1_t n8 = RVV_VNCLIPU(u8m1, n16, 0, bvl);
                __riscv_vse8_v_u8m1((UINT8 *)&((UINT32 *)imOut->image[y])[x], n8, bvl);
                x += vl;
            }
        }
#else
        /* Scalar fallback */
        for (y = 0; y < box[3] / yscale; y++) {
            int yy = box[1] + y * yscale;
            UINT8 *line0 = (UINT8 *)imIn->image[yy + 0];
            UINT8 *line1 = (UINT8 *)imIn->image[yy + 1];
            UINT8 *line2 = (UINT8 *)imIn->image[yy + 2];
            for (x = 0; x < box[2] / xscale; x++) {
                int xx = box[0] + x*xscale;
                int c;
                for (c = 0; c < 4; c++) {
                    UINT32 ss = line0[xx*4+c] + line0[xx*4+4+c] + line0[xx*4+8+c]
                              + line1[xx*4+c] + line1[xx*4+4+c] + line1[xx*4+8+c]
                              + line2[xx*4+c] + line2[xx*4+4+c] + line2[xx*4+8+c];
                    ((UINT8 *)&((UINT32 *)imOut->image[y])[x])[c] =
                        ((ss + amend) * multiplier) >> 24;
                }
            }
        }
#endif
    }
}

void
ImagingReduceCorners(Imaging imOut, Imaging imIn, int box[4], int xscale, int yscale) {
    /* Fill the last row and the last column for any xscale and yscale.
     */
    int x, y, xx, yy;

    if (imIn->image8) {
        if (box[2] % xscale) {
            int scale = (box[2] % xscale) * yscale;
            UINT32 multiplier = division_UINT32(scale, 8);
            UINT32 amend = scale / 2;
            for (y = 0; y < box[3] / yscale; y++) {
                int yy_from = box[1] + y * yscale;
                UINT32 ss = amend;
                x = box[2] / xscale;

                for (yy = yy_from; yy < yy_from + yscale; yy++) {
                    UINT8 *line = (UINT8 *)imIn->image8[yy];
                    for (xx = box[0] + x * xscale; xx < box[0] + box[2]; xx++) {
                        ss += line[xx + 0];
                    }
                }
                imOut->image8[y][x] = (ss * multiplier) >> 24;
            }
        }
        if (box[3] % yscale) {
            int scale = xscale * (box[3] % yscale);
            UINT32 multiplier = division_UINT32(scale, 8);
            UINT32 amend = scale / 2;
            y = box[3] / yscale;
            for (x = 0; x < box[2] / xscale; x++) {
                int xx_from = box[0] + x * xscale;
                UINT32 ss = amend;
                for (yy = box[1] + y * yscale; yy < box[1] + box[3]; yy++) {
                    UINT8 *line = (UINT8 *)imIn->image8[yy];
                    for (xx = xx_from; xx < xx_from + xscale; xx++) {
                        ss += line[xx + 0];
                    }
                }
                imOut->image8[y][x] = (ss * multiplier) >> 24;
            }
        }
        if (box[2] % xscale && box[3] % yscale) {
            int scale = (box[2] % xscale) * (box[3] % yscale);
            UINT32 multiplier = division_UINT32(scale, 8);
            UINT32 amend = scale / 2;
            UINT32 ss = amend;
            x = box[2] / xscale;
            y = box[3] / yscale;
            for (yy = box[1] + y * yscale; yy < box[1] + box[3]; yy++) {
                UINT8 *line = (UINT8 *)imIn->image8[yy];
                for (xx = box[0] + x * xscale; xx < box[0] + box[2]; xx++) {
                    ss += line[xx + 0];
                }
            }
            imOut->image8[y][x] = (ss * multiplier) >> 24;
        }
    } else {
        if (box[2] % xscale) {
            int scale = (box[2] % xscale) * yscale;
            UINT32 multiplier = division_UINT32(scale, 8);
            UINT32 amend = scale / 2;
#if defined(__SSE4_2__)
            __m128i mm_multiplier = _mm_set1_epi32(multiplier);
            __m128i mm_amend = _mm_set1_epi32(amend);
            for (y = 0; y < box[3] / yscale; y++) {
                int yy_from = box[1] + y*yscale;
                __m128i mm_ss0 = mm_amend;
                x = box[2] / xscale;

                for (yy = yy_from; yy < yy_from + yscale; yy++) {
                    UINT8 *line = (UINT8 *)imIn->image[yy];
                    for (xx = box[0] + x*xscale; xx < box[0] + box[2]; xx++) {
                        __m128i pix0 = mm_cvtepu8_epi32(&line[xx*4]);
                        mm_ss0 = _mm_add_epi32(mm_ss0, pix0);
                    }
                }
                mm_ss0 = _mm_mullo_epi32(mm_ss0, mm_multiplier);
                mm_ss0 = _mm_srli_epi32(mm_ss0, 24);
                mm_ss0 = _mm_packs_epi32(mm_ss0, mm_ss0);
                ((UINT32 *)imOut->image[y])[x] = _mm_cvtsi128_si32(_mm_packus_epi16(mm_ss0, mm_ss0));
            }
#elif defined(__riscv_vector)
            vuint32m1_t rv_multiplier = __riscv_vmv_v_x_u32m1(multiplier, 4);
            vuint32m1_t rv_amend = __riscv_vmv_v_x_u32m1(amend, 4);
            for (y = 0; y < box[3] / yscale; y++) {
                int yy_from = box[1] + y*yscale;
                vuint32m1_t rv_ss0 = rv_amend;
                x = box[2] / xscale;

                for (yy = yy_from; yy < yy_from + yscale; yy++) {
                    UINT8 *line = (UINT8 *)imIn->image[yy];
                    for (xx = box[0] + x*xscale; xx < box[0] + box[2]; xx++) {
                        vuint32m1_t pix0 = rvv_cvtepu8_epi32(&line[xx*4]);
                        rv_ss0 = __riscv_vadd_vv_u32m1(rv_ss0, pix0, 4);
                    }
                }
                rv_ss0 = __riscv_vmul_vv_u32m1(rv_ss0, rv_multiplier, 4);
                rv_ss0 = __riscv_vsrl_vx_u32m1(rv_ss0, 24, 4);
                rvv_pack_u32_to_u8x4(&((UINT32 *)imOut->image[y])[x], rv_ss0);
            }
#else
            for (y = 0; y < box[3] / yscale; y++) {
                int yy_from = box[1] + y*yscale;
                UINT32 ss0 = amend, ss1 = amend, ss2 = amend, ss3 = amend;
                x = box[2] / xscale;

                for (yy = yy_from; yy < yy_from + yscale; yy++) {
                    UINT8 *line = (UINT8 *)imIn->image[yy];
                    for (xx = box[0] + x*xscale; xx < box[0] + box[2]; xx++) {
                        ss0 += line[xx*4 + 0];
                        ss1 += line[xx*4 + 1];
                        ss2 += line[xx*4 + 2];
                        ss3 += line[xx*4 + 3];
                    }
                }
                ss0 = (ss0 * multiplier) >> 24;
                ss1 = (ss1 * multiplier) >> 24;
                ss2 = (ss2 * multiplier) >> 24;
                ss3 = (ss3 * multiplier) >> 24;
                ((UINT32 *)imOut->image[y])[x] = scalar_pack_u32(ss0, ss1, ss2, ss3);
            }
#endif
        }
        if (box[3] % yscale) {
            int scale = xscale * (box[3] % yscale);
            UINT32 multiplier = division_UINT32(scale, 8);
            UINT32 amend = scale / 2;
#if defined(__SSE4_2__)
            __m128i mm_multiplier = _mm_set1_epi32(multiplier);
            __m128i mm_amend = _mm_set1_epi32(amend);
            y = box[3] / yscale;
            for (x = 0; x < box[2] / xscale; x++) {
                int xx_from = box[0] + x*xscale;
                __m128i mm_ss0 = mm_amend;
                for (yy = box[1] + y*yscale; yy < box[1] + box[3]; yy++) {
                    UINT8 *line = (UINT8 *)imIn->image[yy];
                    for (xx = xx_from; xx < xx_from + xscale; xx++) {
                        __m128i pix0 = mm_cvtepu8_epi32(&line[xx*4]);
                        mm_ss0 = _mm_add_epi32(mm_ss0, pix0);
                    }
                }
                mm_ss0 = _mm_mullo_epi32(mm_ss0, mm_multiplier);
                mm_ss0 = _mm_srli_epi32(mm_ss0, 24);
                mm_ss0 = _mm_packs_epi32(mm_ss0, mm_ss0);
                ((UINT32 *)imOut->image[y])[x] = _mm_cvtsi128_si32(_mm_packus_epi16(mm_ss0, mm_ss0));
            }
#elif defined(__riscv_vector)
            vuint32m1_t rv_multiplier = __riscv_vmv_v_x_u32m1(multiplier, 4);
            vuint32m1_t rv_amend = __riscv_vmv_v_x_u32m1(amend, 4);
            y = box[3] / yscale;
            for (x = 0; x < box[2] / xscale; x++) {
                int xx_from = box[0] + x*xscale;
                vuint32m1_t rv_ss0 = rv_amend;
                for (yy = box[1] + y*yscale; yy < box[1] + box[3]; yy++) {
                    UINT8 *line = (UINT8 *)imIn->image[yy];
                    for (xx = xx_from; xx < xx_from + xscale; xx++) {
                        vuint32m1_t pix0 = rvv_cvtepu8_epi32(&line[xx*4]);
                        rv_ss0 = __riscv_vadd_vv_u32m1(rv_ss0, pix0, 4);
                    }
                }
                rv_ss0 = __riscv_vmul_vv_u32m1(rv_ss0, rv_multiplier, 4);
                rv_ss0 = __riscv_vsrl_vx_u32m1(rv_ss0, 24, 4);
                rvv_pack_u32_to_u8x4(&((UINT32 *)imOut->image[y])[x], rv_ss0);
            }
#else
            y = box[3] / yscale;
            for (x = 0; x < box[2] / xscale; x++) {
                int xx_from = box[0] + x*xscale;
                UINT32 ss0 = amend, ss1 = amend, ss2 = amend, ss3 = amend;
                for (yy = box[1] + y*yscale; yy < box[1] + box[3]; yy++) {
                    UINT8 *line = (UINT8 *)imIn->image[yy];
                    for (xx = xx_from; xx < xx_from + xscale; xx++) {
                        ss0 += line[xx*4 + 0];
                        ss1 += line[xx*4 + 1];
                        ss2 += line[xx*4 + 2];
                        ss3 += line[xx*4 + 3];
                    }
                }
                ss0 = (ss0 * multiplier) >> 24;
                ss1 = (ss1 * multiplier) >> 24;
                ss2 = (ss2 * multiplier) >> 24;
                ss3 = (ss3 * multiplier) >> 24;
                ((UINT32 *)imOut->image[y])[x] = scalar_pack_u32(ss0, ss1, ss2, ss3);
            }
#endif
        }
        if (box[2] % xscale && box[3] % yscale) {
            int scale = (box[2] % xscale) * (box[3] % yscale);
            UINT32 multiplier = division_UINT32(scale, 8);
            UINT32 amend = scale / 2;
#if defined(__SSE4_2__)
            __m128i mm_multiplier = _mm_set1_epi32(multiplier);
            __m128i mm_amend = _mm_set1_epi32(amend);
            __m128i mm_ss0 = mm_amend;
            x = box[2] / xscale;
            y = box[3] / yscale;
            for (yy = box[1] + y * yscale; yy < box[1] + box[3]; yy++) {
                UINT8 *line = (UINT8 *)imIn->image[yy];
                for (xx = box[0] + x*xscale; xx < box[0] + box[2]; xx++) {
                    __m128i pix0 = mm_cvtepu8_epi32(&line[xx*4]);
                    mm_ss0 = _mm_add_epi32(mm_ss0, pix0);
                }
            }
            mm_ss0 = _mm_mullo_epi32(mm_ss0, mm_multiplier);
            mm_ss0 = _mm_srli_epi32(mm_ss0, 24);
            mm_ss0 = _mm_packs_epi32(mm_ss0, mm_ss0);
            ((UINT32 *)imOut->image[y])[x] = _mm_cvtsi128_si32(_mm_packus_epi16(mm_ss0, mm_ss0));
#elif defined(__riscv_vector)
            vuint32m1_t rv_multiplier = __riscv_vmv_v_x_u32m1(multiplier, 4);
            vuint32m1_t rv_amend = __riscv_vmv_v_x_u32m1(amend, 4);
            vuint32m1_t rv_ss0 = rv_amend;
            x = box[2] / xscale;
            y = box[3] / yscale;
            for (yy = box[1] + y * yscale; yy < box[1] + box[3]; yy++) {
                UINT8 *line = (UINT8 *)imIn->image[yy];
                for (xx = box[0] + x*xscale; xx < box[0] + box[2]; xx++) {
                    vuint32m1_t pix0 = rvv_cvtepu8_epi32(&line[xx*4]);
                    rv_ss0 = __riscv_vadd_vv_u32m1(rv_ss0, pix0, 4);
                }
            }
            rv_ss0 = __riscv_vmul_vv_u32m1(rv_ss0, rv_multiplier, 4);
            rv_ss0 = __riscv_vsrl_vx_u32m1(rv_ss0, 24, 4);
            rvv_pack_u32_to_u8x4(&((UINT32 *)imOut->image[y])[x], rv_ss0);
#else
            {
                UINT32 ss0 = amend, ss1 = amend, ss2 = amend, ss3 = amend;
                x = box[2] / xscale;
                y = box[3] / yscale;
                for (yy = box[1] + y * yscale; yy < box[1] + box[3]; yy++) {
                    UINT8 *line = (UINT8 *)imIn->image[yy];
                    for (xx = box[0] + x*xscale; xx < box[0] + box[2]; xx++) {
                        ss0 += line[xx*4 + 0];
                        ss1 += line[xx*4 + 1];
                        ss2 += line[xx*4 + 2];
                        ss3 += line[xx*4 + 3];
                    }
                }
                ss0 = (ss0 * multiplier) >> 24;
                ss1 = (ss1 * multiplier) >> 24;
                ss2 = (ss2 * multiplier) >> 24;
                ss3 = (ss3 * multiplier) >> 24;
                ((UINT32 *)imOut->image[y])[x] = scalar_pack_u32(ss0, ss1, ss2, ss3);
            }
#endif
        }
    }
}

void
ImagingReduceNxN_32bpc(
    Imaging imOut, Imaging imIn, int box[4], int xscale, int yscale) {
    /* The most general implementation for any xscale and yscale
     */
    int x, y, xx, yy;
    double multiplier = 1.0 / (yscale * xscale);

    switch (imIn->type) {
        case IMAGING_TYPE_INT32:
            for (y = 0; y < box[3] / yscale; y++) {
                int yy_from = box[1] + y * yscale;
                for (x = 0; x < box[2] / xscale; x++) {
                    int xx_from = box[0] + x * xscale;
                    double ss = 0;
                    for (yy = yy_from; yy < yy_from + yscale - 1; yy += 2) {
                        INT32 *line0 = (INT32 *)imIn->image32[yy];
                        INT32 *line1 = (INT32 *)imIn->image32[yy + 1];
                        for (xx = xx_from; xx < xx_from + xscale - 1; xx += 2) {
                            ss += line0[xx + 0] + line0[xx + 1] + line1[xx + 0] +
                                  line1[xx + 1];
                        }
                        if (xscale & 0x01) {
                            ss += line0[xx + 0] + line1[xx + 0];
                        }
                    }
                    if (yscale & 0x01) {
                        INT32 *line = (INT32 *)imIn->image32[yy];
                        for (xx = xx_from; xx < xx_from + xscale - 1; xx += 2) {
                            ss += line[xx + 0] + line[xx + 1];
                        }
                        if (xscale & 0x01) {
                            ss += line[xx + 0];
                        }
                    }
                    IMAGING_PIXEL_I(imOut, x, y) = ROUND_UP(ss * multiplier);
                }
            }
            break;

        case IMAGING_TYPE_FLOAT32:
            for (y = 0; y < box[3] / yscale; y++) {
                int yy_from = box[1] + y * yscale;
                for (x = 0; x < box[2] / xscale; x++) {
                    int xx_from = box[0] + x * xscale;
                    double ss = 0;
                    for (yy = yy_from; yy < yy_from + yscale - 1; yy += 2) {
                        FLOAT32 *line0 = (FLOAT32 *)imIn->image32[yy];
                        FLOAT32 *line1 = (FLOAT32 *)imIn->image32[yy + 1];
                        for (xx = xx_from; xx < xx_from + xscale - 1; xx += 2) {
                            ss += line0[xx + 0] + line0[xx + 1] + line1[xx + 0] +
                                  line1[xx + 1];
                        }
                        if (xscale & 0x01) {
                            ss += line0[xx + 0] + line1[xx + 0];
                        }
                    }
                    if (yscale & 0x01) {
                        FLOAT32 *line = (FLOAT32 *)imIn->image32[yy];
                        for (xx = xx_from; xx < xx_from + xscale - 1; xx += 2) {
                            ss += line[xx + 0] + line[xx + 1];
                        }
                        if (xscale & 0x01) {
                            ss += line[xx + 0];
                        }
                    }
                    IMAGING_PIXEL_F(imOut, x, y) = ss * multiplier;
                }
            }
            break;
    }
}

void
ImagingReduceCorners_32bpc(
    Imaging imOut, Imaging imIn, int box[4], int xscale, int yscale) {
    /* Fill the last row and the last column for any xscale and yscale.
     */
    int x, y, xx, yy;

    switch (imIn->type) {
        case IMAGING_TYPE_INT32:
            if (box[2] % xscale) {
                double multiplier = 1.0 / ((box[2] % xscale) * yscale);
                for (y = 0; y < box[3] / yscale; y++) {
                    int yy_from = box[1] + y * yscale;
                    double ss = 0;
                    x = box[2] / xscale;
                    for (yy = yy_from; yy < yy_from + yscale; yy++) {
                        INT32 *line = (INT32 *)imIn->image32[yy];
                        for (xx = box[0] + x * xscale; xx < box[0] + box[2]; xx++) {
                            ss += line[xx + 0];
                        }
                    }
                    IMAGING_PIXEL_I(imOut, x, y) = ROUND_UP(ss * multiplier);
                }
            }
            if (box[3] % yscale) {
                double multiplier = 1.0 / (xscale * (box[3] % yscale));
                y = box[3] / yscale;
                for (x = 0; x < box[2] / xscale; x++) {
                    int xx_from = box[0] + x * xscale;
                    double ss = 0;
                    for (yy = box[1] + y * yscale; yy < box[1] + box[3]; yy++) {
                        INT32 *line = (INT32 *)imIn->image32[yy];
                        for (xx = xx_from; xx < xx_from + xscale; xx++) {
                            ss += line[xx + 0];
                        }
                    }
                    IMAGING_PIXEL_I(imOut, x, y) = ROUND_UP(ss * multiplier);
                }
            }
            if (box[2] % xscale && box[3] % yscale) {
                double multiplier = 1.0 / ((box[2] % xscale) * (box[3] % yscale));
                double ss = 0;
                x = box[2] / xscale;
                y = box[3] / yscale;
                for (yy = box[1] + y * yscale; yy < box[1] + box[3]; yy++) {
                    INT32 *line = (INT32 *)imIn->image32[yy];
                    for (xx = box[0] + x * xscale; xx < box[0] + box[2]; xx++) {
                        ss += line[xx + 0];
                    }
                }
                IMAGING_PIXEL_I(imOut, x, y) = ROUND_UP(ss * multiplier);
            }
            break;

        case IMAGING_TYPE_FLOAT32:
            if (box[2] % xscale) {
                double multiplier = 1.0 / ((box[2] % xscale) * yscale);
                for (y = 0; y < box[3] / yscale; y++) {
                    int yy_from = box[1] + y * yscale;
                    double ss = 0;
                    x = box[2] / xscale;
                    for (yy = yy_from; yy < yy_from + yscale; yy++) {
                        FLOAT32 *line = (FLOAT32 *)imIn->image32[yy];
                        for (xx = box[0] + x * xscale; xx < box[0] + box[2]; xx++) {
                            ss += line[xx + 0];
                        }
                    }
                    IMAGING_PIXEL_F(imOut, x, y) = ss * multiplier;
                }
            }
            if (box[3] % yscale) {
                double multiplier = 1.0 / (xscale * (box[3] % yscale));
                y = box[3] / yscale;
                for (x = 0; x < box[2] / xscale; x++) {
                    int xx_from = box[0] + x * xscale;
                    double ss = 0;
                    for (yy = box[1] + y * yscale; yy < box[1] + box[3]; yy++) {
                        FLOAT32 *line = (FLOAT32 *)imIn->image32[yy];
                        for (xx = xx_from; xx < xx_from + xscale; xx++) {
                            ss += line[xx + 0];
                        }
                    }
                    IMAGING_PIXEL_F(imOut, x, y) = ss * multiplier;
                }
            }
            if (box[2] % xscale && box[3] % yscale) {
                double multiplier = 1.0 / ((box[2] % xscale) * (box[3] % yscale));
                double ss = 0;
                x = box[2] / xscale;
                y = box[3] / yscale;
                for (yy = box[1] + y * yscale; yy < box[1] + box[3]; yy++) {
                    FLOAT32 *line = (FLOAT32 *)imIn->image32[yy];
                    for (xx = box[0] + x * xscale; xx < box[0] + box[2]; xx++) {
                        ss += line[xx + 0];
                    }
                }
                IMAGING_PIXEL_F(imOut, x, y) = ss * multiplier;
            }
            break;
    }
}

Imaging
ImagingReduce(Imaging imIn, int xscale, int yscale, int box[4]) {
    ImagingSectionCookie cookie;
    Imaging imOut = NULL;

    if (strcmp(imIn->mode, "P") == 0 || strcmp(imIn->mode, "1") == 0) {
        return (Imaging)ImagingError_ModeError();
    }

    if (imIn->type == IMAGING_TYPE_SPECIAL) {
        return (Imaging)ImagingError_ModeError();
    }

    imOut = ImagingNewDirty(
        imIn->mode, (box[2] + xscale - 1) / xscale, (box[3] + yscale - 1) / yscale);
    if (!imOut) {
        return NULL;
    }

    ImagingSectionEnter(&cookie);

    switch (imIn->type) {
        case IMAGING_TYPE_UINT8:
            if (xscale == 1) {
                if (yscale == 2) {
                    ImagingReduce1x2(imOut, imIn, box);
                } else if (yscale == 3) {
                    ImagingReduce1x3(imOut, imIn, box);
                } else {
                    ImagingReduce1xN(imOut, imIn, box, yscale);
                }
            } else if (yscale == 1) {
                if (xscale == 2) {
                    ImagingReduce2x1(imOut, imIn, box);
                } else if (xscale == 3) {
                    ImagingReduce3x1(imOut, imIn, box);
                } else {
                    ImagingReduceNx1(imOut, imIn, box, xscale);
                }
            } else if (xscale == yscale && xscale <= 3) {
                if (xscale == 2) {
                    ImagingReduce2x2(imOut, imIn, box);
                } else if (xscale == 3) {
                    ImagingReduce3x3(imOut, imIn, box);
                }
            } else {
                ImagingReduceNxN(imOut, imIn, box, xscale, yscale);
            }

            ImagingReduceCorners(imOut, imIn, box, xscale, yscale);
            break;

        case IMAGING_TYPE_INT32:
        case IMAGING_TYPE_FLOAT32:
            ImagingReduceNxN_32bpc(imOut, imIn, box, xscale, yscale);

            ImagingReduceCorners_32bpc(imOut, imIn, box, xscale, yscale);
            break;
    }

    ImagingSectionLeave(&cookie);

    return imOut;
}
