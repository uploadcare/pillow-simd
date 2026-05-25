#include "Imaging.h"
#include "ImagingSIMD.h"

#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))


#if defined(__SSE4_2__)
/* ----------------------------------------------------------------
 *  SSE4.2 code path for ImagingLineBoxBlur32
 * ---------------------------------------------------------------- */
void static inline
ImagingLineBoxBlur32(UINT32 *lineOut, UINT32 *lineIn, int lastx, int radius, int edgeA,
    int edgeB, UINT32 ww, UINT32 fw)
{
    int x;
    __m128i acc;
    __m128i bulk;
    __m128i wws = _mm_set1_epi32(ww);
    __m128i fws = _mm_set1_epi32(fw);
    __m128i b23 = _mm_set1_epi32(1 << 23);
    __m128i shiftmask = _mm_set_epi8(0,0,0,0,0,0,0,0,0,0,0,0,3,7,11,15);


    #define MOVE_ACC(acc, subtract, add) \
        { \
        __m128i tmp1 = mm_cvtepu8_epi32(&lineIn[add]); \
        __m128i tmp2 = mm_cvtepu8_epi32(&lineIn[subtract]); \
        acc = _mm_sub_epi32(_mm_add_epi32(acc, tmp1), tmp2); \
        }

    #define ADD_FAR(bulk, acc, left, right) \
        { \
        __m128i tmp3 = mm_cvtepu8_epi32(&lineIn[left]); \
        __m128i tmp4 = mm_cvtepu8_epi32(&lineIn[right]); \
        __m128i tmp5 = _mm_mullo_epi32(_mm_add_epi32(tmp3, tmp4), fws); \
        __m128i tmp6 = _mm_mullo_epi32(acc, wws); \
        bulk = _mm_add_epi32(tmp5, tmp6); \
        }

    #define SAVE(x, bulk) \
        lineOut[x] = _mm_cvtsi128_si32( \
            _mm_shuffle_epi8(_mm_add_epi32(bulk, b23), shiftmask) \
        )

    acc = _mm_mullo_epi32(
        mm_cvtepu8_epi32(&lineIn[0]),
        _mm_set1_epi32(radius + 1)
    );

    for (x = 0; x < edgeA - 1; x++) {
        acc = _mm_add_epi32(acc, mm_cvtepu8_epi32(&lineIn[x]));
    }
    acc = _mm_add_epi32(acc, _mm_mullo_epi32(
        mm_cvtepu8_epi32(&lineIn[x]),
        _mm_set1_epi32(radius - edgeA + 1)
    ));


    if (edgeA <= edgeB) {
        for (x = 0; x < edgeA; x++) {
            MOVE_ACC(acc, 0, x + radius);
            ADD_FAR(bulk, acc, 0, x + radius + 1);
            SAVE(x, bulk);
        }
        for (x = edgeA; x < edgeB; x++) {
            MOVE_ACC(acc, x - radius - 1, x + radius);
            ADD_FAR(bulk, acc, x - radius - 1, x + radius + 1);
            SAVE(x, bulk);
        }
        for (x = edgeB; x <= lastx; x++) {
            MOVE_ACC(acc, x - radius - 1, lastx);
            ADD_FAR(bulk, acc, x - radius - 1, lastx);
            SAVE(x, bulk);
        }
    } else {
        for (x = 0; x < edgeB; x++) {
            MOVE_ACC(acc, 0, x + radius);
            ADD_FAR(bulk, acc, 0, x + radius + 1);
            SAVE(x, bulk);
        }
        for (x = edgeB; x < edgeA; x++) {
            MOVE_ACC(acc, 0, lastx);
            ADD_FAR(bulk, acc, 0, lastx);
            SAVE(x, bulk);
        }
        for (x = edgeA; x <= lastx; x++) {
            MOVE_ACC(acc, x - radius - 1, lastx);
            ADD_FAR(bulk, acc, x - radius - 1, lastx);
            SAVE(x, bulk);
        }
    }

#undef MOVE_ACC
#undef ADD_FAR
#undef SAVE
}

#elif defined(__riscv_vector)
/* ----------------------------------------------------------------
 *  RISC-V Vector (RVV) code path for ImagingLineBoxBlur32
 * ---------------------------------------------------------------- */
void static inline
ImagingLineBoxBlur32(UINT32 *lineOut, UINT32 *lineIn, int lastx, int radius, int edgeA,
    int edgeB, UINT32 ww, UINT32 fw)
{
    int x;
    vuint32m1_t acc;
    vuint32m1_t bulk;
    vuint32m1_t wws = __riscv_vmv_v_x_u32m1(ww, 4);
    vuint32m1_t fws = __riscv_vmv_v_x_u32m1(fw, 4);
    vuint32m1_t b23 = __riscv_vmv_v_x_u32m1(1 << 23, 4);

    #define MOVE_ACC(acc, subtract, add) \
        { \
        vuint32m1_t tmp1 = rvv_cvtepu8_epi32(&lineIn[add]); \
        vuint32m1_t tmp2 = rvv_cvtepu8_epi32(&lineIn[subtract]); \
        acc = __riscv_vsub_vv_u32m1(__riscv_vadd_vv_u32m1(acc, tmp1, 4), tmp2, 4); \
        }

    #define ADD_FAR(bulk, acc, left, right) \
        { \
        vuint32m1_t tmp3 = rvv_cvtepu8_epi32(&lineIn[left]); \
        vuint32m1_t tmp4 = rvv_cvtepu8_epi32(&lineIn[right]); \
        vuint32m1_t tmp5 = __riscv_vmul_vv_u32m1(__riscv_vadd_vv_u32m1(tmp3, tmp4, 4), fws, 4); \
        vuint32m1_t tmp6 = __riscv_vmul_vv_u32m1(acc, wws, 4); \
        bulk = __riscv_vadd_vv_u32m1(tmp5, tmp6, 4); \
        }

    #define SAVE(x, bulk) \
        { \
        vuint32m1_t rounded = __riscv_vadd_vv_u32m1(bulk, b23, 4); \
        vuint32m1_t shifted = __riscv_vsrl_vx_u32m1(rounded, 24, 4); \
        rvv_pack_u32_to_u8x4(&lineOut[x], shifted); \
        }

    acc = __riscv_vmul_vv_u32m1(
        rvv_cvtepu8_epi32(&lineIn[0]),
        __riscv_vmv_v_x_u32m1(radius + 1, 4), 4
    );

    for (x = 0; x < edgeA - 1; x++) {
        acc = __riscv_vadd_vv_u32m1(acc, rvv_cvtepu8_epi32(&lineIn[x]), 4);
    }
    acc = __riscv_vadd_vv_u32m1(acc, __riscv_vmul_vv_u32m1(
        rvv_cvtepu8_epi32(&lineIn[x]),
        __riscv_vmv_v_x_u32m1(radius - edgeA + 1, 4), 4
    ), 4);


    if (edgeA <= edgeB) {
        for (x = 0; x < edgeA; x++) {
            MOVE_ACC(acc, 0, x + radius);
            ADD_FAR(bulk, acc, 0, x + radius + 1);
            SAVE(x, bulk);
        }
        for (x = edgeA; x < edgeB; x++) {
            MOVE_ACC(acc, x - radius - 1, x + radius);
            ADD_FAR(bulk, acc, x - radius - 1, x + radius + 1);
            SAVE(x, bulk);
        }
        for (x = edgeB; x <= lastx; x++) {
            MOVE_ACC(acc, x - radius - 1, lastx);
            ADD_FAR(bulk, acc, x - radius - 1, lastx);
            SAVE(x, bulk);
        }
    } else {
        for (x = 0; x < edgeB; x++) {
            MOVE_ACC(acc, 0, x + radius);
            ADD_FAR(bulk, acc, 0, x + radius + 1);
            SAVE(x, bulk);
        }
        for (x = edgeB; x < edgeA; x++) {
            MOVE_ACC(acc, 0, lastx);
            ADD_FAR(bulk, acc, 0, lastx);
            SAVE(x, bulk);
        }
        for (x = edgeA; x <= lastx; x++) {
            MOVE_ACC(acc, x - radius - 1, lastx);
            ADD_FAR(bulk, acc, x - radius - 1, lastx);
            SAVE(x, bulk);
        }
    }

#undef MOVE_ACC
#undef ADD_FAR
#undef SAVE
}

#else
/* ----------------------------------------------------------------
 *  Scalar fallback for ImagingLineBoxBlur32
 * ---------------------------------------------------------------- */
void static inline
ImagingLineBoxBlur32(UINT32 *lineOut, UINT32 *lineIn, int lastx, int radius, int edgeA,
    int edgeB, UINT32 ww, UINT32 fw)
{
    int x, c;
    UINT32 acc[4];
    UINT32 bulk[4];
    UINT8 *pIn, *pSub, *pAdd, *pLeft, *pRight;
    UINT8 *pOut;

    #define MOVE_ACC(acc, subtract, add) \
        { \
        pAdd = (UINT8 *)&lineIn[add]; \
        pSub = (UINT8 *)&lineIn[subtract]; \
        for (c = 0; c < 4; c++) \
            acc[c] += pAdd[c] - pSub[c]; \
        }

    #define ADD_FAR(bulk, acc, left, right) \
        { \
        pLeft = (UINT8 *)&lineIn[left]; \
        pRight = (UINT8 *)&lineIn[right]; \
        for (c = 0; c < 4; c++) \
            bulk[c] = (acc[c] * ww) + (pLeft[c] + pRight[c]) * fw; \
        }

    #define SAVE(x, bulk) \
        { \
        pOut = (UINT8 *)&lineOut[x]; \
        for (c = 0; c < 4; c++) \
            pOut[c] = (UINT8)((bulk[c] + (1 << 23)) >> 24); \
        }

    pIn = (UINT8 *)&lineIn[0];
    for (c = 0; c < 4; c++)
        acc[c] = pIn[c] * (radius + 1);

    for (x = 0; x < edgeA - 1; x++) {
        pIn = (UINT8 *)&lineIn[x];
        for (c = 0; c < 4; c++)
            acc[c] += pIn[c];
    }
    pIn = (UINT8 *)&lineIn[x];
    for (c = 0; c < 4; c++)
        acc[c] += pIn[c] * (radius - edgeA + 1);


    if (edgeA <= edgeB) {
        for (x = 0; x < edgeA; x++) {
            MOVE_ACC(acc, 0, x + radius);
            ADD_FAR(bulk, acc, 0, x + radius + 1);
            SAVE(x, bulk);
        }
        for (x = edgeA; x < edgeB; x++) {
            MOVE_ACC(acc, x - radius - 1, x + radius);
            ADD_FAR(bulk, acc, x - radius - 1, x + radius + 1);
            SAVE(x, bulk);
        }
        for (x = edgeB; x <= lastx; x++) {
            MOVE_ACC(acc, x - radius - 1, lastx);
            ADD_FAR(bulk, acc, x - radius - 1, lastx);
            SAVE(x, bulk);
        }
    } else {
        for (x = 0; x < edgeB; x++) {
            MOVE_ACC(acc, 0, x + radius);
            ADD_FAR(bulk, acc, 0, x + radius + 1);
            SAVE(x, bulk);
        }
        for (x = edgeB; x < edgeA; x++) {
            MOVE_ACC(acc, 0, lastx);
            ADD_FAR(bulk, acc, 0, lastx);
            SAVE(x, bulk);
        }
        for (x = edgeA; x <= lastx; x++) {
            MOVE_ACC(acc, x - radius - 1, lastx);
            ADD_FAR(bulk, acc, x - radius - 1, lastx);
            SAVE(x, bulk);
        }
    }

#undef MOVE_ACC
#undef ADD_FAR
#undef SAVE
}
#endif /* ImagingLineBoxBlur32 arch selection */


#if defined(__SSE4_2__)
/* ----------------------------------------------------------------
 *  SSE4.2 code path for ImagingLineBoxBlurZero32
 * ---------------------------------------------------------------- */
void static inline
ImagingLineBoxBlurZero32(UINT32 *lineOut, UINT32 *lineIn, int lastx, int edgeA,
    int edgeB, UINT32 ww, UINT32 fw)
{
    int x;
    __m128i acc;
    __m128i bulk;
    __m128i wws = _mm_set1_epi32(ww);
    __m128i fws = _mm_set1_epi32(fw);
    __m128i b23 = _mm_set1_epi32(1 << 23);
    __m128i shiftmask = _mm_set_epi8(0,0,0,0,0,0,0,0,0,0,0,0,3,7,11,15);

    #define ADD_FAR(bulk, acc, left, right) \
        { \
        __m128i tmp3 = mm_cvtepu8_epi32(&lineIn[left]); \
        __m128i tmp4 = mm_cvtepu8_epi32(&lineIn[right]); \
        __m128i tmp5 = _mm_mullo_epi32(_mm_add_epi32(tmp3, tmp4), fws); \
        __m128i tmp6 = _mm_mullo_epi32(acc, wws); \
        bulk = _mm_add_epi32(tmp5, tmp6); \
        }

    #define SAVE(x, bulk) \
        lineOut[x] = _mm_cvtsi128_si32( \
            _mm_shuffle_epi8(_mm_add_epi32(bulk, b23), shiftmask) \
        )

    for (x = 0; x < edgeA; x++) {
        acc = mm_cvtepu8_epi32(&lineIn[0]);
        ADD_FAR(bulk, acc, 0, x + 1);
        SAVE(x, bulk);
    }
    for (x = edgeA; x < edgeB; x++) {
        acc = mm_cvtepu8_epi32(&lineIn[x]);
        ADD_FAR(bulk, acc, x - 1, x + 1);
        SAVE(x, bulk);
    }
    for (x = edgeB; x <= lastx; x++) {
        acc = mm_cvtepu8_epi32(&lineIn[lastx]);
        ADD_FAR(bulk, acc, x - 1, lastx);
        SAVE(x, bulk);
    }

    #undef ADD_FAR
    #undef SAVE
}

#elif defined(__riscv_vector)
/* ----------------------------------------------------------------
 *  RISC-V Vector (RVV) code path for ImagingLineBoxBlurZero32
 * ---------------------------------------------------------------- */
void static inline
ImagingLineBoxBlurZero32(UINT32 *lineOut, UINT32 *lineIn, int lastx, int edgeA,
    int edgeB, UINT32 ww, UINT32 fw)
{
    int x;
    vuint32m1_t acc;
    vuint32m1_t bulk;
    vuint32m1_t wws = __riscv_vmv_v_x_u32m1(ww, 4);
    vuint32m1_t fws = __riscv_vmv_v_x_u32m1(fw, 4);
    vuint32m1_t b23 = __riscv_vmv_v_x_u32m1(1 << 23, 4);

    #define ADD_FAR(bulk, acc, left, right) \
        { \
        vuint32m1_t tmp3 = rvv_cvtepu8_epi32(&lineIn[left]); \
        vuint32m1_t tmp4 = rvv_cvtepu8_epi32(&lineIn[right]); \
        vuint32m1_t tmp5 = __riscv_vmul_vv_u32m1(__riscv_vadd_vv_u32m1(tmp3, tmp4, 4), fws, 4); \
        vuint32m1_t tmp6 = __riscv_vmul_vv_u32m1(acc, wws, 4); \
        bulk = __riscv_vadd_vv_u32m1(tmp5, tmp6, 4); \
        }

    #define SAVE(x, bulk) \
        { \
        vuint32m1_t rounded = __riscv_vadd_vv_u32m1(bulk, b23, 4); \
        vuint32m1_t shifted = __riscv_vsrl_vx_u32m1(rounded, 24, 4); \
        rvv_pack_u32_to_u8x4(&lineOut[x], shifted); \
        }

    for (x = 0; x < edgeA; x++) {
        acc = rvv_cvtepu8_epi32(&lineIn[0]);
        ADD_FAR(bulk, acc, 0, x + 1);
        SAVE(x, bulk);
    }
    for (x = edgeA; x < edgeB; x++) {
        acc = rvv_cvtepu8_epi32(&lineIn[x]);
        ADD_FAR(bulk, acc, x - 1, x + 1);
        SAVE(x, bulk);
    }
    for (x = edgeB; x <= lastx; x++) {
        acc = rvv_cvtepu8_epi32(&lineIn[lastx]);
        ADD_FAR(bulk, acc, x - 1, lastx);
        SAVE(x, bulk);
    }

    #undef ADD_FAR
    #undef SAVE
}

#else
/* ----------------------------------------------------------------
 *  Scalar fallback for ImagingLineBoxBlurZero32
 * ---------------------------------------------------------------- */
void static inline
ImagingLineBoxBlurZero32(UINT32 *lineOut, UINT32 *lineIn, int lastx, int edgeA,
    int edgeB, UINT32 ww, UINT32 fw)
{
    int x, c;
    UINT32 acc[4];
    UINT32 bulk[4];
    UINT8 *pIn, *pLeft, *pRight;
    UINT8 *pOut;

    #define ADD_FAR(bulk, acc, left, right) \
        { \
        pLeft = (UINT8 *)&lineIn[left]; \
        pRight = (UINT8 *)&lineIn[right]; \
        for (c = 0; c < 4; c++) \
            bulk[c] = (acc[c] * ww) + (pLeft[c] + pRight[c]) * fw; \
        }

    #define SAVE(x, bulk) \
        { \
        pOut = (UINT8 *)&lineOut[x]; \
        for (c = 0; c < 4; c++) \
            pOut[c] = (UINT8)((bulk[c] + (1 << 23)) >> 24); \
        }

    for (x = 0; x < edgeA; x++) {
        pIn = (UINT8 *)&lineIn[0];
        for (c = 0; c < 4; c++)
            acc[c] = pIn[c];
        ADD_FAR(bulk, acc, 0, x + 1);
        SAVE(x, bulk);
    }
    for (x = edgeA; x < edgeB; x++) {
        pIn = (UINT8 *)&lineIn[x];
        for (c = 0; c < 4; c++)
            acc[c] = pIn[c];
        ADD_FAR(bulk, acc, x - 1, x + 1);
        SAVE(x, bulk);
    }
    for (x = edgeB; x <= lastx; x++) {
        pIn = (UINT8 *)&lineIn[lastx];
        for (c = 0; c < 4; c++)
            acc[c] = pIn[c];
        ADD_FAR(bulk, acc, x - 1, lastx);
        SAVE(x, bulk);
    }

    #undef ADD_FAR
    #undef SAVE
}
#endif /* ImagingLineBoxBlurZero32 arch selection */


void static inline
ImagingLineBoxBlur8(UINT8 *lineOut, UINT8 *lineIn, int lastx, int radius, int edgeA,
    int edgeB, UINT32 ww, UINT32 fw)
{
    int x;
    UINT32 acc;
    UINT32 bulk;

#define MOVE_ACC(acc, subtract, add) acc += lineIn[add] - lineIn[subtract];

#define ADD_FAR(bulk, acc, left, right) \
    bulk = (acc * ww) + (lineIn[left] + lineIn[right]) * fw;

#define SAVE(x, bulk) lineOut[x] = (UINT8)((bulk + (1 << 23)) >> 24)

    acc = lineIn[0] * (radius + 1);
    for (x = 0; x < edgeA - 1; x++) {
        acc += lineIn[x];
    }
    acc += lineIn[lastx] * (radius - edgeA + 1);

    if (edgeA <= edgeB) {
        for (x = 0; x < edgeA; x++) {
            MOVE_ACC(acc, 0, x + radius);
            ADD_FAR(bulk, acc, 0, x + radius + 1);
            SAVE(x, bulk);
        }
        for (x = edgeA; x < edgeB; x++) {
            MOVE_ACC(acc, x - radius - 1, x + radius);
            ADD_FAR(bulk, acc, x - radius - 1, x + radius + 1);
            SAVE(x, bulk);
        }
        for (x = edgeB; x <= lastx; x++) {
            MOVE_ACC(acc, x - radius - 1, lastx);
            ADD_FAR(bulk, acc, x - radius - 1, lastx);
            SAVE(x, bulk);
        }
    } else {
        for (x = 0; x < edgeB; x++) {
            MOVE_ACC(acc, 0, x + radius);
            ADD_FAR(bulk, acc, 0, x + radius + 1);
            SAVE(x, bulk);
        }
        for (x = edgeB; x < edgeA; x++) {
            MOVE_ACC(acc, 0, lastx);
            ADD_FAR(bulk, acc, 0, lastx);
            SAVE(x, bulk);
        }
        for (x = edgeA; x <= lastx; x++) {
            MOVE_ACC(acc, x - radius - 1, lastx);
            ADD_FAR(bulk, acc, x - radius - 1, lastx);
            SAVE(x, bulk);
        }
    }

#undef MOVE_ACC
#undef ADD_FAR
#undef SAVE
}


void static inline
ImagingLineBoxBlurZero8(UINT8 *lineOut, UINT8 *lineIn, int lastx, int edgeA,
    int edgeB, UINT32 ww, UINT32 fw)
{
    int x;
    UINT32 acc;
    UINT32 bulk;

    #define MOVE_ACC(acc, subtract, add) \
        acc += lineIn[add] - lineIn[subtract];

    #define ADD_FAR(bulk, acc, left, right) \
        bulk = (acc * ww) + (lineIn[left] + lineIn[right]) * fw;

    #define SAVE(x, bulk) \
        lineOut[x] = (UINT8)((bulk + (1 << 23)) >> 24)

    for (x = 0; x < edgeA; x++) {
        acc = lineIn[0];
        ADD_FAR(bulk, acc, 0, x + 1);
        SAVE(x, bulk);
    }
    for (x = edgeA; x < edgeB; x++) {
        acc = lineIn[x];
        ADD_FAR(bulk, acc, x - 1, x + 1);
        SAVE(x, bulk);
    }
    for (x = edgeB; x <= lastx; x++) {
        acc = lineIn[lastx];
        ADD_FAR(bulk, acc, x - 1, lastx);
        SAVE(x, bulk);
    }

    #undef MOVE_ACC
    #undef ADD_FAR
    #undef SAVE
}



Imaging
ImagingHorizontalBoxBlur(Imaging imOut, Imaging imIn, float floatRadius) {
    ImagingSectionCookie cookie;

    int y;

    int radius = (int)floatRadius;
    UINT32 ww = (UINT32)(1 << 24) / (floatRadius * 2 + 1);
    UINT32 fw = ((1 << 24) - (radius * 2 + 1) * ww) / 2;

    int edgeA = MIN(radius + 1, imIn->xsize);
    int edgeB = MAX(imIn->xsize - radius - 1, 0);

    UINT32 *lineOut = calloc(imIn->xsize, sizeof(UINT32));
    if (lineOut == NULL) {
        return ImagingError_MemoryError();
    }

    // printf(">>> radius: %d edges: %d %d, weights: %d %d\n",
    //     radius, edgeA, edgeB, ww, fw);

    ImagingSectionEnter(&cookie);

    if (imIn->image8)
    {
        if (radius) {
            for (y = 0; y < imIn->ysize; y++) {
                ImagingLineBoxBlur8(
                    (imIn == imOut ? (UINT8 *) lineOut : imOut->image8[y]),
                    imIn->image8[y],
                    imIn->xsize - 1,
                    radius, edgeA, edgeB,
                    ww, fw
                );
                if (imIn == imOut) {
                    // Commit.
                    memcpy(imOut->image8[y], lineOut, imIn->xsize);
                }
            }
        } else {
            for (y = 0; y < imIn->ysize; y++) {
                ImagingLineBoxBlurZero8(
                    (imIn == imOut ? (UINT8 *) lineOut : imOut->image8[y]),
                    imIn->image8[y],
                    imIn->xsize - 1,
                    edgeA, edgeB,
                    ww, fw
                );
                if (imIn == imOut) {
                    // Commit.
                    memcpy(imOut->image8[y], lineOut, imIn->xsize);
                }
            }
        }
    }
    else
    {
        if (radius) {
            for (y = 0; y < imIn->ysize; y++) {
                ImagingLineBoxBlur32(
                    imIn == imOut ? (UINT32 *) lineOut : (UINT32 *) imOut->image32[y],
                    (UINT32 *) imIn->image32[y],
                    imIn->xsize - 1,
                    radius, edgeA, edgeB,
                    ww, fw
                );
                if (imIn == imOut) {
                    // Commit.
                    memcpy(imOut->image32[y], lineOut, imIn->xsize * 4);
                }
            }
        } else{
            for (y = 0; y < imIn->ysize; y++) {
                ImagingLineBoxBlurZero32(
                    imIn == imOut ? (UINT32 *) lineOut : (UINT32 *) imOut->image32[y],
                    (UINT32 *) imIn->image32[y],
                    imIn->xsize - 1,
                    edgeA, edgeB,
                    ww, fw
                );
                if (imIn == imOut) {
                    // Commit.
                    memcpy(imOut->image32[y], lineOut, imIn->xsize * 4);
                }
            }
        }
    }

    ImagingSectionLeave(&cookie);

    free(lineOut);

    return imOut;
}

Imaging
ImagingBoxBlur(Imaging imOut, Imaging imIn, float radius, int n) {
    int i;
    Imaging imTransposed;

    if (n < 1) {
        return ImagingError_ValueError("number of passes must be greater than zero");
    }
    if (radius < 0) {
        return ImagingError_ValueError("radius must be >= 0");
    }

    if (strcmp(imIn->mode, imOut->mode) || imIn->type != imOut->type ||
        imIn->bands != imOut->bands || imIn->xsize != imOut->xsize ||
        imIn->ysize != imOut->ysize) {
        return ImagingError_Mismatch();
    }

    if (imIn->type != IMAGING_TYPE_UINT8) {
        return ImagingError_ModeError();
    }

    if (!(strcmp(imIn->mode, "RGB") == 0 || strcmp(imIn->mode, "RGBA") == 0 ||
          strcmp(imIn->mode, "RGBa") == 0 || strcmp(imIn->mode, "RGBX") == 0 ||
          strcmp(imIn->mode, "CMYK") == 0 || strcmp(imIn->mode, "L") == 0 ||
          strcmp(imIn->mode, "LA") == 0 || strcmp(imIn->mode, "La") == 0)) {
        return ImagingError_ModeError();
    }

    imTransposed = ImagingNewDirty(imIn->mode, imIn->ysize, imIn->xsize);
    if (!imTransposed) {
        return NULL;
    }

    /* Apply blur in one dimension.
       Use imOut as a destination at first pass,
       then use imOut as a source too. */
    ImagingHorizontalBoxBlur(imOut, imIn, radius);
    for (i = 1; i < n; i++) {
        ImagingHorizontalBoxBlur(imOut, imOut, radius);
    }
    /* Transpose result for blur in another direction. */
    ImagingTranspose(imTransposed, imOut);

    /* Reuse imTransposed as a source and destination there. */
    for (i = 0; i < n; i++) {
        ImagingHorizontalBoxBlur(imTransposed, imTransposed, radius);
    }
    /* Restore original orientation. */
    ImagingTranspose(imOut, imTransposed);

    ImagingDelete(imTransposed);

    return imOut;
}

Imaging
ImagingGaussianBlur(Imaging imOut, Imaging imIn, float radius, int passes) {
    float sigma2, L, l, a;

    sigma2 = radius * radius / passes;
    // from https://www.mia.uni-saarland.de/Publications/gwosdek-ssvm11.pdf
    // [7] Box length.
    L = sqrt(12.0 * sigma2 + 1.0);
    // [11] Integer part of box radius.
    l = floor((L - 1.0) / 2.0);
    // [14], [Fig. 2] Fractional part of box radius.
    a = (2 * l + 1) * (l * (l + 1) - 3 * sigma2);
    a /= 6 * (sigma2 - (l + 1) * (l + 1));

    return ImagingBoxBlur(imOut, imIn, l + a, passes);
}
