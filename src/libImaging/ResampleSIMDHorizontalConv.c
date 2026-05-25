void
ImagingResampleHorizontalConvolution8u4x(
    UINT32 *lineOut0, UINT32 *lineOut1, UINT32 *lineOut2, UINT32 *lineOut3,
    UINT32 *lineIn0, UINT32 *lineIn1, UINT32 *lineIn2, UINT32 *lineIn3,
    int xsize, int *xbounds, INT16 *kk, int kmax, int coefs_precision)
{
    int xmin, xmax, xx, x;
    INT16 *k;

    for (xx = 0; xx < xsize; xx++) {
        xmin = xbounds[xx * 2 + 0];
        xmax = xbounds[xx * 2 + 1];
        k = &kk[xx * kmax];
        x = 0;

#if defined(__AVX2__)
    {
        __m256i sss0, sss1;
        __m256i zero = _mm256_setzero_si256();
        __m256i initial = _mm256_set1_epi32(1 << (coefs_precision-1));
        sss0 = initial;
        sss1 = initial;

        for (; x < xmax - 3; x += 4) {
            __m256i pix, mmk0, mmk1, source;

            mmk0 = _mm256_set1_epi32(*(INT32 *) &k[x]);
            mmk1 = _mm256_set1_epi32(*(INT32 *) &k[x + 2]);

            source = _mm256_inserti128_si256(_mm256_castsi128_si256(
                _mm_loadu_si128((__m128i *) &lineIn0[x + xmin])),
                _mm_loadu_si128((__m128i *) &lineIn1[x + xmin]), 1);
            pix = _mm256_shuffle_epi8(source, _mm256_set_epi8(
                -1,7, -1,3, -1,6, -1,2, -1,5, -1,1, -1,4, -1,0,
                -1,7, -1,3, -1,6, -1,2, -1,5, -1,1, -1,4, -1,0));
            sss0 = _mm256_add_epi32(sss0, _mm256_madd_epi16(pix, mmk0));
            pix = _mm256_shuffle_epi8(source, _mm256_set_epi8(
                -1,15, -1,11, -1,14, -1,10, -1,13, -1,9, -1,12, -1,8,
                -1,15, -1,11, -1,14, -1,10, -1,13, -1,9, -1,12, -1,8));
            sss0 = _mm256_add_epi32(sss0, _mm256_madd_epi16(pix, mmk1));

            source = _mm256_inserti128_si256(_mm256_castsi128_si256(
                _mm_loadu_si128((__m128i *) &lineIn2[x + xmin])),
                _mm_loadu_si128((__m128i *) &lineIn3[x + xmin]), 1);
            pix = _mm256_shuffle_epi8(source, _mm256_set_epi8(
                -1,7, -1,3, -1,6, -1,2, -1,5, -1,1, -1,4, -1,0,
                -1,7, -1,3, -1,6, -1,2, -1,5, -1,1, -1,4, -1,0));
            sss1 = _mm256_add_epi32(sss1, _mm256_madd_epi16(pix, mmk0));
            pix = _mm256_shuffle_epi8(source, _mm256_set_epi8(
                -1,15, -1,11, -1,14, -1,10, -1,13, -1,9, -1,12, -1,8,
                -1,15, -1,11, -1,14, -1,10, -1,13, -1,9, -1,12, -1,8));
            sss1 = _mm256_add_epi32(sss1, _mm256_madd_epi16(pix, mmk1));
        }

        for (; x < xmax - 1; x += 2) {
            __m256i pix, mmk;

            mmk = _mm256_set1_epi32(*(INT32 *) &k[x]);

            pix = _mm256_inserti128_si256(_mm256_castsi128_si256(
                _mm_loadl_epi64((__m128i *) &lineIn0[x + xmin])),
                _mm_loadl_epi64((__m128i *) &lineIn1[x + xmin]), 1);
            pix = _mm256_shuffle_epi8(pix, _mm256_set_epi8(
                -1,7, -1,3, -1,6, -1,2, -1,5, -1,1, -1,4, -1,0,
                -1,7, -1,3, -1,6, -1,2, -1,5, -1,1, -1,4, -1,0));
            sss0 = _mm256_add_epi32(sss0, _mm256_madd_epi16(pix, mmk));

            pix = _mm256_inserti128_si256(_mm256_castsi128_si256(
                _mm_loadl_epi64((__m128i *) &lineIn2[x + xmin])),
                _mm_loadl_epi64((__m128i *) &lineIn3[x + xmin]), 1);
            pix = _mm256_shuffle_epi8(pix, _mm256_set_epi8(
                -1,7, -1,3, -1,6, -1,2, -1,5, -1,1, -1,4, -1,0,
                -1,7, -1,3, -1,6, -1,2, -1,5, -1,1, -1,4, -1,0));
            sss1 = _mm256_add_epi32(sss1, _mm256_madd_epi16(pix, mmk));
        }

        for (; x < xmax; x ++) {
            __m256i pix, mmk;

            // [16] xx k0 xx k0 xx k0 xx k0 xx k0 xx k0 xx k0 xx k0
            mmk = _mm256_set1_epi32(k[x]);

            // [16] xx a0 xx b0 xx g0 xx r0 xx a0 xx b0 xx g0 xx r0
            pix = _mm256_inserti128_si256(_mm256_castsi128_si256(
                mm_cvtepu8_epi32(&lineIn0[x + xmin])),
                mm_cvtepu8_epi32(&lineIn1[x + xmin]), 1);
            sss0 = _mm256_add_epi32(sss0, _mm256_madd_epi16(pix, mmk));

            pix = _mm256_inserti128_si256(_mm256_castsi128_si256(
                mm_cvtepu8_epi32(&lineIn2[x + xmin])),
                mm_cvtepu8_epi32(&lineIn3[x + xmin]), 1);
            sss1 = _mm256_add_epi32(sss1, _mm256_madd_epi16(pix, mmk));
        }

        sss0 = _mm256_srai_epi32(sss0, coefs_precision);
        sss1 = _mm256_srai_epi32(sss1, coefs_precision);
        sss0 = _mm256_packs_epi32(sss0, zero);
        sss1 = _mm256_packs_epi32(sss1, zero);
        sss0 = _mm256_packus_epi16(sss0, zero);
        sss1 = _mm256_packus_epi16(sss1, zero);
        lineOut0[xx] = _mm_cvtsi128_si32(_mm256_extracti128_si256(sss0, 0));
        lineOut1[xx] = _mm_cvtsi128_si32(_mm256_extracti128_si256(sss0, 1));
        lineOut2[xx] = _mm_cvtsi128_si32(_mm256_extracti128_si256(sss1, 0));
        lineOut3[xx] = _mm_cvtsi128_si32(_mm256_extracti128_si256(sss1, 1));
    }
#elif defined(__SSE4_2__)
    {
        __m128i sss0, sss1, sss2, sss3;
        __m128i initial = _mm_set1_epi32(1 << (coefs_precision-1));
        sss0 = initial;
        sss1 = initial;
        sss2 = initial;
        sss3 = initial;

        for (; x < xmax - 3; x += 4) {
            __m128i pix, mmk_lo, mmk_hi, source;
            __m128i mask_lo = _mm_set_epi8(
                -1,7, -1,3, -1,6, -1,2, -1,5, -1,1, -1,4, -1,0);
            __m128i mask_hi = _mm_set_epi8(
                -1,15, -1,11, -1,14, -1,10, -1,13, -1,9, -1,12, -1,8);

            mmk_lo = _mm_set1_epi32(*(INT32 *) &k[x]);
            mmk_hi = _mm_set1_epi32(*(INT32 *) &k[x + 2]);

            // [8] a3 b3 g3 r3 a2 b2 g2 r2 a1 b1 g1 r1 a0 b0 g0 r0
            source = _mm_loadu_si128((__m128i *) &lineIn0[x + xmin]);
            // [16] a1 a0 b1 b0 g1 g0 r1 r0
            pix = _mm_shuffle_epi8(source, mask_lo);
            sss0 = _mm_add_epi32(sss0, _mm_madd_epi16(pix, mmk_lo));
            // [16] a3 a2 b3 b2 g3 g2 r3 r2
            pix = _mm_shuffle_epi8(source, mask_hi);
            sss0 = _mm_add_epi32(sss0, _mm_madd_epi16(pix, mmk_hi));

            source = _mm_loadu_si128((__m128i *) &lineIn1[x + xmin]);
            pix = _mm_shuffle_epi8(source, mask_lo);
            sss1 = _mm_add_epi32(sss1, _mm_madd_epi16(pix, mmk_lo));
            pix = _mm_shuffle_epi8(source, mask_hi);
            sss1 = _mm_add_epi32(sss1, _mm_madd_epi16(pix, mmk_hi));

            source = _mm_loadu_si128((__m128i *) &lineIn2[x + xmin]);
            pix = _mm_shuffle_epi8(source, mask_lo);
            sss2 = _mm_add_epi32(sss2, _mm_madd_epi16(pix, mmk_lo));
            pix = _mm_shuffle_epi8(source, mask_hi);
            sss2 = _mm_add_epi32(sss2, _mm_madd_epi16(pix, mmk_hi));

            source = _mm_loadu_si128((__m128i *) &lineIn3[x + xmin]);
            pix = _mm_shuffle_epi8(source, mask_lo);
            sss3 = _mm_add_epi32(sss3, _mm_madd_epi16(pix, mmk_lo));
            pix = _mm_shuffle_epi8(source, mask_hi);
            sss3 = _mm_add_epi32(sss3, _mm_madd_epi16(pix, mmk_hi));
        }

        for (; x < xmax - 1; x += 2) {
            __m128i pix, mmk;
            __m128i mask = _mm_set_epi8(
                -1,7, -1,3, -1,6, -1,2, -1,5, -1,1, -1,4, -1,0);

            // [16] k1 k0 k1 k0 k1 k0 k1 k0
            mmk = _mm_set1_epi32(*(INT32 *) &k[x]);

            // [8] x x x x x x x x a1 b1 g1 r1 a0 b0 g0 r0
            pix = _mm_loadl_epi64((__m128i *) &lineIn0[x + xmin]);
            // [16] a1 a0 b1 b0 g1 g0 r1 r0
            pix = _mm_shuffle_epi8(pix, mask);
            sss0 = _mm_add_epi32(sss0, _mm_madd_epi16(pix, mmk));

            pix = _mm_loadl_epi64((__m128i *) &lineIn1[x + xmin]);
            pix = _mm_shuffle_epi8(pix, mask);
            sss1 = _mm_add_epi32(sss1, _mm_madd_epi16(pix, mmk));

            pix = _mm_loadl_epi64((__m128i *) &lineIn2[x + xmin]);
            pix = _mm_shuffle_epi8(pix, mask);
            sss2 = _mm_add_epi32(sss2, _mm_madd_epi16(pix, mmk));

            pix = _mm_loadl_epi64((__m128i *) &lineIn3[x + xmin]);
            pix = _mm_shuffle_epi8(pix, mask);
            sss3 = _mm_add_epi32(sss3, _mm_madd_epi16(pix, mmk));
        }

        for (; x < xmax; x ++) {
            __m128i pix, mmk;
            // [16] xx k0 xx k0 xx k0 xx k0
            mmk = _mm_set1_epi32(k[x]);
            // [16] xx a0 xx b0 xx g0 xx r0
            pix = mm_cvtepu8_epi32(&lineIn0[x + xmin]);
            sss0 = _mm_add_epi32(sss0, _mm_madd_epi16(pix, mmk));

            pix = mm_cvtepu8_epi32(&lineIn1[x + xmin]);
            sss1 = _mm_add_epi32(sss1, _mm_madd_epi16(pix, mmk));

            pix = mm_cvtepu8_epi32(&lineIn2[x + xmin]);
            sss2 = _mm_add_epi32(sss2, _mm_madd_epi16(pix, mmk));

            pix = mm_cvtepu8_epi32(&lineIn3[x + xmin]);
            sss3 = _mm_add_epi32(sss3, _mm_madd_epi16(pix, mmk));
        }

        sss0 = _mm_srai_epi32(sss0, coefs_precision);
        sss1 = _mm_srai_epi32(sss1, coefs_precision);
        sss2 = _mm_srai_epi32(sss2, coefs_precision);
        sss3 = _mm_srai_epi32(sss3, coefs_precision);
        sss0 = _mm_packs_epi32(sss0, sss0);
        sss1 = _mm_packs_epi32(sss1, sss1);
        sss2 = _mm_packs_epi32(sss2, sss2);
        sss3 = _mm_packs_epi32(sss3, sss3);
        lineOut0[xx] = _mm_cvtsi128_si32(_mm_packus_epi16(sss0, sss0));
        lineOut1[xx] = _mm_cvtsi128_si32(_mm_packus_epi16(sss1, sss1));
        lineOut2[xx] = _mm_cvtsi128_si32(_mm_packus_epi16(sss2, sss2));
        lineOut3[xx] = _mm_cvtsi128_si32(_mm_packus_epi16(sss3, sss3));
    }
#elif defined(__riscv_vector)
    {
        /* RVV: merge 4 rows into a single i32m4 accumulator (4 rows x 4 channels = 16 elements).
         * Each kernel tap loads 1 pixel from each of the 4 rows, giving 16 bytes total.
         *
         * Optimization: instead of copying into a buffer, load each row's pixel
         * as a u32 scalar and build the combined vector with vmv + vslideup.
         * Also unroll 2 taps at a time when possible.
         */
        size_t vl = __riscv_vsetvl_e32m4(16);
        vint32m4_t sss = __riscv_vmv_v_x_i32m4(1 << (coefs_precision - 1), vl);

        /* 2-tap unrolled loop: process 2 kernel coefficients per iteration */
        for (; x < xmax - 1; x += 2) {
            int16_t coeff0 = k[x];
            int16_t coeff1 = k[x + 1];

            /* Build pixel vector for tap 0 using vslideup (no buffer copy) */
            vuint32m1_t p0_u32 = __riscv_vmv_v_x_u32m1(lineIn0[x + xmin], 4);
            p0_u32 = __riscv_vslideup_vx_u32m1(p0_u32,
                __riscv_vmv_v_x_u32m1(lineIn1[x + xmin], 4), 1, 4);
            p0_u32 = __riscv_vslideup_vx_u32m1(p0_u32,
                __riscv_vmv_v_x_u32m1(lineIn2[x + xmin], 4), 2, 4);
            p0_u32 = __riscv_vslideup_vx_u32m1(p0_u32,
                __riscv_vmv_v_x_u32m1(lineIn3[x + xmin], 4), 3, 4);
            /* Reinterpret as 16 x u8 */
            vuint8m1_t pu8_0 = __riscv_vreinterpret_v_u32m1_u8m1(p0_u32);
            vuint16m2_t pu16_0 = __riscv_vzext_vf2_u16m2(pu8_0, vl);
            vint16m2_t pi16_0 = __riscv_vreinterpret_v_u16m2_i16m2(pu16_0);
            sss = __riscv_vadd_vv_i32m4(sss,
                __riscv_vwmul_vx_i32m4(pi16_0, coeff0, vl), vl);

            /* Build pixel vector for tap 1 */
            vuint32m1_t p1_u32 = __riscv_vmv_v_x_u32m1(lineIn0[x + 1 + xmin], 4);
            p1_u32 = __riscv_vslideup_vx_u32m1(p1_u32,
                __riscv_vmv_v_x_u32m1(lineIn1[x + 1 + xmin], 4), 1, 4);
            p1_u32 = __riscv_vslideup_vx_u32m1(p1_u32,
                __riscv_vmv_v_x_u32m1(lineIn2[x + 1 + xmin], 4), 2, 4);
            p1_u32 = __riscv_vslideup_vx_u32m1(p1_u32,
                __riscv_vmv_v_x_u32m1(lineIn3[x + 1 + xmin], 4), 3, 4);
            vuint8m1_t pu8_1 = __riscv_vreinterpret_v_u32m1_u8m1(p1_u32);
            vuint16m2_t pu16_1 = __riscv_vzext_vf2_u16m2(pu8_1, vl);
            vint16m2_t pi16_1 = __riscv_vreinterpret_v_u16m2_i16m2(pu16_1);
            sss = __riscv_vadd_vv_i32m4(sss,
                __riscv_vwmul_vx_i32m4(pi16_1, coeff1, vl), vl);
        }

        /* Handle remaining single tap */
        for (; x < xmax; x++) {
            int16_t coeff = k[x];

            vuint32m1_t p_u32 = __riscv_vmv_v_x_u32m1(lineIn0[x + xmin], 4);
            p_u32 = __riscv_vslideup_vx_u32m1(p_u32,
                __riscv_vmv_v_x_u32m1(lineIn1[x + xmin], 4), 1, 4);
            p_u32 = __riscv_vslideup_vx_u32m1(p_u32,
                __riscv_vmv_v_x_u32m1(lineIn2[x + xmin], 4), 2, 4);
            p_u32 = __riscv_vslideup_vx_u32m1(p_u32,
                __riscv_vmv_v_x_u32m1(lineIn3[x + xmin], 4), 3, 4);
            vuint8m1_t pu8 = __riscv_vreinterpret_v_u32m1_u8m1(p_u32);
            vuint16m2_t pu16 = __riscv_vzext_vf2_u16m2(pu8, vl);
            vint16m2_t pi16 = __riscv_vreinterpret_v_u16m2_i16m2(pu16);
            sss = __riscv_vadd_vv_i32m4(sss,
                __riscv_vwmul_vx_i32m4(pi16, coeff, vl), vl);
        }

        /* Arithmetic right shift */
        sss = __riscv_vsra_vx_i32m4(sss, coefs_precision, vl);

        /* Clamp to [0, 255] and narrow i32 -> u8 */
        vint32m4_t zero_v = __riscv_vmv_v_x_i32m4(0, vl);
        vint32m4_t max255 = __riscv_vmv_v_x_i32m4(255, vl);
        sss = __riscv_vmax_vv_i32m4(sss, zero_v, vl);
        sss = __riscv_vmin_vv_i32m4(sss, max255, vl);

        vuint32m4_t usss = __riscv_vreinterpret_v_i32m4_u32m4(sss);
        vuint16m2_t n16 = RVV_VNCLIPU(u16m2, usss, 0, vl);
        vuint8m1_t n8 = RVV_VNCLIPU(u8m1, n16, 0, vl);

        /* Store: extract 4 bytes per row via reinterpret to u32 */
        vuint32m1_t out_u32 = __riscv_vreinterpret_v_u8m1_u32m1(n8);
        lineOut0[xx] = __riscv_vmv_x_s_u32m1_u32(out_u32);
        lineOut1[xx] = __riscv_vmv_x_s_u32m1_u32(
            __riscv_vslidedown_vx_u32m1(out_u32, 1, 4));
        lineOut2[xx] = __riscv_vmv_x_s_u32m1_u32(
            __riscv_vslidedown_vx_u32m1(out_u32, 2, 4));
        lineOut3[xx] = __riscv_vmv_x_s_u32m1_u32(
            __riscv_vslidedown_vx_u32m1(out_u32, 3, 4));
    }
#else
    {
        /* Scalar fallback */
        INT32 ss0, ss1, ss2, ss3;
        int ch;
        UINT8 *pix0, *pix1, *pix2, *pix3;
        UINT8 *out0 = (UINT8 *)&lineOut0[xx];
        UINT8 *out1 = (UINT8 *)&lineOut1[xx];
        UINT8 *out2 = (UINT8 *)&lineOut2[xx];
        UINT8 *out3 = (UINT8 *)&lineOut3[xx];
        INT32 initial = 1 << (coefs_precision - 1);

        for (ch = 0; ch < 4; ch++) {
            ss0 = initial;
            ss1 = initial;
            ss2 = initial;
            ss3 = initial;
            for (x = 0; x < xmax; x++) {
                pix0 = (UINT8 *)&lineIn0[x + xmin];
                pix1 = (UINT8 *)&lineIn1[x + xmin];
                pix2 = (UINT8 *)&lineIn2[x + xmin];
                pix3 = (UINT8 *)&lineIn3[x + xmin];
                ss0 += (INT32)pix0[ch] * k[x];
                ss1 += (INT32)pix1[ch] * k[x];
                ss2 += (INT32)pix2[ch] * k[x];
                ss3 += (INT32)pix3[ch] * k[x];
            }
            ss0 = ss0 >> coefs_precision;
            ss1 = ss1 >> coefs_precision;
            ss2 = ss2 >> coefs_precision;
            ss3 = ss3 >> coefs_precision;
            out0[ch] = (ss0 < 0) ? 0 : (ss0 > 255) ? 255 : (UINT8)ss0;
            out1[ch] = (ss1 < 0) ? 0 : (ss1 > 255) ? 255 : (UINT8)ss1;
            out2[ch] = (ss2 < 0) ? 0 : (ss2 > 255) ? 255 : (UINT8)ss2;
            out3[ch] = (ss3 < 0) ? 0 : (ss3 > 255) ? 255 : (UINT8)ss3;
        }
    }
#endif

    }
}


void
ImagingResampleHorizontalConvolution8u(UINT32 *lineOut, UINT32 *lineIn,
    int xsize, int *xbounds, INT16 *kk, int kmax, int coefs_precision)
{
    int xmin, xmax, xx, x;
    INT16 *k;

    for (xx = 0; xx < xsize; xx++) {
        xmin = xbounds[xx * 2 + 0];
        xmax = xbounds[xx * 2 + 1];
        k = &kk[xx * kmax];
        x = 0;

#if defined(__AVX2__)

    {
        __m128i sss;

        if (xmax < 8) {
            sss = _mm_set1_epi32(1 << (coefs_precision-1));
        } else {

            // Lower part will be added to higher, use only half of the error
            __m256i sss256 = _mm256_set1_epi32(1 << (coefs_precision-2));

            for (; x < xmax - 7; x += 8) {
                __m256i pix, mmk, source;
                __m128i tmp = _mm_loadu_si128((__m128i *) &k[x]);
                __m256i ksource = _mm256_insertf128_si256(
                    _mm256_castsi128_si256(tmp), tmp, 1);

                source = _mm256_loadu_si256((__m256i *) &lineIn[x + xmin]);

                pix = _mm256_shuffle_epi8(source, _mm256_set_epi8(
                    -1,7, -1,3, -1,6, -1,2, -1,5, -1,1, -1,4, -1,0,
                    -1,7, -1,3, -1,6, -1,2, -1,5, -1,1, -1,4, -1,0));
                mmk = _mm256_shuffle_epi8(ksource, _mm256_set_epi8(
                    11,10, 9,8, 11,10, 9,8, 11,10, 9,8, 11,10, 9,8,
                    3,2, 1,0, 3,2, 1,0, 3,2, 1,0, 3,2, 1,0));
                sss256 = _mm256_add_epi32(sss256, _mm256_madd_epi16(pix, mmk));

                pix = _mm256_shuffle_epi8(source, _mm256_set_epi8(
                    -1,15, -1,11, -1,14, -1,10, -1,13, -1,9, -1,12, -1,8,
                    -1,15, -1,11, -1,14, -1,10, -1,13, -1,9, -1,12, -1,8));
                mmk = _mm256_shuffle_epi8(ksource, _mm256_set_epi8(
                    15,14, 13,12, 15,14, 13,12, 15,14, 13,12, 15,14, 13,12,
                    7,6, 5,4, 7,6, 5,4, 7,6, 5,4, 7,6, 5,4));
                sss256 = _mm256_add_epi32(sss256, _mm256_madd_epi16(pix, mmk));
            }

            for (; x < xmax - 3; x += 4) {
                __m256i pix, mmk, source;
                __m128i tmp = _mm_loadl_epi64((__m128i *) &k[x]);
                __m256i ksource = _mm256_insertf128_si256(
                    _mm256_castsi128_si256(tmp), tmp, 1);

                tmp = _mm_loadu_si128((__m128i *) &lineIn[x + xmin]);
                source = _mm256_insertf128_si256(
                    _mm256_castsi128_si256(tmp), tmp, 1);

                pix = _mm256_shuffle_epi8(source, _mm256_set_epi8(
                    -1,15, -1,11, -1,14, -1,10, -1,13, -1,9, -1,12, -1,8,
                    -1,7, -1,3, -1,6, -1,2, -1,5, -1,1, -1,4, -1,0));
                mmk = _mm256_shuffle_epi8(ksource, _mm256_set_epi8(
                    7,6, 5,4, 7,6, 5,4, 7,6, 5,4, 7,6, 5,4,
                    3,2, 1,0, 3,2, 1,0, 3,2, 1,0, 3,2, 1,0));
                sss256 = _mm256_add_epi32(sss256, _mm256_madd_epi16(pix, mmk));
            }

            sss = _mm_add_epi32(
                _mm256_extracti128_si256(sss256, 0),
                _mm256_extracti128_si256(sss256, 1)
            );

        }

        for (; x < xmax - 1; x += 2) {
            __m128i mmk = _mm_set1_epi32(*(INT32 *) &k[x]);
            __m128i source = _mm_loadl_epi64((__m128i *) &lineIn[x + xmin]);
            __m128i pix = _mm_shuffle_epi8(source, _mm_set_epi8(
                -1,7, -1,3, -1,6, -1,2, -1,5, -1,1, -1,4, -1,0));
            sss = _mm_add_epi32(sss, _mm_madd_epi16(pix, mmk));
        }

        for (; x < xmax; x ++) {
            __m128i pix = mm_cvtepu8_epi32(&lineIn[x + xmin]);
            __m128i mmk = _mm_set1_epi32(k[x]);
            sss = _mm_add_epi32(sss, _mm_madd_epi16(pix, mmk));
        }
        sss = _mm_srai_epi32(sss, coefs_precision);
        sss = _mm_packs_epi32(sss, sss);
        lineOut[xx] = _mm_cvtsi128_si32(_mm_packus_epi16(sss, sss));
    }

#elif defined(__SSE4_2__)

    {
        __m128i sss;

        sss = _mm_set1_epi32(1 << (coefs_precision-1));

        for (; x < xmax - 7; x += 8) {
            __m128i pix, mmk, source;
            __m128i ksource = _mm_loadu_si128((__m128i *) &k[x]);

            source = _mm_loadu_si128((__m128i *) &lineIn[x + xmin]);

            pix = _mm_shuffle_epi8(source, _mm_set_epi8(
                -1,11, -1,3, -1,10, -1,2, -1,9, -1,1, -1,8, -1,0));
            mmk = _mm_shuffle_epi8(ksource, _mm_set_epi8(
                5,4, 1,0, 5,4, 1,0, 5,4, 1,0, 5,4, 1,0));
            sss = _mm_add_epi32(sss, _mm_madd_epi16(pix, mmk));

            pix = _mm_shuffle_epi8(source, _mm_set_epi8(
                -1,15, -1,7, -1,14, -1,6, -1,13, -1,5, -1,12, -1,4));
            mmk = _mm_shuffle_epi8(ksource, _mm_set_epi8(
                7,6, 3,2, 7,6, 3,2, 7,6, 3,2, 7,6, 3,2));
            sss = _mm_add_epi32(sss, _mm_madd_epi16(pix, mmk));

            source = _mm_loadu_si128((__m128i *) &lineIn[x + 4 + xmin]);

            pix = _mm_shuffle_epi8(source, _mm_set_epi8(
                -1,11, -1,3, -1,10, -1,2, -1,9, -1,1, -1,8, -1,0));
            mmk = _mm_shuffle_epi8(ksource, _mm_set_epi8(
                13,12, 9,8, 13,12, 9,8, 13,12, 9,8, 13,12, 9,8));
            sss = _mm_add_epi32(sss, _mm_madd_epi16(pix, mmk));

            pix = _mm_shuffle_epi8(source, _mm_set_epi8(
                -1,15, -1,7, -1,14, -1,6, -1,13, -1,5, -1,12, -1,4));
            mmk = _mm_shuffle_epi8(ksource, _mm_set_epi8(
                15,14, 11,10, 15,14, 11,10, 15,14, 11,10, 15,14, 11,10));
            sss = _mm_add_epi32(sss, _mm_madd_epi16(pix, mmk));
        }

        for (; x < xmax - 3; x += 4) {
            __m128i pix, mmk;
            __m128i source = _mm_loadu_si128((__m128i *) &lineIn[x + xmin]);
            __m128i ksource = _mm_loadl_epi64((__m128i *) &k[x]);

            pix = _mm_shuffle_epi8(source, _mm_set_epi8(
                -1,11, -1,3, -1,10, -1,2, -1,9, -1,1, -1,8, -1,0));
            mmk = _mm_shuffle_epi8(ksource, _mm_set_epi8(
                5,4, 1,0, 5,4, 1,0, 5,4, 1,0, 5,4, 1,0));
            sss = _mm_add_epi32(sss, _mm_madd_epi16(pix, mmk));

            pix = _mm_shuffle_epi8(source, _mm_set_epi8(
                -1,15, -1,7, -1,14, -1,6, -1,13, -1,5, -1,12, -1,4));
            mmk = _mm_shuffle_epi8(ksource, _mm_set_epi8(
                7,6, 3,2, 7,6, 3,2, 7,6, 3,2, 7,6, 3,2));
            sss = _mm_add_epi32(sss, _mm_madd_epi16(pix, mmk));
        }

        for (; x < xmax - 1; x += 2) {
            __m128i mmk = _mm_set1_epi32(*(INT32 *) &k[x]);
            __m128i source = _mm_loadl_epi64((__m128i *) &lineIn[x + xmin]);
            __m128i pix = _mm_shuffle_epi8(source, _mm_set_epi8(
                -1,7, -1,3, -1,6, -1,2, -1,5, -1,1, -1,4, -1,0));
            sss = _mm_add_epi32(sss, _mm_madd_epi16(pix, mmk));
        }

        for (; x < xmax; x ++) {
            __m128i pix = mm_cvtepu8_epi32(&lineIn[x + xmin]);
            __m128i mmk = _mm_set1_epi32(k[x]);
            sss = _mm_add_epi32(sss, _mm_madd_epi16(pix, mmk));
        }
        sss = _mm_srai_epi32(sss, coefs_precision);
        sss = _mm_packs_epi32(sss, sss);
        lineOut[xx] = _mm_cvtsi128_si32(_mm_packus_epi16(sss, sss));
    }

#elif defined(__riscv_vector)

    {
        /* RVV: single-pixel horizontal convolution with vl=4.
         * Horizontal conv cannot vectorize across output pixels because each
         * reads from a different x-offset. We process 4 channels of 1 pixel.
         *
         * 2-tap unrolled loop to reduce loop overhead.
         */
        vint32m1_t sss = __riscv_vmv_v_x_i32m1(1 << (coefs_precision - 1), 4);

        for (; x < xmax - 1; x += 2) {
            int16_t coeff0 = k[x];
            int16_t coeff1 = k[x + 1];
            UINT8 *src0 = (UINT8 *)&lineIn[x + xmin];
            UINT8 *src1 = (UINT8 *)&lineIn[x + 1 + xmin];

            vuint8mf4_t pu8_0 = __riscv_vle8_v_u8mf4(src0, 4);
            vuint16mf2_t pu16_0 = __riscv_vzext_vf2_u16mf2(pu8_0, 4);
            vint16mf2_t pi16_0 = __riscv_vreinterpret_v_u16mf2_i16mf2(pu16_0);
            sss = __riscv_vadd_vv_i32m1(sss,
                __riscv_vwmul_vx_i32m1(pi16_0, coeff0, 4), 4);

            vuint8mf4_t pu8_1 = __riscv_vle8_v_u8mf4(src1, 4);
            vuint16mf2_t pu16_1 = __riscv_vzext_vf2_u16mf2(pu8_1, 4);
            vint16mf2_t pi16_1 = __riscv_vreinterpret_v_u16mf2_i16mf2(pu16_1);
            sss = __riscv_vadd_vv_i32m1(sss,
                __riscv_vwmul_vx_i32m1(pi16_1, coeff1, 4), 4);
        }

        for (; x < xmax; x++) {
            int16_t coeff = k[x];
            UINT8 *src = (UINT8 *)&lineIn[x + xmin];

            vuint8mf4_t pu8 = __riscv_vle8_v_u8mf4(src, 4);
            vuint16mf2_t pu16 = __riscv_vzext_vf2_u16mf2(pu8, 4);
            vint16mf2_t pi16 = __riscv_vreinterpret_v_u16mf2_i16mf2(pu16);
            sss = __riscv_vadd_vv_i32m1(sss,
                __riscv_vwmul_vx_i32m1(pi16, coeff, 4), 4);
        }

        sss = __riscv_vsra_vx_i32m1(sss, coefs_precision, 4);
        rvv_pack_i32_to_u8x4(&lineOut[xx], sss);
    }

#else

    {
        /* Scalar fallback */
        INT32 ss;
        int ch;
        UINT8 *out = (UINT8 *)&lineOut[xx];
        INT32 initial = 1 << (coefs_precision - 1);

        for (ch = 0; ch < 4; ch++) {
            ss = initial;
            for (x = 0; x < xmax; x++) {
                UINT8 *pix = (UINT8 *)&lineIn[x + xmin];
                ss += (INT32)pix[ch] * k[x];
            }
            ss = ss >> coefs_precision;
            out[ch] = (ss < 0) ? 0 : (ss > 255) ? 255 : (UINT8)ss;
        }
    }

#endif

    }
}
