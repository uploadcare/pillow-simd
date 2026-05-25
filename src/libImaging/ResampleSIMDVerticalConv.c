
void
ImagingResampleVerticalConvolution8u(UINT32 *lineOut, Imaging imIn,
    int xmin, int xmax, INT16 *k, int coefs_precision)
{
    int x;
    int xx = 0;
    int xsize = imIn->xsize;

#if defined(__AVX2__)

    __m128i initial = _mm_set1_epi32(1 << (coefs_precision-1));
    __m256i initial_256 = _mm256_set1_epi32(1 << (coefs_precision-1));

    for (; xx < xsize - 7; xx += 8) {
        __m256i sss0 = initial_256;
        __m256i sss1 = initial_256;
        __m256i sss2 = initial_256;
        __m256i sss3 = initial_256;
        x = 0;
        for (; x < xmax - 1; x += 2) {
            __m256i source, source1, source2;
            __m256i pix, mmk;

            // Load two coefficients at once
            mmk = _mm256_set1_epi32(*(INT32 *) &k[x]);

            source1 = _mm256_loadu_si256(  // top line
                (__m256i *) &imIn->image32[x + xmin][xx]);
            source2 = _mm256_loadu_si256(  // bottom line
                (__m256i *) &imIn->image32[x + 1 + xmin][xx]);

            source = _mm256_unpacklo_epi8(source1, source2);
            pix = _mm256_unpacklo_epi8(source, _mm256_setzero_si256());
            sss0 = _mm256_add_epi32(sss0, _mm256_madd_epi16(pix, mmk));
            pix = _mm256_unpackhi_epi8(source, _mm256_setzero_si256());
            sss1 = _mm256_add_epi32(sss1, _mm256_madd_epi16(pix, mmk));

            source = _mm256_unpackhi_epi8(source1, source2);
            pix = _mm256_unpacklo_epi8(source, _mm256_setzero_si256());
            sss2 = _mm256_add_epi32(sss2, _mm256_madd_epi16(pix, mmk));
            pix = _mm256_unpackhi_epi8(source, _mm256_setzero_si256());
            sss3 = _mm256_add_epi32(sss3, _mm256_madd_epi16(pix, mmk));
        }
        for (; x < xmax; x += 1) {
            __m256i source, source1, pix, mmk;
            mmk = _mm256_set1_epi32(k[x]);

            source1 = _mm256_loadu_si256(  // top line
                (__m256i *) &imIn->image32[x + xmin][xx]);

            source = _mm256_unpacklo_epi8(source1, _mm256_setzero_si256());
            pix = _mm256_unpacklo_epi8(source, _mm256_setzero_si256());
            sss0 = _mm256_add_epi32(sss0, _mm256_madd_epi16(pix, mmk));
            pix = _mm256_unpackhi_epi8(source, _mm256_setzero_si256());
            sss1 = _mm256_add_epi32(sss1, _mm256_madd_epi16(pix, mmk));

            source = _mm256_unpackhi_epi8(source1, _mm256_setzero_si256());
            pix = _mm256_unpacklo_epi8(source, _mm256_setzero_si256());
            sss2 = _mm256_add_epi32(sss2, _mm256_madd_epi16(pix, mmk));
            pix = _mm256_unpackhi_epi8(source, _mm256_setzero_si256());
            sss3 = _mm256_add_epi32(sss3, _mm256_madd_epi16(pix, mmk));
        }
        sss0 = _mm256_srai_epi32(sss0, coefs_precision);
        sss1 = _mm256_srai_epi32(sss1, coefs_precision);
        sss2 = _mm256_srai_epi32(sss2, coefs_precision);
        sss3 = _mm256_srai_epi32(sss3, coefs_precision);

        sss0 = _mm256_packs_epi32(sss0, sss1);
        sss2 = _mm256_packs_epi32(sss2, sss3);
        sss0 = _mm256_packus_epi16(sss0, sss2);
        _mm256_storeu_si256((__m256i *) &lineOut[xx], sss0);
    }

#elif defined(__SSE4_2__)

    __m128i initial = _mm_set1_epi32(1 << (coefs_precision-1));

    for (; xx < xsize - 7; xx += 8) {
        __m128i sss0 = initial;
        __m128i sss1 = initial;
        __m128i sss2 = initial;
        __m128i sss3 = initial;
        __m128i sss4 = initial;
        __m128i sss5 = initial;
        __m128i sss6 = initial;
        __m128i sss7 = initial;
        x = 0;
        for (; x < xmax - 1; x += 2) {
            __m128i source, source1, source2;
            __m128i pix, mmk;

            // Load two coefficients at once
            mmk = _mm_set1_epi32(*(INT32 *) &k[x]);

            source1 = _mm_loadu_si128(  // top line
                (__m128i *) &imIn->image32[x + xmin][xx]);
            source2 = _mm_loadu_si128(  // bottom line
                (__m128i *) &imIn->image32[x + 1 + xmin][xx]);

            source = _mm_unpacklo_epi8(source1, source2);
            pix = _mm_unpacklo_epi8(source, _mm_setzero_si128());
            sss0 = _mm_add_epi32(sss0, _mm_madd_epi16(pix, mmk));
            pix = _mm_unpackhi_epi8(source, _mm_setzero_si128());
            sss1 = _mm_add_epi32(sss1, _mm_madd_epi16(pix, mmk));

            source = _mm_unpackhi_epi8(source1, source2);
            pix = _mm_unpacklo_epi8(source, _mm_setzero_si128());
            sss2 = _mm_add_epi32(sss2, _mm_madd_epi16(pix, mmk));
            pix = _mm_unpackhi_epi8(source, _mm_setzero_si128());
            sss3 = _mm_add_epi32(sss3, _mm_madd_epi16(pix, mmk));

            source1 = _mm_loadu_si128(  // top line
                (__m128i *) &imIn->image32[x + xmin][xx + 4]);
            source2 = _mm_loadu_si128(  // bottom line
                (__m128i *) &imIn->image32[x + 1 + xmin][xx + 4]);

            source = _mm_unpacklo_epi8(source1, source2);
            pix = _mm_unpacklo_epi8(source, _mm_setzero_si128());
            sss4 = _mm_add_epi32(sss4, _mm_madd_epi16(pix, mmk));
            pix = _mm_unpackhi_epi8(source, _mm_setzero_si128());
            sss5 = _mm_add_epi32(sss5, _mm_madd_epi16(pix, mmk));

            source = _mm_unpackhi_epi8(source1, source2);
            pix = _mm_unpacklo_epi8(source, _mm_setzero_si128());
            sss6 = _mm_add_epi32(sss6, _mm_madd_epi16(pix, mmk));
            pix = _mm_unpackhi_epi8(source, _mm_setzero_si128());
            sss7 = _mm_add_epi32(sss7, _mm_madd_epi16(pix, mmk));
        }
        for (; x < xmax; x += 1) {
            __m128i source, source1, pix, mmk;
            mmk = _mm_set1_epi32(k[x]);

            source1 = _mm_loadu_si128(  // top line
                (__m128i *) &imIn->image32[x + xmin][xx]);

            source = _mm_unpacklo_epi8(source1, _mm_setzero_si128());
            pix = _mm_unpacklo_epi8(source, _mm_setzero_si128());
            sss0 = _mm_add_epi32(sss0, _mm_madd_epi16(pix, mmk));
            pix = _mm_unpackhi_epi8(source, _mm_setzero_si128());
            sss1 = _mm_add_epi32(sss1, _mm_madd_epi16(pix, mmk));

            source = _mm_unpackhi_epi8(source1, _mm_setzero_si128());
            pix = _mm_unpacklo_epi8(source, _mm_setzero_si128());
            sss2 = _mm_add_epi32(sss2, _mm_madd_epi16(pix, mmk));
            pix = _mm_unpackhi_epi8(source, _mm_setzero_si128());
            sss3 = _mm_add_epi32(sss3, _mm_madd_epi16(pix, mmk));

            source1 = _mm_loadu_si128(  // top line
                (__m128i *) &imIn->image32[x + xmin][xx + 4]);

            source = _mm_unpacklo_epi8(source1, _mm_setzero_si128());
            pix = _mm_unpacklo_epi8(source, _mm_setzero_si128());
            sss4 = _mm_add_epi32(sss4, _mm_madd_epi16(pix, mmk));
            pix = _mm_unpackhi_epi8(source, _mm_setzero_si128());
            sss5 = _mm_add_epi32(sss5, _mm_madd_epi16(pix, mmk));

            source = _mm_unpackhi_epi8(source1, _mm_setzero_si128());
            pix = _mm_unpacklo_epi8(source, _mm_setzero_si128());
            sss6 = _mm_add_epi32(sss6, _mm_madd_epi16(pix, mmk));
            pix = _mm_unpackhi_epi8(source, _mm_setzero_si128());
            sss7 = _mm_add_epi32(sss7, _mm_madd_epi16(pix, mmk));
        }
        sss0 = _mm_srai_epi32(sss0, coefs_precision);
        sss1 = _mm_srai_epi32(sss1, coefs_precision);
        sss2 = _mm_srai_epi32(sss2, coefs_precision);
        sss3 = _mm_srai_epi32(sss3, coefs_precision);
        sss4 = _mm_srai_epi32(sss4, coefs_precision);
        sss5 = _mm_srai_epi32(sss5, coefs_precision);
        sss6 = _mm_srai_epi32(sss6, coefs_precision);
        sss7 = _mm_srai_epi32(sss7, coefs_precision);

        sss0 = _mm_packs_epi32(sss0, sss1);
        sss2 = _mm_packs_epi32(sss2, sss3);
        sss0 = _mm_packus_epi16(sss0, sss2);
        _mm_storeu_si128((__m128i *) &lineOut[xx], sss0);
        sss4 = _mm_packs_epi32(sss4, sss5);
        sss6 = _mm_packs_epi32(sss6, sss7);
        sss4 = _mm_packus_epi16(sss4, sss6);
        _mm_storeu_si128((__m128i *) &lineOut[xx + 4], sss4);
    }

#elif defined(__riscv_vector)

    /* RVV: VL-agnostic loop using wider LMUL to process multiple pixels at once.
     * Vertical convolution applies the same coefficient to all pixels in a row,
     * so we can treat the row as a flat array of bytes and process N*4 channels.
     * We use i32m4 accumulators to maximize throughput on wider VLEN hardware.
     */
    for (; xx < xsize; ) {
        size_t n_channels = (size_t)(xsize - xx) * 4;
        size_t vl = __riscv_vsetvl_e32m4(n_channels);

        vint32m4_t sss = __riscv_vmv_v_x_i32m4(1 << (coefs_precision - 1), vl);

        for (x = 0; x < xmax; x++) {
            int16_t coeff = k[x];
            UINT8 *row = (UINT8 *)&imIn->image32[x + xmin][xx];

            /* Load vl bytes, widen u8 -> u16 -> i16, multiply by coeff, accumulate */
            vuint8m1_t pu8 = __riscv_vle8_v_u8m1(row, vl);
            vuint16m2_t pu16 = __riscv_vzext_vf2_u16m2(pu8, vl);
            vint16m2_t pi16 = __riscv_vreinterpret_v_u16m2_i16m2(pu16);
            vint32m4_t wide = __riscv_vwmul_vx_i32m4(pi16, coeff, vl);
            sss = __riscv_vadd_vv_i32m4(sss, wide, vl);
        }

        /* Arithmetic right shift */
        sss = __riscv_vsra_vx_i32m4(sss, coefs_precision, vl);

        /* Clamp to [0, 255] and narrow i32 -> u8 */
        vint32m4_t zero = __riscv_vmv_v_x_i32m4(0, vl);
        vint32m4_t max255 = __riscv_vmv_v_x_i32m4(255, vl);
        sss = __riscv_vmax_vv_i32m4(sss, zero, vl);
        sss = __riscv_vmin_vv_i32m4(sss, max255, vl);

        /* Pack i32 -> u16 -> u8 and store */
        vuint32m4_t usss = __riscv_vreinterpret_v_i32m4_u32m4(sss);
        vuint16m2_t n16 = RVV_VNCLIPU(u16m2, usss, 0, vl);
        vuint8m1_t n8 = RVV_VNCLIPU(u8m1, n16, 0, vl);
        __riscv_vse8_v_u8m1((UINT8 *)&lineOut[xx], n8, vl);

        xx += vl / 4;
    }

#else

    /* Scalar fallback for the bulk loop */
    for (; xx < xsize - 3; xx += 4) {
        INT32 ss[16]; /* 4 pixels x 4 channels */
        int ch, px;
        INT32 init = 1 << (coefs_precision - 1);
        for (px = 0; px < 16; px++) {
            ss[px] = init;
        }
        for (x = 0; x < xmax; x++) {
            UINT8 *row = (UINT8 *)&imIn->image32[x + xmin][xx];
            int16_t coeff = k[x];
            for (px = 0; px < 4; px++) {
                for (ch = 0; ch < 4; ch++) {
                    ss[px * 4 + ch] += (INT32)row[px * 4 + ch] * coeff;
                }
            }
        }
        for (px = 0; px < 4; px++) {
            UINT8 *out = (UINT8 *)&lineOut[xx + px];
            for (ch = 0; ch < 4; ch++) {
                INT32 v = ss[px * 4 + ch] >> coefs_precision;
                out[ch] = (v < 0) ? 0 : (v > 255) ? 255 : (UINT8)v;
            }
        }
    }

#endif

#if defined(__SSE4_2__) || defined(__AVX2__)

    {
    __m128i initial = _mm_set1_epi32(1 << (coefs_precision-1));

    for (; xx < xsize - 1; xx += 2) {
        __m128i sss0 = initial;  // left row
        __m128i sss1 = initial;  // right row
        x = 0;
        for (; x < xmax - 1; x += 2) {
            __m128i source, source1, source2;
            __m128i pix, mmk;

            // Load two coefficients at once
            mmk = _mm_set1_epi32(*(INT32 *) &k[x]);

            source1 = _mm_loadl_epi64(  // top line
                (__m128i *) &imIn->image32[x + xmin][xx]);
            source2 = _mm_loadl_epi64(  // bottom line
                (__m128i *) &imIn->image32[x + 1 + xmin][xx]);

            source = _mm_unpacklo_epi8(source1, source2);
            pix = _mm_unpacklo_epi8(source, _mm_setzero_si128());
            sss0 = _mm_add_epi32(sss0, _mm_madd_epi16(pix, mmk));
            pix = _mm_unpackhi_epi8(source, _mm_setzero_si128());
            sss1 = _mm_add_epi32(sss1, _mm_madd_epi16(pix, mmk));
        }
        for (; x < xmax; x += 1) {
            __m128i source, source1, pix, mmk;
            mmk = _mm_set1_epi32(k[x]);

            source1 = _mm_loadl_epi64(  // top line
                (__m128i *) &imIn->image32[x + xmin][xx]);

            source = _mm_unpacklo_epi8(source1, _mm_setzero_si128());
            pix = _mm_unpacklo_epi8(source, _mm_setzero_si128());
            sss0 = _mm_add_epi32(sss0, _mm_madd_epi16(pix, mmk));
            pix = _mm_unpackhi_epi8(source, _mm_setzero_si128());
            sss1 = _mm_add_epi32(sss1, _mm_madd_epi16(pix, mmk));
        }
        sss0 = _mm_srai_epi32(sss0, coefs_precision);
        sss1 = _mm_srai_epi32(sss1, coefs_precision);

        sss0 = _mm_packs_epi32(sss0, sss1);
        sss0 = _mm_packus_epi16(sss0, sss0);
        _mm_storel_epi64((__m128i *) &lineOut[xx], sss0);
    }

    for (; xx < xsize; xx++) {
        __m128i sss = initial;
        x = 0;
        for (; x < xmax - 1; x += 2) {
            __m128i source, source1, source2;
            __m128i pix, mmk;

            // Load two coefficients at once
            mmk = _mm_set1_epi32(*(INT32 *) &k[x]);

            source1 = _mm_cvtsi32_si128(  // top line
                *(int *) &imIn->image32[x + xmin][xx]);
            source2 = _mm_cvtsi32_si128(  // bottom line
                *(int *) &imIn->image32[x + 1 + xmin][xx]);

            source = _mm_unpacklo_epi8(source1, source2);
            pix = _mm_unpacklo_epi8(source, _mm_setzero_si128());
            sss = _mm_add_epi32(sss, _mm_madd_epi16(pix, mmk));
        }
        for (; x < xmax; x++) {
            __m128i pix = mm_cvtepu8_epi32(&imIn->image32[x + xmin][xx]);
            __m128i mmk = _mm_set1_epi32(k[x]);
            sss = _mm_add_epi32(sss, _mm_madd_epi16(pix, mmk));
        }
        sss = _mm_srai_epi32(sss, coefs_precision);
        sss = _mm_packs_epi32(sss, sss);
        lineOut[xx] = _mm_cvtsi128_si32(_mm_packus_epi16(sss, sss));
    }

    }

#elif defined(__riscv_vector)

    /* RVV: all pixels already handled by the VL-agnostic loop above */

#else

    /* Scalar fallback for remaining pixels */
    for (; xx < xsize; xx++) {
        INT32 init = 1 << (coefs_precision - 1);
        int ch;
        UINT8 *out = (UINT8 *)&lineOut[xx];
        for (ch = 0; ch < 4; ch++) {
            INT32 ss = init;
            for (x = 0; x < xmax; x++) {
                UINT8 *pix = (UINT8 *)&imIn->image32[x + xmin][xx];
                ss += (INT32)pix[ch] * k[x];
            }
            ss = ss >> coefs_precision;
            out[ch] = (ss < 0) ? 0 : (ss > 255) ? 255 : (UINT8)ss;
        }
    }

#endif
}
