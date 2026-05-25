void
ImagingFilter3x3i_4u8(Imaging imOut, Imaging im, const INT16* kernel,
                      INT32 offset)
{
    int x, y;

    offset += 1 << (PRECISION_BITS-1);

    memcpy(imOut->image32[0], im->image32[0], im->linesize);

    for (y = 1; y < im->ysize-1; y++) {
        INT32* in_1 = im->image32[y-1];
        INT32* in0 = im->image32[y];
        INT32* in1 = im->image32[y+1];
        INT32* out = imOut->image32[y];

#if defined(__AVX2__)

    #define MM_KERNEL_LOAD(x) \
        source = _mm256_castsi128_si256(_mm_shuffle_epi8(_mm_loadu_si128((__m128i*) &in1[x]), shuffle)); \
        source = _mm256_inserti128_si256(source, _mm256_castsi256_si128(source), 1); \
        pix00 = _mm256_unpacklo_epi8(source, _mm256_setzero_si256()); \
        pix01 = _mm256_unpackhi_epi8(source, _mm256_setzero_si256()); \
        source = _mm256_castsi128_si256(_mm_shuffle_epi8(_mm_loadu_si128((__m128i*) &in0[x]), shuffle)); \
        source = _mm256_inserti128_si256(source, _mm256_castsi256_si128(source), 1); \
        pix10 = _mm256_unpacklo_epi8(source, _mm256_setzero_si256()); \
        pix11 = _mm256_unpackhi_epi8(source, _mm256_setzero_si256()); \
        source = _mm256_castsi128_si256(_mm_shuffle_epi8(_mm_loadu_si128((__m128i*) &in_1[x]), shuffle)); \
        source = _mm256_inserti128_si256(source, _mm256_castsi256_si128(source), 1); \
        pix20 = _mm256_unpacklo_epi8(source, _mm256_setzero_si256()); \
        pix21 = _mm256_unpackhi_epi8(source, _mm256_setzero_si256());

    #define MM_KERNEL_SUM(ss, row, kctl) \
        ss = _mm256_add_epi32(ss, _mm256_madd_epi16( \
            pix0##row, _mm256_shuffle_epi32(kernel00, kctl))); \
        ss = _mm256_add_epi32(ss, _mm256_madd_epi16( \
            pix1##row, _mm256_shuffle_epi32(kernel10, kctl))); \
        ss = _mm256_add_epi32(ss, _mm256_madd_epi16( \
            pix2##row, _mm256_shuffle_epi32(kernel20, kctl)));

        __m128i shuffle = _mm_set_epi8(15,11,14,10,13,9,12,8, 7,3,6,2,5,1,4,0);
        __m256i kernel00 = _mm256_set_epi16(
            0, 0, 0, 0, kernel[2], kernel[1], kernel[0], 0,
            0, 0, 0, 0, 0, kernel[2], kernel[1], kernel[0]);
        __m256i kernel10 = _mm256_set_epi16(
            0, 0, 0, 0, kernel[5], kernel[4], kernel[3], 0,
            0, 0, 0, 0, 0, kernel[5], kernel[4], kernel[3]);
        __m256i kernel20 = _mm256_set_epi16(
            0, 0, 0, 0, kernel[8], kernel[7], kernel[6], 0,
            0, 0, 0, 0, 0, kernel[8], kernel[7], kernel[6]);
        __m256i pix00, pix10, pix20;
        __m256i pix01, pix11, pix21;
        __m256i source;

        out[0] = in0[0];
        x = 1;
        if (im->xsize >= 4) {
            MM_KERNEL_LOAD(0);
        }
        for (; x < im->xsize-1-5; x += 4) {
            __m256i ss0, ss2;
            __m128i ssout;

            ss0 = _mm256_set1_epi32(offset);
            MM_KERNEL_SUM(ss0, 0, 0x00);
            MM_KERNEL_SUM(ss0, 1, 0x55);
            ss0 = _mm256_srai_epi32(ss0, PRECISION_BITS);

            ss2 = _mm256_set1_epi32(offset);
            MM_KERNEL_SUM(ss2, 1, 0x00);
            MM_KERNEL_LOAD(x+3);
            MM_KERNEL_SUM(ss2, 0, 0x55);
            ss2 = _mm256_srai_epi32(ss2, PRECISION_BITS);

            ss0 = _mm256_packs_epi32(ss0, ss2);
            ssout = _mm_packus_epi16(
                _mm256_extracti128_si256(ss0, 0),
                _mm256_extracti128_si256(ss0, 1));
            ssout = _mm_shuffle_epi32(ssout, 0xd8);
            _mm_storeu_si128((__m128i*) &out[x], ssout);
        }
        for (; x < im->xsize-1; x++) {
            __m256i ss = _mm256_set1_epi32(offset);

            source = _mm256_castsi128_si256(_mm_shuffle_epi8(_mm_or_si128(
                _mm_slli_si128(_mm_cvtsi32_si128(*(INT32*) &in1[x+1]), 8),
                _mm_loadl_epi64((__m128i*) &in1[x-1])), shuffle));
            source = _mm256_inserti128_si256(source, _mm256_castsi256_si128(source), 1);
            pix00 = _mm256_unpacklo_epi8(source, _mm256_setzero_si256());
            pix01 = _mm256_unpackhi_epi8(source, _mm256_setzero_si256());
            source = _mm256_castsi128_si256(_mm_shuffle_epi8(_mm_or_si128(
                _mm_slli_si128(_mm_cvtsi32_si128(*(INT32*) &in0[x+1]), 8),
                _mm_loadl_epi64((__m128i*) &in0[x-1])), shuffle));
            source = _mm256_inserti128_si256(source, _mm256_castsi256_si128(source), 1);
            pix10 = _mm256_unpacklo_epi8(source, _mm256_setzero_si256());
            pix11 = _mm256_unpackhi_epi8(source, _mm256_setzero_si256());
            source = _mm256_castsi128_si256(_mm_shuffle_epi8(_mm_or_si128(
                _mm_slli_si128(_mm_cvtsi32_si128(*(INT32*) &in_1[x+1]), 8),
                _mm_loadl_epi64((__m128i*) &in_1[x-1])), shuffle));
            source = _mm256_inserti128_si256(source, _mm256_castsi256_si128(source), 1);
            pix20 = _mm256_unpacklo_epi8(source, _mm256_setzero_si256());
            pix21 = _mm256_unpackhi_epi8(source, _mm256_setzero_si256());

            MM_KERNEL_SUM(ss, 0, 0x00);
            MM_KERNEL_SUM(ss, 1, 0x55);

            ss = _mm256_srai_epi32(ss, PRECISION_BITS);

            ss = _mm256_packs_epi32(ss, ss);
            ss = _mm256_packus_epi16(ss, ss);
            out[x] = _mm_cvtsi128_si32(_mm256_castsi256_si128(ss));
        }
        out[x] = in0[x];
    #undef MM_KERNEL_LOAD
    #undef MM_KERNEL_SUM

#elif defined(__SSE4_2__)

    #define MM_KERNEL_LOAD(x) \
        source = _mm_shuffle_epi8(_mm_loadu_si128((__m128i*) &in1[x]), shuffle); \
        pix00 = _mm_unpacklo_epi8(source, _mm_setzero_si128()); \
        pix01 = _mm_unpackhi_epi8(source, _mm_setzero_si128()); \
        source = _mm_shuffle_epi8(_mm_loadu_si128((__m128i*) &in0[x]), shuffle); \
        pix10 = _mm_unpacklo_epi8(source, _mm_setzero_si128()); \
        pix11 = _mm_unpackhi_epi8(source, _mm_setzero_si128()); \
        source = _mm_shuffle_epi8(_mm_loadu_si128((__m128i*) &in_1[x]), shuffle); \
        pix20 = _mm_unpacklo_epi8(source, _mm_setzero_si128()); \
        pix21 = _mm_unpackhi_epi8(source, _mm_setzero_si128());

    #define MM_KERNEL_SUM(ss, row, kctl) \
        ss = _mm_add_epi32(ss, _mm_madd_epi16( \
            pix0##row, _mm_shuffle_epi32(kernel00, kctl))); \
        ss = _mm_add_epi32(ss, _mm_madd_epi16( \
            pix1##row, _mm_shuffle_epi32(kernel10, kctl))); \
        ss = _mm_add_epi32(ss, _mm_madd_epi16( \
            pix2##row, _mm_shuffle_epi32(kernel20, kctl)));

        __m128i shuffle = _mm_set_epi8(15,11,14,10,13,9,12,8, 7,3,6,2,5,1,4,0);
        __m128i kernel00 = _mm_set_epi16(
            kernel[2], kernel[1], kernel[0], 0,
            0, kernel[2], kernel[1], kernel[0]);
        __m128i kernel10 = _mm_set_epi16(
            kernel[5], kernel[4], kernel[3], 0,
            0, kernel[5], kernel[4], kernel[3]);
        __m128i kernel20 = _mm_set_epi16(
            kernel[8], kernel[7], kernel[6], 0,
            0, kernel[8], kernel[7], kernel[6]);
        __m128i pix00, pix10, pix20;
        __m128i pix01, pix11, pix21;
        __m128i source;

        out[0] = in0[0];
        x = 1;
        if (im->xsize >= 4) {
            MM_KERNEL_LOAD(0);
        }
        for (; x < im->xsize-1-5; x += 4) {
            __m128i ss0 = _mm_set1_epi32(offset);
            __m128i ss1 = _mm_set1_epi32(offset);
            __m128i ss2 = _mm_set1_epi32(offset);
            __m128i ss3 = _mm_set1_epi32(offset);

            MM_KERNEL_SUM(ss0, 0, 0x00);
            MM_KERNEL_SUM(ss0, 1, 0x55);
            ss0 = _mm_srai_epi32(ss0, PRECISION_BITS);

            MM_KERNEL_SUM(ss1, 0, 0xaa);
            MM_KERNEL_SUM(ss1, 1, 0xff);
            ss1 = _mm_srai_epi32(ss1, PRECISION_BITS);

            MM_KERNEL_SUM(ss2, 1, 0x00);
            MM_KERNEL_SUM(ss3, 1, 0xaa);

            MM_KERNEL_LOAD(x+3);

            MM_KERNEL_SUM(ss2, 0, 0x55);
            MM_KERNEL_SUM(ss3, 0, 0xff);
            ss2 = _mm_srai_epi32(ss2, PRECISION_BITS);
            ss3 = _mm_srai_epi32(ss3, PRECISION_BITS);

            ss0 = _mm_packs_epi32(ss0, ss1);
            ss2 = _mm_packs_epi32(ss2, ss3);
            ss0 = _mm_packus_epi16(ss0, ss2);
            _mm_storeu_si128((__m128i*) &out[x], ss0);
        }
        for (; x < im->xsize-1; x++) {
            __m128i ss = _mm_set1_epi32(offset);

            source = _mm_shuffle_epi8(_mm_loadl_epi64((__m128i*) &in1[x-1]), shuffle);
            pix00 = _mm_unpacklo_epi8(source, _mm_setzero_si128());
            source = _mm_shuffle_epi8(_mm_cvtsi32_si128(*(int*) &in1[x+1]), shuffle);
            pix01 = _mm_unpacklo_epi8(source, _mm_setzero_si128());
            source = _mm_shuffle_epi8(_mm_loadl_epi64((__m128i*) &in0[x-1]), shuffle);
            pix10 = _mm_unpacklo_epi8(source, _mm_setzero_si128());
            source = _mm_shuffle_epi8(_mm_cvtsi32_si128(*(int*) &in0[x+1]), shuffle);
            pix11 = _mm_unpacklo_epi8(source, _mm_setzero_si128());
            source = _mm_shuffle_epi8(_mm_loadl_epi64((__m128i*) &in_1[x-1]), shuffle);
            pix20 = _mm_unpacklo_epi8(source, _mm_setzero_si128());
            source = _mm_shuffle_epi8(_mm_cvtsi32_si128(*(int*) &in_1[x+1]), shuffle);
            pix21 = _mm_unpacklo_epi8(source, _mm_setzero_si128());

            MM_KERNEL_SUM(ss, 0, 0x00);
            MM_KERNEL_SUM(ss, 1, 0x55);

            ss = _mm_srai_epi32(ss, PRECISION_BITS);

            ss = _mm_packs_epi32(ss, ss);
            ss = _mm_packus_epi16(ss, ss);
            out[x] = _mm_cvtsi128_si32(ss);
        }
        out[x] = in0[x];
    #undef MM_KERNEL_LOAD
    #undef MM_KERNEL_SUM

#elif defined(__riscv_vector)

        out[0] = in0[0];
        int npixels = im->xsize - 2;
        x = 1;
        for (int done = 0; done < npixels; ) {
            size_t vl = __riscv_vsetvl_e32m4(npixels - done);
            UINT8* pin1 = (UINT8*)&in1[x];
            UINT8* pin0 = (UINT8*)&in0[x];
            UINT8* pin_1 = (UINT8*)&in_1[x];

            /* Accumulators for R, G, B, A channels */
            vint32m4_t accR = __riscv_vmv_v_x_i32m4(offset, vl);
            vint32m4_t accG = __riscv_vmv_v_x_i32m4(offset, vl);
            vint32m4_t accB = __riscv_vmv_v_x_i32m4(offset, vl);
            vint32m4_t accA = __riscv_vmv_v_x_i32m4(offset, vl);

#define RVV_3x3I_ACCUM(base_ptr, kidx) do { \
                vuint8m1_t vr_, vg_, vb_, va_; \
                RVV_VLSEG4E8_U8M1(vr_, vg_, vb_, va_, (const uint8_t*)(base_ptr), vl); \
                vuint16m2_t wr_ = __riscv_vzext_vf2_u16m2(vr_, vl); \
                vuint32m4_t dr_ = __riscv_vzext_vf2_u32m4(wr_, vl); \
                vint32m4_t sr_ = __riscv_vreinterpret_v_u32m4_i32m4(dr_); \
                accR = __riscv_vmacc_vx_i32m4(accR, (INT32)kernel[kidx], sr_, vl); \
                vuint16m2_t wg_ = __riscv_vzext_vf2_u16m2(vg_, vl); \
                vuint32m4_t dg_ = __riscv_vzext_vf2_u32m4(wg_, vl); \
                vint32m4_t sg_ = __riscv_vreinterpret_v_u32m4_i32m4(dg_); \
                accG = __riscv_vmacc_vx_i32m4(accG, (INT32)kernel[kidx], sg_, vl); \
                vuint16m2_t wb_ = __riscv_vzext_vf2_u16m2(vb_, vl); \
                vuint32m4_t db_ = __riscv_vzext_vf2_u32m4(wb_, vl); \
                vint32m4_t sb_ = __riscv_vreinterpret_v_u32m4_i32m4(db_); \
                accB = __riscv_vmacc_vx_i32m4(accB, (INT32)kernel[kidx], sb_, vl); \
                vuint16m2_t wa_ = __riscv_vzext_vf2_u16m2(va_, vl); \
                vuint32m4_t da_ = __riscv_vzext_vf2_u32m4(wa_, vl); \
                vint32m4_t sa_ = __riscv_vreinterpret_v_u32m4_i32m4(da_); \
                accA = __riscv_vmacc_vx_i32m4(accA, (INT32)kernel[kidx], sa_, vl); \
            } while(0)

            /* Row y+1 (in1) * kernel[0..2] */
            RVV_3x3I_ACCUM(pin1 - 4, 0);
            RVV_3x3I_ACCUM(pin1,     1);
            RVV_3x3I_ACCUM(pin1 + 4, 2);

            /* Row y (in0) * kernel[3..5] */
            RVV_3x3I_ACCUM(pin0 - 4, 3);
            RVV_3x3I_ACCUM(pin0,     4);
            RVV_3x3I_ACCUM(pin0 + 4, 5);

            /* Row y-1 (in_1) * kernel[6..8] */
            RVV_3x3I_ACCUM(pin_1 - 4, 6);
            RVV_3x3I_ACCUM(pin_1,     7);
            RVV_3x3I_ACCUM(pin_1 + 4, 8);

#undef RVV_3x3I_ACCUM

            /* Shift right by PRECISION_BITS */
            accR = __riscv_vsra_vx_i32m4(accR, PRECISION_BITS, vl);
            accG = __riscv_vsra_vx_i32m4(accG, PRECISION_BITS, vl);
            accB = __riscv_vsra_vx_i32m4(accB, PRECISION_BITS, vl);
            accA = __riscv_vsra_vx_i32m4(accA, PRECISION_BITS, vl);

            /* Clamp to [0, 255] */
            vint32m4_t zero = __riscv_vmv_v_x_i32m4(0, vl);
            vint32m4_t max255 = __riscv_vmv_v_x_i32m4(255, vl);
            accR = __riscv_vmax_vv_i32m4(accR, zero, vl);
            accR = __riscv_vmin_vv_i32m4(accR, max255, vl);
            accG = __riscv_vmax_vv_i32m4(accG, zero, vl);
            accG = __riscv_vmin_vv_i32m4(accG, max255, vl);
            accB = __riscv_vmax_vv_i32m4(accB, zero, vl);
            accB = __riscv_vmin_vv_i32m4(accB, max255, vl);
            accA = __riscv_vmax_vv_i32m4(accA, zero, vl);
            accA = __riscv_vmin_vv_i32m4(accA, max255, vl);

            /* Narrow i32 -> u16 -> u8 and interleave-store */
            vuint32m4_t uR = __riscv_vreinterpret_v_i32m4_u32m4(accR);
            vuint32m4_t uG = __riscv_vreinterpret_v_i32m4_u32m4(accG);
            vuint32m4_t uB = __riscv_vreinterpret_v_i32m4_u32m4(accB);
            vuint32m4_t uA = __riscv_vreinterpret_v_i32m4_u32m4(accA);

            vuint16m2_t nR16 = RVV_VNCLIPU(u16m2, uR, 0, vl);
            vuint8m1_t  nR8  = RVV_VNCLIPU(u8m1, nR16, 0, vl);
            vuint16m2_t nG16 = RVV_VNCLIPU(u16m2, uG, 0, vl);
            vuint8m1_t  nG8  = RVV_VNCLIPU(u8m1, nG16, 0, vl);
            vuint16m2_t nB16 = RVV_VNCLIPU(u16m2, uB, 0, vl);
            vuint8m1_t  nB8  = RVV_VNCLIPU(u8m1, nB16, 0, vl);
            vuint16m2_t nA16 = RVV_VNCLIPU(u16m2, uA, 0, vl);
            vuint8m1_t  nA8  = RVV_VNCLIPU(u8m1, nA16, 0, vl);

            RVV_VSSEG4E8_U8M1((UINT8*)&out[x], nR8, nG8, nB8, nA8, vl);

            x += vl;
            done += vl;
        }
        out[im->xsize - 1] = in0[im->xsize - 1];

#else /* scalar fallback */

        out[0] = in0[0];
        for (x = 1; x < im->xsize-1; x++) {
            int ch;
            UINT8* pOut = (UINT8*)&out[x];
            for (ch = 0; ch < 4; ch++) {
                INT32 ss = offset;
                ss += ((UINT8*)&in1[x-1])[ch] * kernel[0];
                ss += ((UINT8*)&in1[x])[ch]   * kernel[1];
                ss += ((UINT8*)&in1[x+1])[ch] * kernel[2];
                ss += ((UINT8*)&in0[x-1])[ch] * kernel[3];
                ss += ((UINT8*)&in0[x])[ch]   * kernel[4];
                ss += ((UINT8*)&in0[x+1])[ch] * kernel[5];
                ss += ((UINT8*)&in_1[x-1])[ch] * kernel[6];
                ss += ((UINT8*)&in_1[x])[ch]   * kernel[7];
                ss += ((UINT8*)&in_1[x+1])[ch] * kernel[8];
                ss >>= PRECISION_BITS;
                if (ss < 0) ss = 0;
                if (ss > 255) ss = 255;
                pOut[ch] = (UINT8)ss;
            }
        }
        out[x] = in0[x];

#endif
    }
    memcpy(imOut->image32[y], im->image32[y], im->linesize);
}
