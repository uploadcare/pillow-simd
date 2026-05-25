void
ImagingFilter3x3f_4u8(Imaging imOut, Imaging im, const float* kernel,
                      float offset)
{
    int x = 0, y = 0;

    memcpy(imOut->image32[0], im->image32[0], im->linesize);

    for (y = 1; y < im->ysize-1; y++) {
        INT32* in_1 = im->image32[y-1];
        INT32* in0 = im->image32[y];
        INT32* in1 = im->image32[y+1];
        INT32* out = imOut->image32[y];

#if defined(__AVX2__)

#define MM_KERNEL1x3_LOAD(row, x) \
    pix0##row = _mm_cvtepi32_ps(mm_cvtepu8_epi32(&in1[x])); \
    pix1##row = _mm_cvtepi32_ps(mm_cvtepu8_epi32(&in0[x])); \
    pix2##row = _mm_cvtepi32_ps(mm_cvtepu8_epi32(&in_1[x]));

#define MM_KERNEL1x3_SUM(row, kindex) \
    ss = _mm_add_ps(ss, _mm_mul_ps(pix0##row, _mm_set1_ps(kernel[0 + kindex]))); \
    ss = _mm_add_ps(ss, _mm_mul_ps(pix1##row, _mm_set1_ps(kernel[3 + kindex]))); \
    ss = _mm_add_ps(ss, _mm_mul_ps(pix2##row, _mm_set1_ps(kernel[6 + kindex])));

#define MM256_LOAD(row, x) \
    pix0##row = _mm256_cvtepi32_ps(mm256_cvtepu8_epi32(&in1[x])); \
    pix1##row = _mm256_cvtepi32_ps(mm256_cvtepu8_epi32(&in0[x])); \
    pix2##row = _mm256_cvtepi32_ps(mm256_cvtepu8_epi32(&in_1[x]));

#define MM256_SUM(pixrow, kernelrow) \
    ss = _mm256_add_ps(ss, _mm256_mul_ps(pix0##pixrow, kernel0##kernelrow)); \
    ss = _mm256_add_ps(ss, _mm256_mul_ps(pix1##pixrow, kernel1##kernelrow)); \
    ss = _mm256_add_ps(ss, _mm256_mul_ps(pix2##pixrow, kernel2##kernelrow));

#define MM256_PERMUTE(row, from0, from1, control) \
    pix0##row = _mm256_permute2f128_ps(pix0##from0, pix0##from1, control); \
    pix1##row = _mm256_permute2f128_ps(pix1##from0, pix1##from1, control); \
    pix2##row = _mm256_permute2f128_ps(pix2##from0, pix2##from1, control);

#define MM256_OUT(x) \
    _mm_cvtps_epi32(_mm_add_ps( \
        _mm256_extractf128_ps(ss, 0), \
        _mm256_extractf128_ps(ss, 1) \
    ))

        __m256 kernel00 = _mm256_insertf128_ps(
            _mm256_set1_ps(kernel[0+0]),
            _mm_set1_ps(kernel[0+1]), 1);
        __m256 kernel01 = _mm256_castps128_ps256(_mm_set1_ps(kernel[0+2]));
        __m256 kernel10 = _mm256_insertf128_ps(
            _mm256_set1_ps(kernel[3+0]),
            _mm_set1_ps(kernel[3+1]), 1);
        __m256 kernel11 = _mm256_castps128_ps256(_mm_set1_ps(kernel[3+2]));
        __m256 kernel20 = _mm256_insertf128_ps(
            _mm256_set1_ps(kernel[6+0]),
            _mm_set1_ps(kernel[6+1]), 1);
        __m256 kernel21 = _mm256_castps128_ps256(_mm_set1_ps(kernel[6+2]));
        __m256 pix00, pix10, pix20;
        __m256 pix01, pix11, pix21;
        __m256 pix02, pix12, pix22;

        out[0] = in0[0];
        x = 1;
        MM256_LOAD(0, 0);
        for (; x < im->xsize-1-7; x += 8) {
            __m256 ss;
            __m128i ssi0, ssi1, ssi2, ssi3;

            ss = _mm256_castps128_ps256(_mm_set1_ps(offset));
            MM256_SUM(0, 0);
            MM256_LOAD(1, x+1);
            MM256_SUM(1, 1);
            ssi0 = MM256_OUT(x);

            ss = _mm256_castps128_ps256(_mm_set1_ps(offset));
            MM256_SUM(1, 0);
            MM256_LOAD(2, x+3);
            MM256_SUM(2, 1);
            ssi2 = MM256_OUT(x+2);

            ss = _mm256_castps128_ps256(_mm_set1_ps(offset));
            MM256_PERMUTE(0, 0, 1, 0x21);
            MM256_SUM(0, 0);
            MM256_PERMUTE(1, 1, 2, 0x21);
            MM256_SUM(1, 1);
            ssi1 = MM256_OUT(x+1);

            ss = _mm256_castps128_ps256(_mm_set1_ps(offset));
            MM256_SUM(1, 0);
            MM256_PERMUTE(0, 2, 2, 0x21);
            MM256_SUM(0, 1);
            ssi3 = MM256_OUT(x+3);

            ssi0 = _mm_packs_epi32(ssi0, ssi1);
            ssi2 = _mm_packs_epi32(ssi2, ssi3);
            ssi0 = _mm_packus_epi16(ssi0, ssi2);
            _mm_storeu_si128((__m128i*) &out[x], ssi0);

            ss = _mm256_castps128_ps256(_mm_set1_ps(offset));
            MM256_SUM(2, 0);
            MM256_LOAD(1, x+5);
            MM256_SUM(1, 1);
            ssi0 = MM256_OUT(x+4);

            ss = _mm256_castps128_ps256(_mm_set1_ps(offset));
            MM256_SUM(1, 0);
            MM256_LOAD(0, x+7);
            MM256_SUM(0, 1);
            ssi2 = MM256_OUT(x+6);

            ss = _mm256_castps128_ps256(_mm_set1_ps(offset));
            MM256_PERMUTE(2, 2, 1, 0x21);
            MM256_SUM(2, 0);
            MM256_PERMUTE(1, 1, 0, 0x21);
            MM256_SUM(1, 1);
            ssi1 = MM256_OUT(x+5);

            ss = _mm256_castps128_ps256(_mm_set1_ps(offset));
            MM256_SUM(1, 0);
            MM256_PERMUTE(2, 0, 0, 0x21);
            MM256_SUM(2, 1);
            ssi3 = MM256_OUT(x+7);

            ssi0 = _mm_packs_epi32(ssi0, ssi1);
            ssi2 = _mm_packs_epi32(ssi2, ssi3);
            ssi0 = _mm_packus_epi16(ssi0, ssi2);
            _mm_storeu_si128((__m128i*) &out[x+4], ssi0);
        }

        for (; x < im->xsize-1-1; x += 2) {
            __m256 ss;
            __m128i ssi0, ssi1;

            ss = _mm256_castps128_ps256(_mm_set1_ps(offset));
            MM256_SUM(0, 0);
            MM256_LOAD(1, x+1);
            MM256_SUM(1, 1);
            ssi0 = MM256_OUT(x);

            ss = _mm256_castps128_ps256(_mm_set1_ps(offset));
            MM256_PERMUTE(0, 0, 1, 0x21);
            MM256_SUM(0, 0);
            MM256_PERMUTE(1, 0, 1, 0xf3);
            MM256_SUM(1, 1);
            ssi1 = MM256_OUT(x+1);

            ssi0 = _mm_packs_epi32(ssi0, ssi1);
            ssi0 = _mm_packus_epi16(ssi0, ssi0);
            _mm_storel_epi64((__m128i*) &out[x], ssi0);

            MM256_PERMUTE(0, 0, 1, 0x21);
        }
        for (; x < im->xsize-1; x++) {
            __m128 pix00, pix10, pix20;
            __m128 ss = _mm_set1_ps(offset);
            __m128i ssi0;
            MM_KERNEL1x3_LOAD(0, x-1);
            MM_KERNEL1x3_SUM(0, 0);
            MM_KERNEL1x3_LOAD(0, x+0);
            MM_KERNEL1x3_SUM(0, 1);
            MM_KERNEL1x3_LOAD(0, x+1);
            MM_KERNEL1x3_SUM(0, 2);
            ssi0 = _mm_cvtps_epi32(ss);
            ssi0 = _mm_packs_epi32(ssi0, ssi0);
            ssi0 = _mm_packus_epi16(ssi0, ssi0);
            out[x] = _mm_cvtsi128_si32(ssi0);
        }
        out[x] = in0[x];

#undef MM_KERNEL1x3_LOAD
#undef MM_KERNEL1x3_SUM
#undef MM256_LOAD
#undef MM256_SUM
#undef MM256_PERMUTE
#undef MM256_OUT

#elif defined(__SSE4_2__)

#define MM_KERNEL1x3_LOAD(row, x) \
    pix0##row = _mm_cvtepi32_ps(mm_cvtepu8_epi32(&in1[x])); \
    pix1##row = _mm_cvtepi32_ps(mm_cvtepu8_epi32(&in0[x])); \
    pix2##row = _mm_cvtepi32_ps(mm_cvtepu8_epi32(&in_1[x]));

#define MM_KERNEL1x3_SUM(row, kindex) \
    ss = _mm_add_ps(ss, _mm_mul_ps(pix0##row, _mm_set1_ps(kernel[0 + kindex]))); \
    ss = _mm_add_ps(ss, _mm_mul_ps(pix1##row, _mm_set1_ps(kernel[3 + kindex]))); \
    ss = _mm_add_ps(ss, _mm_mul_ps(pix2##row, _mm_set1_ps(kernel[6 + kindex])));

        __m128 pix00, pix10, pix20;
        __m128 pix01, pix11, pix21;
        __m128 pix02, pix12, pix22;
        MM_KERNEL1x3_LOAD(0, 0);
        MM_KERNEL1x3_LOAD(1, 1);

        out[0] = in0[0];
        x = 1;
        for (; x < im->xsize-1-2; x += 3) {
            __m128 ss;
            __m128i ssi0, ssi1, ssi2;

            ss = _mm_set1_ps(offset);
            MM_KERNEL1x3_SUM(0, 0);
            MM_KERNEL1x3_SUM(1, 1);
            MM_KERNEL1x3_LOAD(2, x+1);
            MM_KERNEL1x3_SUM(2, 2);
            ssi0 = _mm_cvtps_epi32(ss);

            ss = _mm_set1_ps(offset);
            MM_KERNEL1x3_SUM(1, 0);
            MM_KERNEL1x3_SUM(2, 1);
            MM_KERNEL1x3_LOAD(0, x+2);
            MM_KERNEL1x3_SUM(0, 2);
            ssi1 = _mm_cvtps_epi32(ss);

            ss = _mm_set1_ps(offset);
            MM_KERNEL1x3_SUM(2, 0);
            MM_KERNEL1x3_SUM(0, 1);
            MM_KERNEL1x3_LOAD(1, x+3);
            MM_KERNEL1x3_SUM(1, 2);
            ssi2 = _mm_cvtps_epi32(ss);

            ssi0 = _mm_packs_epi32(ssi0, ssi1);
            ssi1 = _mm_packs_epi32(ssi2, ssi2);
            ssi0 = _mm_packus_epi16(ssi0, ssi1);
            _mm_storeu_si128((__m128i*) &out[x], ssi0);
        }
        for (; x < im->xsize-1; x++) {
            __m128 ss = _mm_set1_ps(offset);
            __m128i ssi0;
            MM_KERNEL1x3_LOAD(0, x-1);
            MM_KERNEL1x3_SUM(0, 0);
            MM_KERNEL1x3_LOAD(0, x+0);
            MM_KERNEL1x3_SUM(0, 1);
            MM_KERNEL1x3_LOAD(0, x+1);
            MM_KERNEL1x3_SUM(0, 2);
            ssi0 = _mm_cvtps_epi32(ss);
            ssi0 = _mm_packs_epi32(ssi0, ssi0);
            ssi0 = _mm_packus_epi16(ssi0, ssi0);
            out[x] = _mm_cvtsi128_si32(ssi0);
        }
        out[x] = in0[x];

#undef MM_KERNEL1x3_LOAD
#undef MM_KERNEL1x3_SUM

#elif defined(__riscv_vector)

        out[0] = in0[0];
        int npixels = im->xsize - 2;
        x = 1;
        for (int done = 0; done < npixels; ) {
            size_t vl = __riscv_vsetvl_e32m4(npixels - done);
            UINT8* pin1 = (UINT8*)&in1[x];
            UINT8* pin0 = (UINT8*)&in0[x];
            UINT8* pin_1 = (UINT8*)&in_1[x];

            /* Accumulators for R, G, B, A channels -- process vl pixels at once */
            vfloat32m4_t accR = __riscv_vfmv_v_f_f32m4(offset, vl);
            vfloat32m4_t accG = __riscv_vfmv_v_f_f32m4(offset, vl);
            vfloat32m4_t accB = __riscv_vfmv_v_f_f32m4(offset, vl);
            vfloat32m4_t accA = __riscv_vfmv_v_f_f32m4(offset, vl);

            /* Macro: load vl pixels deinterleaved, widen each channel, fmacc */
#define RVV_3x3F_ACCUM(base_ptr, kidx) do { \
                vuint8m1_t vr_, vg_, vb_, va_; \
                RVV_VLSEG4E8_U8M1(vr_, vg_, vb_, va_, (const uint8_t*)(base_ptr), vl); \
                vuint16m2_t wr_ = __riscv_vzext_vf2_u16m2(vr_, vl); \
                vuint32m4_t dr_ = __riscv_vzext_vf2_u32m4(wr_, vl); \
                accR = __riscv_vfmacc_vf_f32m4(accR, kernel[kidx], __riscv_vfcvt_f_xu_v_f32m4(dr_, vl), vl); \
                vuint16m2_t wg_ = __riscv_vzext_vf2_u16m2(vg_, vl); \
                vuint32m4_t dg_ = __riscv_vzext_vf2_u32m4(wg_, vl); \
                accG = __riscv_vfmacc_vf_f32m4(accG, kernel[kidx], __riscv_vfcvt_f_xu_v_f32m4(dg_, vl), vl); \
                vuint16m2_t wb_ = __riscv_vzext_vf2_u16m2(vb_, vl); \
                vuint32m4_t db_ = __riscv_vzext_vf2_u32m4(wb_, vl); \
                accB = __riscv_vfmacc_vf_f32m4(accB, kernel[kidx], __riscv_vfcvt_f_xu_v_f32m4(db_, vl), vl); \
                vuint16m2_t wa_ = __riscv_vzext_vf2_u16m2(va_, vl); \
                vuint32m4_t da_ = __riscv_vzext_vf2_u32m4(wa_, vl); \
                accA = __riscv_vfmacc_vf_f32m4(accA, kernel[kidx], __riscv_vfcvt_f_xu_v_f32m4(da_, vl), vl); \
            } while(0)

            /* Row y+1 (in1) * kernel[0..2] */
            RVV_3x3F_ACCUM(pin1 - 4, 0);
            RVV_3x3F_ACCUM(pin1,     1);
            RVV_3x3F_ACCUM(pin1 + 4, 2);

            /* Row y (in0) * kernel[3..5] */
            RVV_3x3F_ACCUM(pin0 - 4, 3);
            RVV_3x3F_ACCUM(pin0,     4);
            RVV_3x3F_ACCUM(pin0 + 4, 5);

            /* Row y-1 (in_1) * kernel[6..8] */
            RVV_3x3F_ACCUM(pin_1 - 4, 6);
            RVV_3x3F_ACCUM(pin_1,     7);
            RVV_3x3F_ACCUM(pin_1 + 4, 8);

#undef RVV_3x3F_ACCUM

            /* Convert f32 -> u32, narrow u32 -> u16 -> u8, interleave and store */
            vuint32m4_t uR = __riscv_vfcvt_xu_f_v_u32m4(accR, vl);
            vuint32m4_t uG = __riscv_vfcvt_xu_f_v_u32m4(accG, vl);
            vuint32m4_t uB = __riscv_vfcvt_xu_f_v_u32m4(accB, vl);
            vuint32m4_t uA = __riscv_vfcvt_xu_f_v_u32m4(accA, vl);

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
                float sum = offset;
                sum += ((UINT8*)&in1[x-1])[ch] * kernel[0];
                sum += ((UINT8*)&in1[x])[ch]   * kernel[1];
                sum += ((UINT8*)&in1[x+1])[ch] * kernel[2];
                sum += ((UINT8*)&in0[x-1])[ch] * kernel[3];
                sum += ((UINT8*)&in0[x])[ch]   * kernel[4];
                sum += ((UINT8*)&in0[x+1])[ch] * kernel[5];
                sum += ((UINT8*)&in_1[x-1])[ch] * kernel[6];
                sum += ((UINT8*)&in_1[x])[ch]   * kernel[7];
                sum += ((UINT8*)&in_1[x+1])[ch] * kernel[8];
                int val = (int)(sum + 0.5f);
                if (val < 0) val = 0;
                if (val > 255) val = 255;
                pOut[ch] = (UINT8)val;
            }
        }
        out[x] = in0[x];

#endif
    }
    memcpy(imOut->image32[y], im->image32[y], im->linesize);
}
