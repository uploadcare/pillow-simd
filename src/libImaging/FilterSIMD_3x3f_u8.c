void
ImagingFilter3x3f_u8(Imaging imOut, Imaging im, const float* kernel,
                     float offset)
{
    int x, y;

#if defined(__SSE4_2__)

#define MM_KERNEL1x3_SUM1(ss, row, kernel) \
    ss = _mm_mul_ps(pix0##row, kernel0##kernel); \
    ss = _mm_add_ps(ss, _mm_mul_ps(pix1##row, kernel1##kernel)); \
    ss = _mm_add_ps(ss, _mm_mul_ps(pix2##row, kernel2##kernel));

#define MM_KERNEL1x3_SUM1_2(ss, row, kernel) \
    ss = _mm_mul_ps(pix3##row, kernel0##kernel); \
    ss = _mm_add_ps(ss, _mm_mul_ps(pix0##row, kernel1##kernel)); \
    ss = _mm_add_ps(ss, _mm_mul_ps(pix1##row, kernel2##kernel));

#define MM_KERNEL1x3_LOAD(row, x) \
    pix0##row = _mm_cvtepi32_ps(mm_cvtepu8_epi32(&in1[x])); \
    pix1##row = _mm_cvtepi32_ps(mm_cvtepu8_epi32(&in0[x])); \
    pix2##row = _mm_cvtepi32_ps(mm_cvtepu8_epi32(&in_1[x]));

    __m128 kernel00 = _mm_set_ps(0, kernel[2], kernel[1], kernel[0]);
    __m128 kernel10 = _mm_set_ps(0, kernel[5], kernel[4], kernel[3]);
    __m128 kernel20 = _mm_set_ps(0, kernel[8], kernel[7], kernel[6]);
    __m128 kernel01 = _mm_set_ps(kernel[2], kernel[1], kernel[0], 0);
    __m128 kernel11 = _mm_set_ps(kernel[5], kernel[4], kernel[3], 0);
    __m128 kernel21 = _mm_set_ps(kernel[8], kernel[7], kernel[6], 0);
    __m128 mm_offset = _mm_set1_ps(offset);

    memcpy(imOut->image8[0], im->image8[0], im->linesize);
    y = 1;
    for (; y < im->ysize-1-1; y += 2) {
        UINT8* in_1 = im->image8[y-1];
        UINT8* in0 = im->image8[y];
        UINT8* in1 = im->image8[y+1];
        UINT8* in2 = im->image8[y+2];
        UINT8* out0 = imOut->image8[y];
        UINT8* out1 = imOut->image8[y+1];

        out0[0] = in0[0];
        out1[0] = in1[0];
        x = 1;
        for (; x < im->xsize-1-3; x += 4) {
            __m128 ss0, ss1, ss2, ss3, ss4, ss5;
            __m128 pix00, pix10, pix20, pix30;
            __m128i ssi0;

            MM_KERNEL1x3_LOAD(0, x-1);
            MM_KERNEL1x3_SUM1(ss0, 0, 0);
            MM_KERNEL1x3_SUM1(ss1, 0, 1);
            ss0 = _mm_hadd_ps(ss0, ss1);
            pix30 = _mm_cvtepi32_ps(mm_cvtepu8_epi32(&in2[x-1]));
            MM_KERNEL1x3_SUM1_2(ss3, 0, 0);
            MM_KERNEL1x3_SUM1_2(ss4, 0, 1);
            ss3 = _mm_hadd_ps(ss3, ss4);

            MM_KERNEL1x3_LOAD(0, x+1);
            MM_KERNEL1x3_SUM1(ss1, 0, 0);
            MM_KERNEL1x3_SUM1(ss2, 0, 1);
            ss1 = _mm_hadd_ps(ss1, ss2);
            pix30 = _mm_cvtepi32_ps(mm_cvtepu8_epi32(&in2[x+1]));
            MM_KERNEL1x3_SUM1_2(ss4, 0, 0);
            MM_KERNEL1x3_SUM1_2(ss5, 0, 1);
            ss4 = _mm_hadd_ps(ss4, ss5);

            ss0 = _mm_hadd_ps(ss0, ss1);
            ss0 = _mm_add_ps(ss0, mm_offset);
            ssi0 = _mm_cvtps_epi32(ss0);
            ssi0 = _mm_packs_epi32(ssi0, ssi0);
            ssi0 = _mm_packus_epi16(ssi0, ssi0);
            *((UINT32*) &out0[x]) = _mm_cvtsi128_si32(ssi0);

            ss3 = _mm_hadd_ps(ss3, ss4);
            ss3 = _mm_add_ps(ss3, mm_offset);
            ssi0 = _mm_cvtps_epi32(ss3);
            ssi0 = _mm_packs_epi32(ssi0, ssi0);
            ssi0 = _mm_packus_epi16(ssi0, ssi0);
            *((UINT32*) &out1[x]) = _mm_cvtsi128_si32(ssi0);
        }
        for (; x < im->xsize-1; x++) {
            __m128 ss0, ss1;
            __m128 pix00, pix10, pix20, pix30;
            __m128i ssi0;

            pix00 = _mm_set_ps(0, in1[x+1], in1[x], in1[x-1]);
            pix10 = _mm_set_ps(0, in0[x+1], in0[x], in0[x-1]);
            pix20 = _mm_set_ps(0, in_1[x+1], in_1[x], in_1[x-1]);
            pix30 = _mm_set_ps(0, in2[x+1], in2[x], in2[x-1]);
            MM_KERNEL1x3_SUM1(ss0, 0, 0);
            MM_KERNEL1x3_SUM1_2(ss1, 0, 0);

            ss0 = _mm_hadd_ps(ss0, ss0);
            ss0 = _mm_hadd_ps(ss0, ss0);
            ss0 = _mm_add_ps(ss0, mm_offset);
            ssi0 = _mm_cvtps_epi32(ss0);
            ssi0 = _mm_packs_epi32(ssi0, ssi0);
            ssi0 = _mm_packus_epi16(ssi0, ssi0);
            out0[x] = _mm_cvtsi128_si32(ssi0);

            ss1 = _mm_hadd_ps(ss1, ss1);
            ss1 = _mm_hadd_ps(ss1, ss1);
            ss1 = _mm_add_ps(ss1, mm_offset);
            ssi0 = _mm_cvtps_epi32(ss1);
            ssi0 = _mm_packs_epi32(ssi0, ssi0);
            ssi0 = _mm_packus_epi16(ssi0, ssi0);
            out1[x] = _mm_cvtsi128_si32(ssi0);
        }
        out0[x] = in0[x];
        out1[x] = in1[x];
    }
    for (; y < im->ysize-1; y++) {
        UINT8* in_1 = im->image8[y-1];
        UINT8* in0 = im->image8[y];
        UINT8* in1 = im->image8[y+1];
        UINT8* out = imOut->image8[y];

        out[0] = in0[0];
        x = 1;
        for (; x < im->xsize-2; x++) {
            __m128 ss;
            __m128 pix00, pix10, pix20;
            __m128i ssi0;

            MM_KERNEL1x3_LOAD(0, x-1);
            MM_KERNEL1x3_SUM1(ss, 0, 0);

            ss = _mm_hadd_ps(ss, ss);
            ss = _mm_hadd_ps(ss, ss);
            ss = _mm_add_ps(ss, mm_offset);
            ssi0 = _mm_cvtps_epi32(ss);
            ssi0 = _mm_packs_epi32(ssi0, ssi0);
            ssi0 = _mm_packus_epi16(ssi0, ssi0);
            out[x] = _mm_cvtsi128_si32(ssi0);
        }
        for (; x < im->xsize-1; x++) {
            __m128 ss;
            __m128 pix00, pix10, pix20;
            __m128i ssi0;

            pix00 = _mm_set_ps(0, in1[x+1], in1[x], in1[x-1]);
            pix10 = _mm_set_ps(0, in0[x+1], in0[x], in0[x-1]);
            pix20 = _mm_set_ps(0, in_1[x+1], in_1[x], in_1[x-1]);
            MM_KERNEL1x3_SUM1(ss, 0, 0);

            ss = _mm_hadd_ps(ss, ss);
            ss = _mm_hadd_ps(ss, ss);
            ss = _mm_add_ps(ss, mm_offset);
            ssi0 = _mm_cvtps_epi32(ss);
            ssi0 = _mm_packs_epi32(ssi0, ssi0);
            ssi0 = _mm_packus_epi16(ssi0, ssi0);
            out[x] = _mm_cvtsi128_si32(ssi0);
        }
        out[x] = in0[x];
    }
    memcpy(imOut->image8[y], im->image8[y], im->linesize);

#undef MM_KERNEL1x3_SUM1
#undef MM_KERNEL1x3_SUM1_2
#undef MM_KERNEL1x3_LOAD

#elif defined(__riscv_vector)

    memcpy(imOut->image8[0], im->image8[0], im->linesize);
    for (y = 1; y < im->ysize-1; y++) {
        UINT8* in_1 = im->image8[y-1];
        UINT8* in0 = im->image8[y];
        UINT8* in1 = im->image8[y+1];
        UINT8* out = imOut->image8[y];

        out[0] = in0[0];
        int npixels = im->xsize - 2; /* number of interior pixels */
        x = 1;
        for (int done = 0; done < npixels; ) {
            size_t vl = __riscv_vsetvl_e32m4(npixels - done);

            vfloat32m4_t acc = __riscv_vfmv_v_f_f32m4(offset, vl);

            /* Row y+1 (in1) * kernel row 0 */
            vuint8m1_t raw;
            vuint16m2_t w16;
            vuint32m4_t w32;
            vfloat32m4_t fp;

            raw = __riscv_vle8_v_u8m1(&in1[x - 1], vl);
            w16 = __riscv_vzext_vf2_u16m2(raw, vl);
            w32 = __riscv_vzext_vf2_u32m4(w16, vl);
            fp = __riscv_vfcvt_f_xu_v_f32m4(w32, vl);
            acc = __riscv_vfmacc_vf_f32m4(acc, kernel[0], fp, vl);

            raw = __riscv_vle8_v_u8m1(&in1[x], vl);
            w16 = __riscv_vzext_vf2_u16m2(raw, vl);
            w32 = __riscv_vzext_vf2_u32m4(w16, vl);
            fp = __riscv_vfcvt_f_xu_v_f32m4(w32, vl);
            acc = __riscv_vfmacc_vf_f32m4(acc, kernel[1], fp, vl);

            raw = __riscv_vle8_v_u8m1(&in1[x + 1], vl);
            w16 = __riscv_vzext_vf2_u16m2(raw, vl);
            w32 = __riscv_vzext_vf2_u32m4(w16, vl);
            fp = __riscv_vfcvt_f_xu_v_f32m4(w32, vl);
            acc = __riscv_vfmacc_vf_f32m4(acc, kernel[2], fp, vl);

            /* Row y (in0) * kernel row 1 */
            raw = __riscv_vle8_v_u8m1(&in0[x - 1], vl);
            w16 = __riscv_vzext_vf2_u16m2(raw, vl);
            w32 = __riscv_vzext_vf2_u32m4(w16, vl);
            fp = __riscv_vfcvt_f_xu_v_f32m4(w32, vl);
            acc = __riscv_vfmacc_vf_f32m4(acc, kernel[3], fp, vl);

            raw = __riscv_vle8_v_u8m1(&in0[x], vl);
            w16 = __riscv_vzext_vf2_u16m2(raw, vl);
            w32 = __riscv_vzext_vf2_u32m4(w16, vl);
            fp = __riscv_vfcvt_f_xu_v_f32m4(w32, vl);
            acc = __riscv_vfmacc_vf_f32m4(acc, kernel[4], fp, vl);

            raw = __riscv_vle8_v_u8m1(&in0[x + 1], vl);
            w16 = __riscv_vzext_vf2_u16m2(raw, vl);
            w32 = __riscv_vzext_vf2_u32m4(w16, vl);
            fp = __riscv_vfcvt_f_xu_v_f32m4(w32, vl);
            acc = __riscv_vfmacc_vf_f32m4(acc, kernel[5], fp, vl);

            /* Row y-1 (in_1) * kernel row 2 */
            raw = __riscv_vle8_v_u8m1(&in_1[x - 1], vl);
            w16 = __riscv_vzext_vf2_u16m2(raw, vl);
            w32 = __riscv_vzext_vf2_u32m4(w16, vl);
            fp = __riscv_vfcvt_f_xu_v_f32m4(w32, vl);
            acc = __riscv_vfmacc_vf_f32m4(acc, kernel[6], fp, vl);

            raw = __riscv_vle8_v_u8m1(&in_1[x], vl);
            w16 = __riscv_vzext_vf2_u16m2(raw, vl);
            w32 = __riscv_vzext_vf2_u32m4(w16, vl);
            fp = __riscv_vfcvt_f_xu_v_f32m4(w32, vl);
            acc = __riscv_vfmacc_vf_f32m4(acc, kernel[7], fp, vl);

            raw = __riscv_vle8_v_u8m1(&in_1[x + 1], vl);
            w16 = __riscv_vzext_vf2_u16m2(raw, vl);
            w32 = __riscv_vzext_vf2_u32m4(w16, vl);
            fp = __riscv_vfcvt_f_xu_v_f32m4(w32, vl);
            acc = __riscv_vfmacc_vf_f32m4(acc, kernel[8], fp, vl);

            /* Convert f32 -> u32 with rounding (round-to-nearest via default FRM) */
            vuint32m4_t ures = __riscv_vfcvt_xu_f_v_u32m4(acc, vl);
            /* Narrow u32 -> u16 -> u8 with saturation, then store */
            vuint16m2_t n16 = RVV_VNCLIPU(u16m2, ures, 0, vl);
            vuint8m1_t n8 = RVV_VNCLIPU(u8m1, n16, 0, vl);
            __riscv_vse8_v_u8m1(&out[x], n8, vl);

            x += vl;
            done += vl;
        }
        out[im->xsize - 1] = in0[im->xsize - 1];
    }
    memcpy(imOut->image8[y], im->image8[y], im->linesize);

#else /* scalar fallback */

    memcpy(imOut->image8[0], im->image8[0], im->linesize);
    for (y = 1; y < im->ysize-1; y++) {
        UINT8* in_1 = im->image8[y-1];
        UINT8* in0 = im->image8[y];
        UINT8* in1 = im->image8[y+1];
        UINT8* out = imOut->image8[y];

        out[0] = in0[0];
        for (x = 1; x < im->xsize-1; x++) {
            float sum = offset;
            sum += (float)in1[x-1] * kernel[0];
            sum += (float)in1[x]   * kernel[1];
            sum += (float)in1[x+1] * kernel[2];
            sum += (float)in0[x-1] * kernel[3];
            sum += (float)in0[x]   * kernel[4];
            sum += (float)in0[x+1] * kernel[5];
            sum += (float)in_1[x-1] * kernel[6];
            sum += (float)in_1[x]   * kernel[7];
            sum += (float)in_1[x+1] * kernel[8];

            int val = (int)(sum + 0.5f);
            if (val < 0) val = 0;
            if (val > 255) val = 255;
            out[x] = (UINT8)val;
        }
        out[x] = in0[x];
    }
    memcpy(imOut->image8[y], im->image8[y], im->linesize);

#endif
}
