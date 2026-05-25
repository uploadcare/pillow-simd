void
ImagingFilter5x5f_u8(Imaging imOut, Imaging im, const float* kernel,
                     float offset)
{
    int x, y;

#if defined(__SSE4_2__)

#define MM_KERNEL1x5_LOAD(row, x) \
    pix0##row = _mm_cvtepi32_ps(mm_cvtepu8_epi32(&in2[x])); \
    pix1##row = _mm_cvtepi32_ps(mm_cvtepu8_epi32(&in1[x])); \
    pix2##row = _mm_cvtepi32_ps(mm_cvtepu8_epi32(&in0[x])); \
    pix3##row = _mm_cvtepi32_ps(mm_cvtepu8_epi32(&in_1[x])); \
    pix4##row = _mm_cvtepi32_ps(mm_cvtepu8_epi32(&in_2[x]));

#define MM_KERNEL1x5_SUM(ss, row, krow) \
    ss = _mm_mul_ps(pix0##row, kernel0##krow); \
    ss = _mm_add_ps(ss, _mm_mul_ps(pix1##row, kernel1##krow)); \
    ss = _mm_add_ps(ss, _mm_mul_ps(pix2##row, kernel2##krow)); \
    ss = _mm_add_ps(ss, _mm_mul_ps(pix3##row, kernel3##krow)); \
    ss = _mm_add_ps(ss, _mm_mul_ps(pix4##row, kernel4##krow));

    memcpy(imOut->image8[0], im->image8[0], im->linesize);
    memcpy(imOut->image8[1], im->image8[1], im->linesize);
    for (y = 2; y < im->ysize-2; y++) {
        UINT8* in_2 = im->image8[y-2];
        UINT8* in_1 = im->image8[y-1];
        UINT8* in0 = im->image8[y];
        UINT8* in1 = im->image8[y+1];
        UINT8* in2 = im->image8[y+2];
        UINT8* out = imOut->image8[y];
        __m128 kernelx0 = _mm_set_ps(kernel[15], kernel[10], kernel[5], kernel[0]);
        __m128 kernel00 = _mm_loadu_ps(&kernel[0+1]);
        __m128 kernel10 = _mm_loadu_ps(&kernel[5+1]);
        __m128 kernel20 = _mm_loadu_ps(&kernel[10+1]);
        __m128 kernel30 = _mm_loadu_ps(&kernel[15+1]);
        __m128 kernel40 = _mm_loadu_ps(&kernel[20+1]);
        __m128 kernelx1 = _mm_set_ps(kernel[19], kernel[14], kernel[9], kernel[4]);
        __m128 kernel01 = _mm_loadu_ps(&kernel[0+0]);
        __m128 kernel11 = _mm_loadu_ps(&kernel[5+0]);
        __m128 kernel21 = _mm_loadu_ps(&kernel[10+0]);
        __m128 kernel31 = _mm_loadu_ps(&kernel[15+0]);
        __m128 kernel41 = _mm_loadu_ps(&kernel[20+0]);

        out[0] = in0[0];
        out[1] = in0[1];
        x = 2;
        for (; x < im->xsize-2-1; x += 2) {
            __m128 ss0, ss1;
            __m128i ssi0;
            __m128 pix00, pix10, pix20, pix30, pix40;

            MM_KERNEL1x5_LOAD(0, x-1);
            MM_KERNEL1x5_SUM(ss0, 0, 0);
            ss0 = _mm_add_ps(ss0, _mm_mul_ps(
                _mm_set_ps(in_1[x-2], in0[x-2], in1[x-2], in2[x-2]),
                kernelx0));

            MM_KERNEL1x5_SUM(ss1, 0, 1);
            ss1 = _mm_add_ps(ss1, _mm_mul_ps(
                _mm_set_ps(in_1[x+3], in0[x+3], in1[x+3], in2[x+3]),
                kernelx1));

            ss0 = _mm_hadd_ps(ss0, ss1);
            ss0 = _mm_add_ps(ss0, _mm_set_ps(
                offset, kernel[24] * in_2[x+3],
                offset, kernel[20] * in_2[x-2]));
            ss0 = _mm_hadd_ps(ss0, ss0);
            ssi0 = _mm_cvtps_epi32(ss0);
            ssi0 = _mm_packs_epi32(ssi0, ssi0);
            ssi0 = _mm_packus_epi16(ssi0, ssi0);
            *((UINT32*) &out[x]) = _mm_cvtsi128_si32(ssi0);
        }
        for (; x < im->xsize-2; x ++) {
            __m128 ss0;
            __m128i ssi0;
            __m128 pix00, pix10, pix20, pix30, pix40;

            MM_KERNEL1x5_LOAD(0, x-2);
            MM_KERNEL1x5_SUM(ss0, 0, 1);
            ss0 = _mm_add_ps(ss0, _mm_mul_ps(
                _mm_set_ps(in_1[x+2], in0[x+2], in1[x+2], in2[x+2]),
                kernelx1));

            ss0 = _mm_add_ps(ss0, _mm_set_ps(
                0, 0, offset, kernel[24] * in_2[x+2]));
            ss0 = _mm_hadd_ps(ss0, ss0);
            ss0 = _mm_hadd_ps(ss0, ss0);
            ssi0 = _mm_cvtps_epi32(ss0);
            ssi0 = _mm_packs_epi32(ssi0, ssi0);
            ssi0 = _mm_packus_epi16(ssi0, ssi0);
            out[x] = _mm_cvtsi128_si32(ssi0);
        }
        out[x+0] = in0[x+0];
        out[x+1] = in0[x+1];
    }
    memcpy(imOut->image8[y], im->image8[y], im->linesize);
    memcpy(imOut->image8[y+1], im->image8[y+1], im->linesize);

#undef MM_KERNEL1x5_LOAD
#undef MM_KERNEL1x5_SUM

#elif defined(__riscv_vector)

    memcpy(imOut->image8[0], im->image8[0], im->linesize);
    memcpy(imOut->image8[1], im->image8[1], im->linesize);
    for (y = 2; y < im->ysize-2; y++) {
        UINT8* in_2 = im->image8[y-2];
        UINT8* in_1 = im->image8[y-1];
        UINT8* in0 = im->image8[y];
        UINT8* in1 = im->image8[y+1];
        UINT8* in2 = im->image8[y+2];
        UINT8* out = imOut->image8[y];
        UINT8* rows[5] = {in2, in1, in0, in_1, in_2};

        out[0] = in0[0];
        out[1] = in0[1];
        int npixels = im->xsize - 4; /* number of interior pixels */
        x = 2;
        for (int done = 0; done < npixels; ) {
            size_t vl = __riscv_vsetvl_e32m4(npixels - done);

            vfloat32m4_t acc = __riscv_vfmv_v_f_f32m4(offset, vl);

            int ky, kx;
            for (ky = 0; ky < 5; ky++) {
                for (kx = 0; kx < 5; kx++) {
                    vuint8m1_t raw = __riscv_vle8_v_u8m1(&rows[ky][x - 2 + kx], vl);
                    vuint16m2_t w16 = __riscv_vzext_vf2_u16m2(raw, vl);
                    vuint32m4_t w32 = __riscv_vzext_vf2_u32m4(w16, vl);
                    vfloat32m4_t fp = __riscv_vfcvt_f_xu_v_f32m4(w32, vl);
                    acc = __riscv_vfmacc_vf_f32m4(acc, kernel[ky * 5 + kx], fp, vl);
                }
            }

            /* Convert f32 -> u32, narrow to u8 with saturation */
            vuint32m4_t ures = __riscv_vfcvt_xu_f_v_u32m4(acc, vl);
            vuint16m2_t n16 = RVV_VNCLIPU(u16m2, ures, 0, vl);
            vuint8m1_t n8 = RVV_VNCLIPU(u8m1, n16, 0, vl);
            __riscv_vse8_v_u8m1(&out[x], n8, vl);

            x += vl;
            done += vl;
        }
        out[im->xsize - 2] = in0[im->xsize - 2];
        out[im->xsize - 1] = in0[im->xsize - 1];
    }
    memcpy(imOut->image8[y], im->image8[y], im->linesize);
    memcpy(imOut->image8[y+1], im->image8[y+1], im->linesize);

#else /* scalar fallback */

    memcpy(imOut->image8[0], im->image8[0], im->linesize);
    memcpy(imOut->image8[1], im->image8[1], im->linesize);
    for (y = 2; y < im->ysize-2; y++) {
        UINT8* in_2 = im->image8[y-2];
        UINT8* in_1 = im->image8[y-1];
        UINT8* in0 = im->image8[y];
        UINT8* in1 = im->image8[y+1];
        UINT8* in2 = im->image8[y+2];
        UINT8* out = imOut->image8[y];

        out[0] = in0[0];
        out[1] = in0[1];
        for (x = 2; x < im->xsize-2; x++) {
            float sum = offset;
            int ky, kx;
            UINT8* rows[5] = {in2, in1, in0, in_1, in_2};
            for (ky = 0; ky < 5; ky++) {
                for (kx = 0; kx < 5; kx++) {
                    sum += (float)rows[ky][x - 2 + kx] * kernel[ky * 5 + kx];
                }
            }
            int val = (int)(sum + 0.5f);
            if (val < 0) val = 0;
            if (val > 255) val = 255;
            out[x] = (UINT8)val;
        }
        out[x+0] = in0[x+0];
        out[x+1] = in0[x+1];
    }
    memcpy(imOut->image8[y], im->image8[y], im->linesize);
    memcpy(imOut->image8[y+1], im->image8[y+1], im->linesize);

#endif
}
