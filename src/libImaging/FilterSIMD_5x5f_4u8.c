void
ImagingFilter5x5f_4u8(Imaging imOut, Imaging im, const float* kernel,
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

#define MM_KERNEL1x5_SUM(row, kindex) \
    ss = _mm_add_ps(ss, _mm_mul_ps(pix0##row, _mm_set1_ps(kernel[0 + kindex]))); \
    ss = _mm_add_ps(ss, _mm_mul_ps(pix1##row, _mm_set1_ps(kernel[5 + kindex]))); \
    ss = _mm_add_ps(ss, _mm_mul_ps(pix2##row, _mm_set1_ps(kernel[10 + kindex]))); \
    ss = _mm_add_ps(ss, _mm_mul_ps(pix3##row, _mm_set1_ps(kernel[15 + kindex]))); \
    ss = _mm_add_ps(ss, _mm_mul_ps(pix4##row, _mm_set1_ps(kernel[20 + kindex])));

    memcpy(imOut->image32[0], im->image32[0], im->linesize);
    memcpy(imOut->image32[1], im->image32[1], im->linesize);
    for (y = 2; y < im->ysize-2; y++) {
        INT32* in_2 = im->image32[y-2];
        INT32* in_1 = im->image32[y-1];
        INT32* in0 = im->image32[y];
        INT32* in1 = im->image32[y+1];
        INT32* in2 = im->image32[y+2];
        INT32* out = imOut->image32[y];
        __m128 pix00, pix10, pix20, pix30, pix40;
        __m128 pix01, pix11, pix21, pix31, pix41;
        __m128 pix02, pix12, pix22, pix32, pix42;
        __m128 pix03, pix13, pix23, pix33, pix43;
        __m128 pix04, pix14, pix24, pix34, pix44;
        MM_KERNEL1x5_LOAD(0, 0);
        MM_KERNEL1x5_LOAD(1, 1);
        MM_KERNEL1x5_LOAD(2, 2);
        MM_KERNEL1x5_LOAD(3, 3);

        out[0] = in0[0];
        out[1] = in0[1];
        x = 2;
        for (; x < im->xsize-2-4; x += 5) {
            __m128 ss;
            __m128i ssi0, ssi1, ssi2, ssi3;

            ss = _mm_set1_ps(offset);
            MM_KERNEL1x5_SUM(0, 0);
            MM_KERNEL1x5_SUM(1, 1);
            MM_KERNEL1x5_SUM(2, 2);
            MM_KERNEL1x5_SUM(3, 3);
            MM_KERNEL1x5_LOAD(4, x+2);
            MM_KERNEL1x5_SUM(4, 4);
            ssi0 = _mm_cvtps_epi32(ss);

            ss = _mm_set1_ps(offset);
            MM_KERNEL1x5_SUM(1, 0);
            MM_KERNEL1x5_SUM(2, 1);
            MM_KERNEL1x5_SUM(3, 2);
            MM_KERNEL1x5_SUM(4, 3);
            MM_KERNEL1x5_LOAD(0, x+3);
            MM_KERNEL1x5_SUM(0, 4);
            ssi1 = _mm_cvtps_epi32(ss);

            ss = _mm_set1_ps(offset);
            MM_KERNEL1x5_SUM(2, 0);
            MM_KERNEL1x5_SUM(3, 1);
            MM_KERNEL1x5_SUM(4, 2);
            MM_KERNEL1x5_SUM(0, 3);
            MM_KERNEL1x5_LOAD(1, x+4);
            MM_KERNEL1x5_SUM(1, 4);
            ssi2 = _mm_cvtps_epi32(ss);

            ss = _mm_set1_ps(offset);
            MM_KERNEL1x5_SUM(3, 0);
            MM_KERNEL1x5_SUM(4, 1);
            MM_KERNEL1x5_SUM(0, 2);
            MM_KERNEL1x5_SUM(1, 3);
            MM_KERNEL1x5_LOAD(2, x+5);
            MM_KERNEL1x5_SUM(2, 4);
            ssi3 = _mm_cvtps_epi32(ss);

            ssi0 = _mm_packs_epi32(ssi0, ssi1);
            ssi1 = _mm_packs_epi32(ssi2, ssi3);
            ssi0 = _mm_packus_epi16(ssi0, ssi1);
            _mm_storeu_si128((__m128i*) &out[x], ssi0);

            ss = _mm_set1_ps(offset);
            MM_KERNEL1x5_SUM(4, 0);
            MM_KERNEL1x5_SUM(0, 1);
            MM_KERNEL1x5_SUM(1, 2);
            MM_KERNEL1x5_SUM(2, 3);
            MM_KERNEL1x5_LOAD(3, x+6);
            MM_KERNEL1x5_SUM(3, 4);
            ssi0 = _mm_cvtps_epi32(ss);
            ssi0 = _mm_packs_epi32(ssi0, ssi0);
            ssi0 = _mm_packus_epi16(ssi0, ssi0);
            out[x+4] = _mm_cvtsi128_si32(ssi0);
        }
        for (; x < im->xsize-2; x++) {
            __m128 ss = _mm_set1_ps(offset);
            __m128i ssi0;
            MM_KERNEL1x5_LOAD(0, x-2);
            MM_KERNEL1x5_SUM(0, 0);
            MM_KERNEL1x5_LOAD(0, x-1);
            MM_KERNEL1x5_SUM(0, 1);
            MM_KERNEL1x5_LOAD(0, x+0);
            MM_KERNEL1x5_SUM(0, 2);
            MM_KERNEL1x5_LOAD(0, x+1);
            MM_KERNEL1x5_SUM(0, 3);
            MM_KERNEL1x5_LOAD(0, x+2);
            MM_KERNEL1x5_SUM(0, 4);
            ssi0 = _mm_cvtps_epi32(ss);
            ssi0 = _mm_packs_epi32(ssi0, ssi0);
            ssi0 = _mm_packus_epi16(ssi0, ssi0);
            out[x] = _mm_cvtsi128_si32(ssi0);
        }
        out[x] = in0[x];
        out[x+1] = in0[x+1];
    }
    memcpy(imOut->image32[y], im->image32[y], im->linesize);
    memcpy(imOut->image32[y+1], im->image32[y+1], im->linesize);

#undef MM_KERNEL1x5_LOAD
#undef MM_KERNEL1x5_SUM

#elif defined(__riscv_vector)

    memcpy(imOut->image32[0], im->image32[0], im->linesize);
    memcpy(imOut->image32[1], im->image32[1], im->linesize);
    for (y = 2; y < im->ysize-2; y++) {
        INT32* in_2 = im->image32[y-2];
        INT32* in_1 = im->image32[y-1];
        INT32* in0 = im->image32[y];
        INT32* in1 = im->image32[y+1];
        INT32* in2 = im->image32[y+2];
        INT32* out = imOut->image32[y];
        INT32* rows[5] = {in2, in1, in0, in_1, in_2};

        out[0] = in0[0];
        out[1] = in0[1];
        int npixels = im->xsize - 4;
        x = 2;
        for (int done = 0; done < npixels; ) {
            size_t vl = __riscv_vsetvl_e32m4(npixels - done);

            vfloat32m4_t accR = __riscv_vfmv_v_f_f32m4(offset, vl);
            vfloat32m4_t accG = __riscv_vfmv_v_f_f32m4(offset, vl);
            vfloat32m4_t accB = __riscv_vfmv_v_f_f32m4(offset, vl);
            vfloat32m4_t accA = __riscv_vfmv_v_f_f32m4(offset, vl);

            int ky, kx;
            for (ky = 0; ky < 5; ky++) {
                for (kx = 0; kx < 5; kx++) {
                    UINT8* base = (UINT8*)&rows[ky][x - 2 + kx];
                    vuint8m1_t vr, vg, vb, va;
                    RVV_VLSEG4E8_U8M1(vr, vg, vb, va, (const uint8_t*)base, vl);

                    float kval = kernel[ky * 5 + kx];

                    vuint16m2_t wr = __riscv_vzext_vf2_u16m2(vr, vl);
                    vuint32m4_t dr = __riscv_vzext_vf2_u32m4(wr, vl);
                    accR = __riscv_vfmacc_vf_f32m4(accR, kval, __riscv_vfcvt_f_xu_v_f32m4(dr, vl), vl);

                    vuint16m2_t wg = __riscv_vzext_vf2_u16m2(vg, vl);
                    vuint32m4_t dg = __riscv_vzext_vf2_u32m4(wg, vl);
                    accG = __riscv_vfmacc_vf_f32m4(accG, kval, __riscv_vfcvt_f_xu_v_f32m4(dg, vl), vl);

                    vuint16m2_t wb = __riscv_vzext_vf2_u16m2(vb, vl);
                    vuint32m4_t db = __riscv_vzext_vf2_u32m4(wb, vl);
                    accB = __riscv_vfmacc_vf_f32m4(accB, kval, __riscv_vfcvt_f_xu_v_f32m4(db, vl), vl);

                    vuint16m2_t wa = __riscv_vzext_vf2_u16m2(va, vl);
                    vuint32m4_t da = __riscv_vzext_vf2_u32m4(wa, vl);
                    accA = __riscv_vfmacc_vf_f32m4(accA, kval, __riscv_vfcvt_f_xu_v_f32m4(da, vl), vl);
                }
            }

            /* Convert f32 -> u32, narrow to u8, interleave-store */
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
        out[im->xsize - 2] = in0[im->xsize - 2];
        out[im->xsize - 1] = in0[im->xsize - 1];
    }
    memcpy(imOut->image32[y], im->image32[y], im->linesize);
    memcpy(imOut->image32[y+1], im->image32[y+1], im->linesize);

#else /* scalar fallback */

    memcpy(imOut->image32[0], im->image32[0], im->linesize);
    memcpy(imOut->image32[1], im->image32[1], im->linesize);
    for (y = 2; y < im->ysize-2; y++) {
        INT32* in_2 = im->image32[y-2];
        INT32* in_1 = im->image32[y-1];
        INT32* in0 = im->image32[y];
        INT32* in1 = im->image32[y+1];
        INT32* in2 = im->image32[y+2];
        INT32* out = imOut->image32[y];

        out[0] = in0[0];
        out[1] = in0[1];
        for (x = 2; x < im->xsize-2; x++) {
            int ch;
            UINT8* pOut = (UINT8*)&out[x];
            INT32* rows[5] = {in2, in1, in0, in_1, in_2};
            for (ch = 0; ch < 4; ch++) {
                float sum = offset;
                int ky, kx;
                for (ky = 0; ky < 5; ky++) {
                    for (kx = 0; kx < 5; kx++) {
                        sum += ((UINT8*)&rows[ky][x - 2 + kx])[ch] * kernel[ky * 5 + kx];
                    }
                }
                int val = (int)(sum + 0.5f);
                if (val < 0) val = 0;
                if (val > 255) val = 255;
                pOut[ch] = (UINT8)val;
            }
        }
        out[x] = in0[x];
        out[x+1] = in0[x+1];
    }
    memcpy(imOut->image32[y], im->image32[y], im->linesize);
    memcpy(imOut->image32[y+1], im->image32[y+1], im->linesize);

#endif
}
