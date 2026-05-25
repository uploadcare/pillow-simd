void
ImagingFilter5x5i_4u8(Imaging imOut, Imaging im, const INT16* kernel,
                      INT32 offset)
{
    int x, y;

    offset += 1 << (PRECISION_BITS-1);

    memcpy(imOut->image32[0], im->image32[0], im->linesize);
    memcpy(imOut->image32[1], im->image32[1], im->linesize);
    for (y = 2; y < im->ysize-2; y++) {
        INT32* in_2 = im->image32[y-2];
        INT32* in_1 = im->image32[y-1];
        INT32* in0 = im->image32[y];
        INT32* in1 = im->image32[y+1];
        INT32* in2 = im->image32[y+2];
        INT32* out = imOut->image32[y];
#if defined(__AVX2__)

    #define MM_KERNEL_LOAD(row, x) \
        pix0##row = _mm256_castsi128_si256(_mm_shuffle_epi8(_mm_loadu_si128((__m128i*) &in2[x]), shuffle)); \
        pix0##row = _mm256_inserti128_si256(pix0##row, _mm256_castsi256_si128(pix0##row), 1); \
        pix1##row = _mm256_castsi128_si256(_mm_shuffle_epi8(_mm_loadu_si128((__m128i*) &in1[x]), shuffle)); \
        pix1##row = _mm256_inserti128_si256(pix1##row, _mm256_castsi256_si128(pix1##row), 1); \
        pix2##row = _mm256_castsi128_si256(_mm_shuffle_epi8(_mm_loadu_si128((__m128i*) &in0[x]), shuffle)); \
        pix2##row = _mm256_inserti128_si256(pix2##row, _mm256_castsi256_si128(pix2##row), 1); \
        pix3##row = _mm256_castsi128_si256(_mm_shuffle_epi8(_mm_loadu_si128((__m128i*) &in_1[x]), shuffle)); \
        pix3##row = _mm256_inserti128_si256(pix3##row, _mm256_castsi256_si128(pix3##row), 1); \
        pix4##row = _mm256_castsi128_si256(_mm_shuffle_epi8(_mm_loadu_si128((__m128i*) &in_2[x]), shuffle)); \
        pix4##row = _mm256_inserti128_si256(pix4##row, _mm256_castsi256_si128(pix4##row), 1);

    #define MM_KERNEL_SUM(ss, row, unpack_epi8, krow, kctl) \
        ss = _mm256_add_epi32(ss, _mm256_madd_epi16( \
            unpack_epi8(pix0##row, zero), _mm256_shuffle_epi32(kernel0##krow, kctl))); \
        ss = _mm256_add_epi32(ss, _mm256_madd_epi16( \
            unpack_epi8(pix1##row, zero), _mm256_shuffle_epi32(kernel1##krow, kctl))); \
        ss = _mm256_add_epi32(ss, _mm256_madd_epi16( \
            unpack_epi8(pix2##row, zero), _mm256_shuffle_epi32(kernel2##krow, kctl))); \
        ss = _mm256_add_epi32(ss, _mm256_madd_epi16( \
            unpack_epi8(pix3##row, zero), _mm256_shuffle_epi32(kernel3##krow, kctl))); \
        ss = _mm256_add_epi32(ss, _mm256_madd_epi16( \
            unpack_epi8(pix4##row, zero), _mm256_shuffle_epi32(kernel4##krow, kctl)));

        __m128i shuffle = _mm_set_epi8(15,11,14,10,13,9,12,8, 7,3,6,2,5,1,4,0);
        __m256i kernel00 = _mm256_set_epi16(
            0, 0, kernel[4], kernel[3], kernel[2], kernel[1], kernel[0], 0,
            0, 0, 0, kernel[4], kernel[3], kernel[2], kernel[1], kernel[0]);
        __m256i kernel10 = _mm256_set_epi16(
            0, 0, kernel[9], kernel[8], kernel[7], kernel[6], kernel[5], 0,
            0, 0, 0, kernel[9], kernel[8], kernel[7], kernel[6], kernel[5]);
        __m256i kernel20 = _mm256_set_epi16(
            0, 0, kernel[14], kernel[13], kernel[12], kernel[11], kernel[10], 0,
            0, 0, 0, kernel[14], kernel[13], kernel[12], kernel[11], kernel[10]);
        __m256i kernel30 = _mm256_set_epi16(
            0, 0, kernel[19], kernel[18], kernel[17], kernel[16], kernel[15], 0,
            0, 0, 0, kernel[19], kernel[18], kernel[17], kernel[16], kernel[15]);
        __m256i kernel40 = _mm256_set_epi16(
            0, 0, kernel[24], kernel[23], kernel[22], kernel[21], kernel[20], 0,
            0, 0, 0, kernel[24], kernel[23], kernel[22], kernel[21], kernel[20]);
        __m256i zero = _mm256_setzero_si256();
        __m256i pix00, pix10, pix20, pix30, pix40;
        __m256i pix01, pix11, pix21, pix31, pix41;

        out[0] = in0[0];
        out[1] = in0[1];
        x = 2;

        MM_KERNEL_LOAD(0, 0);

        for (; x < im->xsize-2-3; x += 4) {
            __m256i ss0, ss2;
            __m128i ssout;

            ss0 = _mm256_set1_epi32(offset);
            ss2 = _mm256_set1_epi32(offset);

            MM_KERNEL_SUM(ss0, 0, _mm256_unpacklo_epi8, 0, 0x00);
            MM_KERNEL_SUM(ss0, 0, _mm256_unpackhi_epi8, 0, 0x55);
            MM_KERNEL_SUM(ss2, 0, _mm256_unpackhi_epi8, 0, 0x00);

            MM_KERNEL_LOAD(1, x+2);
            MM_KERNEL_SUM(ss0, 1, _mm256_unpacklo_epi8, 0, 0xaa);
            ss0 = _mm256_srai_epi32(ss0, PRECISION_BITS);
            MM_KERNEL_SUM(ss2, 1, _mm256_unpacklo_epi8, 0, 0x55);
            MM_KERNEL_SUM(ss2, 1, _mm256_unpackhi_epi8, 0, 0xaa);
            ss2 = _mm256_srai_epi32(ss2, PRECISION_BITS);

            pix00 = pix01;
            pix10 = pix11;
            pix20 = pix21;
            pix30 = pix31;
            pix40 = pix41;

            ss0 = _mm256_packs_epi32(ss0, ss2);
            ssout = _mm_packus_epi16(
                _mm256_extracti128_si256(ss0, 0),
                _mm256_extracti128_si256(ss0, 1));
            ssout = _mm_shuffle_epi32(ssout, 0xd8);
            _mm_storeu_si128((__m128i*) &out[x], ssout);
        }
        for (; x < im->xsize-2; x++) {
            __m256i ss0;

            MM_KERNEL_LOAD(0, x-2);
            pix01 = _mm256_castsi128_si256(_mm_shuffle_epi8(_mm_cvtsi32_si128(*(INT32*) &in2[x+2]), shuffle));
            pix01 = _mm256_inserti128_si256(pix01, _mm256_castsi256_si128(pix01), 1);
            pix11 = _mm256_castsi128_si256(_mm_shuffle_epi8(_mm_cvtsi32_si128(*(INT32*) &in1[x+2]), shuffle));
            pix11 = _mm256_inserti128_si256(pix11, _mm256_castsi256_si128(pix11), 1);
            pix21 = _mm256_castsi128_si256(_mm_shuffle_epi8(_mm_cvtsi32_si128(*(INT32*) &in0[x+2]), shuffle));
            pix21 = _mm256_inserti128_si256(pix21, _mm256_castsi256_si128(pix21), 1);
            pix31 = _mm256_castsi128_si256(_mm_shuffle_epi8(_mm_cvtsi32_si128(*(INT32*) &in_1[x+2]), shuffle));
            pix31 = _mm256_inserti128_si256(pix31, _mm256_castsi256_si128(pix31), 1);
            pix41 = _mm256_castsi128_si256(_mm_shuffle_epi8(_mm_cvtsi32_si128(*(INT32*) &in_2[x+2]), shuffle));
            pix41 = _mm256_inserti128_si256(pix41, _mm256_castsi256_si128(pix41), 1);

            ss0 = _mm256_set1_epi32(offset);
            MM_KERNEL_SUM(ss0, 0, _mm256_unpacklo_epi8, 0, 0x00);
            MM_KERNEL_SUM(ss0, 0, _mm256_unpackhi_epi8, 0, 0x55);
            MM_KERNEL_SUM(ss0, 1, _mm256_unpacklo_epi8, 0, 0xaa);
            ss0 = _mm256_srai_epi32(ss0, PRECISION_BITS);

            ss0 = _mm256_packs_epi32(ss0, ss0);
            ss0 = _mm256_packus_epi16(ss0, ss0);
            out[x] = _mm_cvtsi128_si32(_mm256_castsi256_si128(ss0));
        }
        out[x] = in0[x];
        out[x+1] = in0[x+1];
    #undef MM_KERNEL_LOAD
    #undef MM_KERNEL_SUM

#elif defined(__SSE4_2__)

    #define MM_KERNEL_LOAD(row, x) \
        pix0##row = _mm_shuffle_epi8(_mm_loadu_si128((__m128i*) &in2[x]), shuffle); \
        pix1##row = _mm_shuffle_epi8(_mm_loadu_si128((__m128i*) &in1[x]), shuffle); \
        pix2##row = _mm_shuffle_epi8(_mm_loadu_si128((__m128i*) &in0[x]), shuffle); \
        pix3##row = _mm_shuffle_epi8(_mm_loadu_si128((__m128i*) &in_1[x]), shuffle); \
        pix4##row = _mm_shuffle_epi8(_mm_loadu_si128((__m128i*) &in_2[x]), shuffle);

    #define MM_KERNEL_SUM(ss, row, unpack_epi8, krow, kctl) \
        ss = _mm_add_epi32(ss, _mm_madd_epi16( \
            unpack_epi8(pix0##row, zero), _mm_shuffle_epi32(kernel0##krow, kctl))); \
        ss = _mm_add_epi32(ss, _mm_madd_epi16( \
            unpack_epi8(pix1##row, zero), _mm_shuffle_epi32(kernel1##krow, kctl))); \
        ss = _mm_add_epi32(ss, _mm_madd_epi16( \
            unpack_epi8(pix2##row, zero), _mm_shuffle_epi32(kernel2##krow, kctl))); \
        ss = _mm_add_epi32(ss, _mm_madd_epi16( \
            unpack_epi8(pix3##row, zero), _mm_shuffle_epi32(kernel3##krow, kctl))); \
        ss = _mm_add_epi32(ss, _mm_madd_epi16( \
            unpack_epi8(pix4##row, zero), _mm_shuffle_epi32(kernel4##krow, kctl)));

        __m128i shuffle = _mm_set_epi8(15,11,14,10,13,9,12,8, 7,3,6,2,5,1,4,0);
        __m128i kernel00 = _mm_set_epi16(
            0, 0, 0, kernel[4], kernel[3], kernel[2], kernel[1], kernel[0]);
        __m128i kernel10 = _mm_set_epi16(
            0, 0, 0, kernel[9], kernel[8], kernel[7], kernel[6], kernel[5]);
        __m128i kernel20 = _mm_set_epi16(
            0, 0, 0, kernel[14], kernel[13], kernel[12], kernel[11], kernel[10]);
        __m128i kernel30 = _mm_set_epi16(
            0, 0, 0, kernel[19], kernel[18], kernel[17], kernel[16], kernel[15]);
        __m128i kernel40 = _mm_set_epi16(
            0, 0, 0, kernel[24], kernel[23], kernel[22], kernel[21], kernel[20]);
        __m128i kernel01 = _mm_set_epi16(
            0, 0, kernel[4], kernel[3], kernel[2], kernel[1], kernel[0], 0);
        __m128i kernel11 = _mm_set_epi16(
            0, 0, kernel[9], kernel[8], kernel[7], kernel[6], kernel[5], 0);
        __m128i kernel21 = _mm_set_epi16(
            0, 0, kernel[14], kernel[13], kernel[12], kernel[11], kernel[10], 0);
        __m128i kernel31 = _mm_set_epi16(
            0, 0, kernel[19], kernel[18], kernel[17], kernel[16], kernel[15], 0);
        __m128i kernel41 = _mm_set_epi16(
            0, 0, kernel[24], kernel[23], kernel[22], kernel[21], kernel[20], 0);
        __m128i zero = _mm_setzero_si128();
        __m128i pix00, pix10, pix20, pix30, pix40;
        __m128i pix01, pix11, pix21, pix31, pix41;

        out[0] = in0[0];
        out[1] = in0[1];
        x = 2;

        MM_KERNEL_LOAD(0, 0);

        for (; x < im->xsize-2-3; x += 4) {
            __m128i ss0, ss1, ss2, ss3;

            ss0 = _mm_set1_epi32(offset);
            ss1 = _mm_set1_epi32(offset);
            ss2 = _mm_set1_epi32(offset);
            ss3 = _mm_set1_epi32(offset);

            MM_KERNEL_SUM(ss0, 0, _mm_unpacklo_epi8, 0, 0x00);
            MM_KERNEL_SUM(ss0, 0, _mm_unpackhi_epi8, 0, 0x55);
            MM_KERNEL_SUM(ss1, 0, _mm_unpacklo_epi8, 1, 0x00);
            MM_KERNEL_SUM(ss1, 0, _mm_unpackhi_epi8, 1, 0x55);
            MM_KERNEL_SUM(ss2, 0, _mm_unpackhi_epi8, 0, 0x00);
            MM_KERNEL_SUM(ss3, 0, _mm_unpackhi_epi8, 1, 0x00);

            MM_KERNEL_LOAD(1, x+2);
            MM_KERNEL_SUM(ss0, 1, _mm_unpacklo_epi8, 0, 0xaa);
            ss0 = _mm_srai_epi32(ss0, PRECISION_BITS);
            MM_KERNEL_SUM(ss1, 1, _mm_unpacklo_epi8, 1, 0xaa);
            ss1 = _mm_srai_epi32(ss1, PRECISION_BITS);
            MM_KERNEL_SUM(ss2, 1, _mm_unpacklo_epi8, 0, 0x55);
            MM_KERNEL_SUM(ss2, 1, _mm_unpackhi_epi8, 0, 0xaa);
            ss2 = _mm_srai_epi32(ss2, PRECISION_BITS);
            MM_KERNEL_SUM(ss3, 1, _mm_unpacklo_epi8, 1, 0x55);
            MM_KERNEL_SUM(ss3, 1, _mm_unpackhi_epi8, 1, 0xaa);
            ss3 = _mm_srai_epi32(ss3, PRECISION_BITS);

            pix00 = pix01;
            pix10 = pix11;
            pix20 = pix21;
            pix30 = pix31;
            pix40 = pix41;

            ss0 = _mm_packs_epi32(ss0, ss1);
            ss2 = _mm_packs_epi32(ss2, ss3);
            ss0 = _mm_packus_epi16(ss0, ss2);
            _mm_storeu_si128((__m128i*) &out[x], ss0);
        }
        for (; x < im->xsize-2; x++) {
            __m128i ss0;

            MM_KERNEL_LOAD(0, x-2);
            pix01 = _mm_shuffle_epi8(_mm_cvtsi32_si128(*(INT32*) &in2[x+2]), shuffle);
            pix11 = _mm_shuffle_epi8(_mm_cvtsi32_si128(*(INT32*) &in1[x+2]), shuffle);
            pix21 = _mm_shuffle_epi8(_mm_cvtsi32_si128(*(INT32*) &in0[x+2]), shuffle);
            pix31 = _mm_shuffle_epi8(_mm_cvtsi32_si128(*(INT32*) &in_1[x+2]), shuffle);
            pix41 = _mm_shuffle_epi8(_mm_cvtsi32_si128(*(INT32*) &in_2[x+2]), shuffle);

            ss0 = _mm_set1_epi32(offset);
            MM_KERNEL_SUM(ss0, 0, _mm_unpacklo_epi8, 0, 0x00);
            MM_KERNEL_SUM(ss0, 0, _mm_unpackhi_epi8, 0, 0x55);
            MM_KERNEL_SUM(ss0, 1, _mm_unpacklo_epi8, 0, 0xaa);
            ss0 = _mm_srai_epi32(ss0, PRECISION_BITS);

            ss0 = _mm_packs_epi32(ss0, ss0);
            ss0 = _mm_packus_epi16(ss0, ss0);
            out[x] = _mm_cvtsi128_si32(ss0);
        }
        out[x] = in0[x];
        out[x+1] = in0[x+1];
    #undef MM_KERNEL_LOAD
    #undef MM_KERNEL_SUM

#elif defined(__riscv_vector)

        out[0] = in0[0];
        out[1] = in0[1];
        {
            INT32* rows[5] = {in2, in1, in0, in_1, in_2};
            int npixels = im->xsize - 4;
            x = 2;
            for (int done = 0; done < npixels; ) {
                size_t vl = __riscv_vsetvl_e32m4(npixels - done);

                vint32m4_t accR = __riscv_vmv_v_x_i32m4(offset, vl);
                vint32m4_t accG = __riscv_vmv_v_x_i32m4(offset, vl);
                vint32m4_t accB = __riscv_vmv_v_x_i32m4(offset, vl);
                vint32m4_t accA = __riscv_vmv_v_x_i32m4(offset, vl);

                int ky, kx;
                for (ky = 0; ky < 5; ky++) {
                    for (kx = 0; kx < 5; kx++) {
                        UINT8* base = (UINT8*)&rows[ky][x - 2 + kx];
                        INT32 kval = (INT32)kernel[ky * 5 + kx];
                        vuint8m1_t vr, vg, vb, va;
                        RVV_VLSEG4E8_U8M1(vr, vg, vb, va, (const uint8_t*)base, vl);

                        vuint16m2_t wr = __riscv_vzext_vf2_u16m2(vr, vl);
                        vuint32m4_t dr = __riscv_vzext_vf2_u32m4(wr, vl);
                        vint32m4_t sr = __riscv_vreinterpret_v_u32m4_i32m4(dr);
                        accR = __riscv_vmacc_vx_i32m4(accR, kval, sr, vl);

                        vuint16m2_t wg = __riscv_vzext_vf2_u16m2(vg, vl);
                        vuint32m4_t dg = __riscv_vzext_vf2_u32m4(wg, vl);
                        vint32m4_t sg = __riscv_vreinterpret_v_u32m4_i32m4(dg);
                        accG = __riscv_vmacc_vx_i32m4(accG, kval, sg, vl);

                        vuint16m2_t wb = __riscv_vzext_vf2_u16m2(vb, vl);
                        vuint32m4_t db = __riscv_vzext_vf2_u32m4(wb, vl);
                        vint32m4_t sb = __riscv_vreinterpret_v_u32m4_i32m4(db);
                        accB = __riscv_vmacc_vx_i32m4(accB, kval, sb, vl);

                        vuint16m2_t wa = __riscv_vzext_vf2_u16m2(va, vl);
                        vuint32m4_t da = __riscv_vzext_vf2_u32m4(wa, vl);
                        vint32m4_t sa = __riscv_vreinterpret_v_u32m4_i32m4(da);
                        accA = __riscv_vmacc_vx_i32m4(accA, kval, sa, vl);
                    }
                }

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
        }
        out[im->xsize - 2] = in0[im->xsize - 2];
        out[im->xsize - 1] = in0[im->xsize - 1];

#else /* scalar fallback */

        out[0] = in0[0];
        out[1] = in0[1];
        for (x = 2; x < im->xsize-2; x++) {
            int ch;
            UINT8* pOut = (UINT8*)&out[x];
            INT32* rows[5] = {in2, in1, in0, in_1, in_2};
            for (ch = 0; ch < 4; ch++) {
                INT32 ss = offset;
                int ky, kx;
                for (ky = 0; ky < 5; ky++) {
                    for (kx = 0; kx < 5; kx++) {
                        ss += ((UINT8*)&rows[ky][x - 2 + kx])[ch] * kernel[ky * 5 + kx];
                    }
                }
                ss >>= PRECISION_BITS;
                if (ss < 0) ss = 0;
                if (ss > 255) ss = 255;
                pOut[ch] = (UINT8)ss;
            }
        }
        out[x] = in0[x];
        out[x+1] = in0[x+1];

#endif
    }
    memcpy(imOut->image32[y], im->image32[y], im->linesize);
    memcpy(imOut->image32[y+1], im->image32[y+1], im->linesize);
}
