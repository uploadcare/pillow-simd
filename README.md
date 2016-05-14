# Pillow-SIMD

Pillow-SIMD is "following" Pillow fork (which is PIL fork itself).

For more information about original Pillow, please
[read the documentation][original-docs],
[check the changelog][original-changelog] and
[find out how to contribute][original-contribute].


## Why SIMD

There are many ways to improve the performance of image processing.
You can use better algorithms for the same task, you can make better
implementation for current algorithms, or you can use more processing unit
resources. It is perfect when you can just use more efficient algorithm like
when gaussian blur based on convolutions [was replaced][gaussian-blur-changes]
by sequential box filters. But a number of such improvements are very limited.
It is also very tempting to use more processor unit resources 
(via parallelization) when they are available. But it is handier just
to make things faster on the same resources. And that is where SIMD works better.

SIMD stands for "single instruction, multiple data". This is a way to perform
same operations against the huge amount of homogeneous data. 
Modern CPU have different SIMD instructions sets like
MMX, SSE-SSE4, AVX, AVX2, AVX512, NEON.

Currently, Pillow-SIMD can be [compiled](#installation) with SSE4 (default)
and AVX2 support.


## Status

Pillow-SIMD can be used in production. Pillow-SIMD has been operating on
[Uploadcare][uploadcare.com] servers for more than 1 year.
Uploadcare is SAAS for image storing and processing in the cloud
and the main sponsor of Pillow-SIMD project.

Currently, following operations are accelerated:

- Resize (convolution-based resample): SSE4, AVX2
- Gaussian and box blur: SSE4


## Benchmarks

The numbers in the table represent processed megapixels of source image
per second. For example, if resize of 7712×4352 image is done in 0.5 seconds,
the result will be 67.1 Mpx/s.

- ImageMagick 6.9.3-8 Q8 x86_64
- Pillow 3.2.0
- Pillow-SIMD 3.2.0.post1

Source    | Operation               | Filter  | IM   | Pillow | SIMD SSE4 | SIMD AVX2 
----------|-------------------------|---------|------|--------|-----------|-----------
7712×4352 | **Resize to 16x16**     | Bilinear| 27.0 | 217    | 456       | 545
          |                         | Bicubic | 10.9 | 115    | 240       | 278
          |                         | Lanczos | 6.6  | 76.1   | 162       | 194
          | **Resize to 320x180**   | Bilinear| 32.0 | 166    | 354       | 410
          |                         | Bicubic | 16.5 | 92.3   | 198       | 204
          |                         | Lanczos | 11.0 | 63.2   | 133       | 147
          | **Resize to 2048x1155** | Bilinear| 20.7 | 87.6   | 202       | 217
          |                         | Bicubic | 12.2 | 65.7   | 126       | 130
          |                         | Lanczos | 8.7  | 41.3   | 88.2      | 95.6
          | **Blur**                | 1px     | 8.1  | 17.1   | 37.8
          |                         | 10px    | 2.6  | 17.4   | 39.0
          |                         | 100px   | 0.3  | 17.2   | 39.0
1920×1280 | **Resize to 16x16**     | Bilinear| 41.6 | 196    | 422       | 489
          |                         | Bicubic | 18.9 | 102    | 225       | 263
          |                         | Lanczos | 13.7 | 68.6   | 118       | 167
          | **Resize to 320x180**   | Bilinear| 27.6 | 111    | 196       | 197
          |                         | Bicubic | 14.5 | 66.3   | 154       | 162
          |                         | Lanczos | 9.8  | 44.3   | 102       | 107
          | **Resize to 2048x1155** | Bilinear| 9.1  | 20.7   | 71.3      | 72.6
          |                         | Bicubic | 6.3  | 16.9   | 49.3      | 54.3
          |                         | Lanczos | 4.7  | 14.6   | 36.8      | 40.6
          | **Blur**                | 1px     | 8.7  | 16.2   | 35.7
          |                         | 10px    | 2.8  | 16.7   | 35.4
          |                         | 100px   | 0.4  | 16.4   | 36.2


### Some conclusion

Pillow is always faster than ImageMagick. And Pillow-SIMD is faster
than Pillow in 2—2.5 times. In general, Pillow-SIMD with AVX2 almost always
**10 times faster** than ImageMagick.

### Methodology

All tests were performed on Ubuntu 14.04 64-bit running on
Intel Core i5 4258U with AVX2 CPU on the single thread.

ImageMagick performance was measured with command-line tool `convert` with
`-verbose` and `-bench` arguments. I use command line because
I need to test the latest version and this is the easiest way to do that.

All operations produce exactly the same results.
Resizing filters compliance:

- PIL.Image.BILINEAR == Triangle
- PIL.Image.BICUBIC == Catrom
- PIL.Image.LANCZOS == Lanczos

In ImageMagick, the radius of gaussian blur is called sigma and the second
parameter is called radius. In fact, there should not be additional parameters
for *gaussian blur*, because if the radius is too small, this is *not*
gaussian blur anymore. And if the radius is big this does not give any
advantages but makes operation slower. For the test, I set the radius
to sigma × 2.5.

Following script was used for testing:
https://gist.github.com/homm/f9b8d8a84a57a7e51f9c2a5828e40e63


## Why Pillow itself is so fast

There are no cheats. High-quality resize and blur methods are used for all
benchmarks. Results are almost pixel-perfect. The difference is only effective
algorithms. Resampling in Pillow was rewritten in version 2.7 with 
minimal usage of floating point numbers, precomputed coefficients and
cache-awareness transposition.


## Why Pillow-SIMD is even faster

Because of SIMD, of course. There are some ideas how to achieve even better
performance.

- **Efficient work with memory** Currently, each pixel is read from 
  memory to the SSE register, while every SSE register can handle
  four pixels at once.
- **Integer-based arithmetic** Experiments show that integer-based arithmetic
  does not affect the quality and increases the performance of non-SIMD code
  up to 50%.
- **Aligned pixels allocation** Well-known that the SIMD load and store
  commands work better with aligned memory.


## Why do not contribute SIMD to the original Pillow

Well, it's not that simple. First of all, Pillow supports a large number
of architectures, not only x86. But even for x86 platforms, Pillow is often
distributed via precompiled binaries. To integrate SIMD in precompiled binaries
we need to do runtime checks of CPU capabilities.
To compile the code with runtime checks we need to pass `-mavx2` option
to the compiler. However this automatically activates all `if (__AVX2__)`
and below conditions. And SIMD instructions under such conditions exist
even in standard C library and they do not have any runtime checks.
Currently, I don't know how to allow SIMD instructions in the code
but *do not allow* such instructions without runtime checks.


## Installation

In general, you need to do `pip install pillow-simd` as always and if you
are using SSE4-capable CPU everything should run smoothly.
Do not forget to remove original Pillow package first.

If you want the AVX2-enabled version, you need to pass the additional flag to C
compiler. The easiest way to do that is define `CC` variable while compilation.

```bash
$ pip uninstall pillow
$ CC="cc -mavx2" pip install -U --force-reinstall pillow-simd
```


## Contributing to Pillow-SIMD

Pillow-SIMD and Pillow are two separate projects.
Please submit bugs and improvements not related to SIMD to 
[original Pillow][original-issues]. All bugs and fixes in Pillow
will appear in next Pillow-SIMD version automatically.


  [original-docs]: http://pillow.readthedocs.io/
  [original-issues]: https://github.com/python-pillow/Pillow/issues/new
  [original-changelog]: https://github.com/python-pillow/Pillow/blob/master/CHANGES.rst
  [original-contribute]: https://github.com/python-pillow/Pillow/blob/master/.github/CONTRIBUTING.md
  [gaussian-blur-changes]: http://pillow.readthedocs.io/en/3.2.x/releasenotes/2.7.0.html#gaussian-blur-and-unsharp-mask
  [uploadcare.com]: https://uploadcare.com/?utm_source=github&utm_medium=description&utm_campaign=pillow-simd
