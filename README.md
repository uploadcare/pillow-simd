# Pillow-SIMD

Pillow-SIMD is "following" the Pillow fork (which is a PIL's fork itself).

For more information on the original Pillow, please refer to:
[read the documentation][original-docs],
[check the changelog][original-changelog] and
[find out how to contribute][original-contribute].


## Why SIMD

There are multiple ways to tweak image processing performance.
To name a few, such ways can be: utilizing better algorithms, optimizing existing implementations, using more processing power and/or resources. One of the great examples of using a more efficient algorithm is [replacing][gaussian-blur-changes] a convoloution-based Gaussian blur with a sequential-box one. Such examples are rather rare though. It is also known that certain processes might be optimized by using parallel processing to run the respective routines. But a more practical key to optimizations might be making things work faster using the resources at hand. For instance, SIMD computing might be the case.

SIMD stands for "single instruction, multiple data" and it's essence is in performing the same operation on multiple data points simultaneously by using multiple processing elements. Common CPU SIMD instruction sets are: MMX, SSE-SSE4, AVX, AVX2, AVX512, NEON.

Currently, Pillow-SIMD can be [compiled](#installation) with SSE4 (default) and **or???** AVX2 support.

## Status

Pillow-SIMD project is a production-ready implementation of SIMD computing into image processing.
The project is mostly sponsored by Uploadcare, a SAAS for cloud-based image storing and processing.
[![Uploadcare][uploadcare.logo]][uploadcare.com]
Uploadcare itself has been running Pillow-SIMD for about **how many years** now.

The following Uploadcare image operations are currently SIMD-accelerated:

- Resize (convolution-based resampling): SSE4, AVX2
- Gaussian and box blur: SSE4
- Alpha composition: SSE4, AVX2
- RGBA → RGBa (alpha premultiplication): SSE4, AVX2
- RGBa → RGBA (division by alpha): AVX2

See [CHANGES](CHANGES.SIMD.rst) for more information.


## Benchmarks

In order for you to clearly assess the productivity of implementing SIMD computing into Pillow image processing, we ran a number of benchmarks. The respective results can be found in the table below. The numbers represent processing rates in megapixels per second (Mpx/s). For instance, the rate at which a 2560x1600 RGB image is processed in 0.5 seconds equals to 8.2 Mpx/s.
Here are the instruments we've been up to during the benchmarks:

- Skia 53
- ImageMagick 6.9.3-8 Q8 x86_64
- Pillow 3.3.0
- Pillow-SIMD 3.3.0.post1

Now, let's proceed to the numbers (the more — the better):

Operation               | Filter  | IM   | Pillow| SIMD SSE4| SIMD AVX2| Skia 53
------------------------|---------|------|-------|----------|----------|--------
**Resize to 16x16**     | Bilinear| 41.37| 337.12|    571.67|    903.40|  809.49
                        | Bicubic | 20.58| 185.79|    305.72|    552.85|  453.10
                        | Lanczos | 14.17| 113.27|    189.19|    355.40|  292.57
**Resize to 320x180**   | Bilinear| 29.46| 209.06|    366.33|    558.57|  592.76
                        | Bicubic | 15.75| 124.43|    224.91|    353.53|  327.68
                        | Lanczos | 10.80|  82.25|    153.10|    244.22|  196.92
**Resize to 1920x1200** | Bilinear| 17.80|  55.87|    131.27|    152.11|  192.30
                        | Bicubic |  9.99|  43.64|     90.20|    112.34|  112.84
                        | Lanczos |  6.95|  34.51|     72.55|    103.16|  104.76
**Resize to 7712x4352** | Bilinear|  2.54|   6.71|     16.06|     20.33|   20.58
                        | Bicubic |  1.60|   5.51|     12.65|     16.46|   16.52
                        | Lanczos |  1.09|   4.62|      9.84|     13.38|   12.05
**Blur**                | 1px     |  6.60|  16.94|     35.16|          |        
                        | 10px    |  2.28|  16.94|     35.47|          |        
                        | 100px   |  0.34|  16.93|     35.53|          |        


### A brief conclusion

The results show that Pillow itself is generally faster than ImageMagick while Pillow-SIMD is even faster than the original Pillow by the factor of 2.0 – 2.5. In general, Pillow-SIMD with AVX2 is always **8 to 20 times faster** than ImageMagick and is almost equivalent in speed to the Skia, the high-speed graphics library used in Chromium.

### Methodology

All rates were measured using the following setup: Ubuntu 14.04 64-bit, single-thread AVX2-enabled intel i5 4258U CPU.
ImageMagick performance was measured with the `convert` command-line tool followed by `-verbose` and `-bench` arguments.
Such approach was used because there's usually a need in testing the latest software versions and command-line is the easiest way to do that.
All the routines involved with the testing procedure produced identic results.
Resizing filters compliance:

- PIL.Image.BILINEAR == Triangle
- PIL.Image.BICUBIC == Catrom
- PIL.Image.LANCZOS == Lanczos

In ImageMagick, Gaussian blur operation invokes two parameters: the first is called 'radius' and the second is called 'sigma'.
In fact, in order for the blur operation to be Gaussian, there should be no additional parameters. When the radius value is too small the blur procedure ceases to be Gaussian and if the value is excessively big the operation gets slown down with zero benefits in exchange. For the benchmarking purposes the radius was set to sigma × 2.5.

Following script was used for the benchmarking procedure:
https://gist.github.com/homm/f9b8d8a84a57a7e51f9c2a5828e40e63


## Why Pillow itself is so fast

No cheats involved. We've used identical high-quality resize and blur methods for the benchmark. Outcomes produced by different libraries are in almost pixel-perfect agreement. The difference in measured rates is only provided with the performance of every involved algorithm. Resampling for Pillow 2.7 was rewritten with minimal usage of floating point calculations, precomputed coefficients and cache-awareness transposition.

## Why Pillow-SIMD is even faster

Because of the SIMD computing, of course. Let us share som ideas on how to achieve even better performance.

- **Efficient memory operation** Currently, each single pixel is read from the dynamic memory and written to a single SSE register, while every SSE register can handle up to four pixels simultaneously.
- **Integer-based arithmetic** Experiments show that an integer-based arithmetic does not affect the resulting image quality while it still provides for up to 50% increase in the performance of non-SIMD code.
- **Aligned pixels allocation** It is rather well-known that the SIMD 'load' and 'store' commands work better with aligned memory.


## Why do not contribute SIMD to the original Pillow

Well, it's not that simple. First of all, the original Pillow supports a large number of architectures, not just x86.
But even for x86 platforms, Pillow is often distributed via precompiled binaries.
In order for us to integrate SIMD into the precompiled binaries we'd need to execute runtime CPU capabilities checks.
To compile the code this way we need to pass the `-mavx2` option to the compiler. This, in turn, automatically activates all `if (__AVX2__)` and below conditions. 

**explanation needed**
And SIMD instructions under such conditions exist even in standard C library and they do not have any runtime checks.
Currently, I don't know how to allow SIMD instructions in the code
but *do not allow* such instructions without runtime checks.


## Installation

If there's a copy of the original Pillow installed, it has to be removed first.
In general, you need to run `pip install pillow-simd`, and if you're using SSE4-capable CPU everything should run smoothly.
If you'd like to install the AVX2-enabled version, you need to pass the additional flag to a C compiler. The easiest way to do so is to define the `CC` variable prior to **while? рили?** compilation.

```bash
$ pip uninstall pillow
$ CC="cc -mavx2" pip install -U --force-reinstall pillow-simd
```


## Contributing to Pillow-SIMD

Please be aware that Pillow-SIMD and Pillow are two separate projects.
Please submit bugs and improvements not related to SIMD to the [original Pillow][original-issues].
All bugs **точно надо это?** and fixes of of the original Pillow will then be transferred to the next Pillow-SIMD version automatically.

  [original-docs]: http://pillow.readthedocs.io/
  [original-issues]: https://github.com/python-pillow/Pillow/issues/new
  [original-changelog]: https://github.com/python-pillow/Pillow/blob/master/CHANGES.rst
  [original-contribute]: https://github.com/python-pillow/Pillow/blob/master/.github/CONTRIBUTING.md
  [gaussian-blur-changes]: http://pillow.readthedocs.io/en/3.2.x/releasenotes/2.7.0.html#gaussian-blur-and-unsharp-mask
  [uploadcare.com]: https://uploadcare.com/?utm_source=github&utm_medium=description&utm_campaign=pillow-simd
  [uploadcare.logo]: https://ucarecdn.com/dc4b8363-e89f-402f-8ea8-ce606664069c/-/preview/
