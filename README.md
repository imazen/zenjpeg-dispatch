# zenjpeg-dispatch

> **This repository is archived and no longer developed.**
>
> Encoder selection now happens through the picker stack — see
> [`zenpredict`/`zenpicker`](https://github.com/imazen/zenanalyze) for
> per-codec and cross-codec selection, and
> [imazen/zenjpeg](https://github.com/imazen/zenjpeg) for JPEG encoding itself.
>
> This repository no longer accepts issues or pull requests.

Dispatcher library for selecting optimal JPEG encoder based on image characteristics.

Intelligently selects between mozjpeg and jpegli encoders based on target quality and image content to achieve Pareto-optimal compression.

## Key insight

mozjpeg's trellis quantization excels at low quality (Q < 70), while jpegli's adaptive quantization excels at high quality (Q >= 70). zenjpeg-dispatch automatically selects the best encoder and parameters.

## Features

- Adaptive codec selection based on quality target and image content
- GPU-accelerated quality metrics (SSIMULACRA2, Butteraugli, DSSIM) via `--features gpu`
- Heuristic discovery benchmarking tool
- Multi-corpus evaluation support

## License

BSD-3-Clause
