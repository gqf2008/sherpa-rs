# Building

### Prerequisites

[Cargo](https://www.rust-lang.org/tools/install) | [Clang](https://releases.llvm.org/download.html) | [Cmake](https://cmake.org/download/)

### Linux

```console
sudo apt-get update
sudo apt-get install -y pkg-config build-essential clang cmake
```

### Prepare repository

```console
git clone https://github.com/thewh1teagle/sherpa-rs --recursive
cd sherpa-rs
```

### Build

```console
cargo build --release
```

### Instructions (for builds with `cuda` enabled)

1. Download [CUDA](https://developer.nvidia.com/cuda-downloads?target_os=Windows)
2. Download [Visual Studio with Desktop C++ and Clang enabled](https://visualstudio.microsoft.com/de/downloads/) (see clang link below for installer walkthrough)
3. Download [CLANG](https://www.wikihow.com/Install-Clang-on-Windows)
4. Download [CMAKE](https://cmake.org/download/)
5. Run `where.exe clang`, then `setx LIBCLANG_PATH "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\Llvm\x64\bin"` or something like that
6. Restart your shell!!!
7. Cargo build

### Resample wav file for 16khz

```console
ffmpeg -i <file> -ar 16000 -ac 1 -c:a pcm_s16le <out>
```

### Update sherpa-onnx

```console
cd sys/sherpa-onnx
git pull origin master
```
