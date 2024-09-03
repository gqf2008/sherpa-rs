use cmake::Config;
use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

fn copy_folder(src: &Path, dst: &Path) {
    if let Err(err) = dircpy::copy_dir(src, dst) {
        panic!(
            "copy {} {} to {} {}, {}",
            src.display(),
            src.exists(),
            dst.display(),
            dst.exists(),
            err
        );
    };
}

fn main() {
    let target = env::var("TARGET").unwrap();
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    // let sherpa_dst = out_dir.join("sherpa-onnx");
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").expect("Failed to get CARGO_MANIFEST_DIR");
    let sherpa_src = Path::new(&manifest_dir).join("sherpa-onnx");
    let build_shared_libs = cfg!(feature = "directml") || cfg!(feature = "cuda");
    let profile = if cfg!(debug_assertions) {
        "Debug"
    } else {
        "Release"
    };

    // Prepare sherpa-onnx source
    // if !sherpa_dst.exists() {
    //     copy_folder(&sherpa_src, &out_dir);
    // }
    // Speed up build
    env::set_var(
        "CMAKE_BUILD_PARALLEL_LEVEL",
        std::thread::available_parallelism()
            .unwrap()
            .get()
            .to_string(),
    );

    // Bindings
    let bindings = bindgen::Builder::default()
        .header("wrapper.h")
        .clang_arg(format!("-I{}", sherpa_src.display()))
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate()
        .expect("Failed to generate bindings");

    // Write the generated bindings to an output file
    let bindings_path = out_dir.join("bindings.rs");
    bindings
        .write_to_file(bindings_path)
        .expect("Failed to write bindings");

    println!("cargo:rerun-if-changed=wrapper.h");
    println!("cargo:rerun-if-changed=./sherpa-onnx");

    // Build with Cmake
    // why not sherpa_src?
    let mut config = Config::new(&sherpa_src);

    config
        .define("SHERPA_ONNX_ENABLE_C_API", "ON")
        .define("SHERPA_ONNX_ENABLE_BINARY", "OFF")
        .define("BUILD_SHARED_LIBS", "OFF")
        .define("SHERPA_ONNX_ENABLE_WEBSOCKET", "OFF");

    // TTS
    config.define(
        "SHERPA_ONNX_ENABLE_TTS",
        if cfg!(feature = "tts") { "ON" } else { "OFF" },
    );

    // Cuda https://k2-fsa.github.io/k2/installation/cuda-cudnn.html
    if cfg!(feature = "cuda") {
        config.define("SHERPA_ONNX_ENABLE_GPU", "ON");
        config.define("BUILD_SHARED_LIBS", "ON");
    }

    // DirectML https://onnxruntime.ai/docs/execution-providers/DirectML-ExecutionProvider.html
    if cfg!(feature = "directml") {
        config.define("SHERPA_ONNX_ENABLE_DIRECTML", "ON");
        config.define("BUILD_SHARED_LIBS", "ON");
    }

    if cfg!(any(windows, target_os = "linux")) {
        config.define("SHERPA_ONNX_ENABLE_PORTAUDIO", "ON");
    }

    // General
    config
        .profile(profile)
        .very_verbose(false)
        .always_configure(false);

    let bindings_dir = config.build();

    // Search paths
    println!("cargo:rustc-link-search={}", out_dir.join("lib").display());
    println!("cargo:rustc-link-search=native={}", bindings_dir.display());

    // Cuda
    if cfg!(feature = "cuda") {
        println!(
            "cargo:rustc-link-search={}",
            out_dir.join(format!("build/lib/{}", profile)).display()
        );
    }

    if cfg!(feature = "cuda") {
        if cfg!(windows) {
            println!(
                "cargo:rustc-link-search=native={}",
                out_dir.join("build/_deps/onnxruntime-src/lib").display()
            );
        }
        if cfg!(target_os = "linux") {
            println!(
                "cargo:rustc-link-search=native={}",
                out_dir.join("build/lib").display()
            );
        }
    }

    // Link libraries

    println!("cargo:rustc-link-lib=static=onnxruntime");
    println!("cargo:rustc-link-lib=static=sherpa-onnx-c-api");

    // Sherpa API
    if !build_shared_libs {
        println!("cargo:rustc-link-lib=static=kaldi-native-fbank-core");
        println!("cargo:rustc-link-lib=static=sherpa-onnx-core");
        println!("cargo:rustc-link-lib=static=kaldi-decoder-core");
        println!("cargo:rustc-link-lib=static=sherpa-onnx-kaldifst-core");
        println!("cargo:rustc-link-lib=static=sherpa-onnx-fstfar");
        println!("cargo:rustc-link-lib=static=ssentencepiece_core");
        println!("cargo:rustc-link-lib=static=sherpa-onnx-fst");
    }

    // TTS
    if cfg!(feature = "tts") {
        if !build_shared_libs {
            println!("cargo:rustc-link-lib=static=espeak-ng");
            println!("cargo:rustc-link-lib=static=piper_phonemize");
            println!("cargo:rustc-link-lib=static=ucd");
        }
    }

    // Cuda
    if cfg!(feature = "cuda") && cfg!(windows) {
        println!("cargo:rustc-link-lib=static=onnxruntime_providers_cuda");
        println!("cargo:rustc-link-lib=static=onnxruntime_providers_shared");
        println!("cargo:rustc-link-lib=static=onnxruntime_providers_tensorrt");
    }

    // macOS
    if cfg!(target_os = "macos") {
        println!("cargo:rustc-link-lib=framework=Foundation");
        println!("cargo:rustc-link-lib=c++");
    }

    // Linux
    if cfg!(target_os = "linux") {
        println!("cargo:rustc-link-lib=dylib=stdc++");
    }

    if target.contains("apple") {
        // On (older) OSX we need to link against the clang runtime,
        // which is hidden in some non-default path.
        //
        // More details at https://github.com/alexcrichton/curl-rust/issues/279.
        if let Some(path) = macos_link_search_path() {
            println!("cargo:rustc-link-lib=clang_rt.osx");
            println!("cargo:rustc-link-search={}", path);
        }
    }

    // copy DLLs to target
    if build_shared_libs {
        let suffix = if cfg!(windows) {
            ".dll"
        } else if cfg!(target_os = "macos") {
            ".dylib"
        } else {
            ".so"
        };
        for entry in glob::glob(&format!(
            "{}/*{}",
            out_dir.join("lib").to_str().unwrap(),
            suffix
        ))
        .unwrap()
        .flatten()
        {
            let target_dir = out_dir
                .parent()
                .unwrap()
                .parent()
                .unwrap()
                .parent()
                .unwrap();
            let dst = target_dir.join(entry.file_name().unwrap());
            std::fs::copy(entry, dst).unwrap();
        }
    }
}

fn macos_link_search_path() -> Option<String> {
    let output = Command::new("clang")
        .arg("--print-search-dirs")
        .output()
        .ok()?;
    if !output.status.success() {
        println!(
            "failed to run 'clang --print-search-dirs', continuing without a link search path"
        );
        return None;
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    for line in stdout.lines() {
        if line.contains("libraries: =") {
            let path = line.split('=').nth(1)?;
            return Some(format!("{}/lib/darwin", path));
        }
    }

    println!("failed to determine link search path, continuing without it");
    None
}
