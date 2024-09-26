fn main() {
    if cfg!(target_os = "windows") {
        let wlibs = [
            "mfplat", "strmiids", "mfuuid", "comdlg32", "gdi32", "ole32", "oleaut32", "shlwapi",
            "user32", "advapi32", "shell32", "uuid", "odbc32", "winspool", "vfw32", "bcrypt",
            "crypt32",
        ];
        wlibs.iter().for_each(|lib| {
            println!("cargo:rustc-link-lib={}", lib);
        });

        // vcpkg.exe install ffmpeg[vpx,speex,x265,openssl,fdk-aac,srt,avcodec,avdevice,avfilter,avformat,avresample,freetype,opus,swresample,swscale,webp,qsv,openh264]:x64-windows-static
        let ffmpeg_static = "static";
        let ffmpeg_libs = [
            "libcrypto",
            "libssl",
            "fdk-aac",
            "vpx",
            "openh264",
            "x265-static",
            "speex",
            "opus",
            "srt",
            "libmfx",
            "libwebp",
            "libwebpdecoder",
            "libwebpdemux",
            "libwebpmux",
            "libsharpyuv",
            "freetype",
            "libpng16",
            "brotlienc",
            "brotlidec",
            "brotlicommon",
            "libszip",
            "zlib",
            "bz2",
        ];

        ffmpeg_libs.iter().for_each(|lib| {
            println!("cargo:rustc-link-lib={}={}", ffmpeg_static, lib);
        });
    }

    if cfg!(target_os = "linux") {
        let libs = ["ssl", "crypto", "dl", "pthread"];
        let lib_path = ["/usr/lib/x86_64-linux-gnu"];
        lib_path.iter().for_each(|lib_path| {
            println!("cargo::rustc-link-search=all={}", lib_path);
        });
        libs.iter().for_each(|lib| {
            println!("cargo:rustc-link-lib=static={}", lib);
        });
    }
}
