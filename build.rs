fn main() {
    if cfg!(target_os = "windows") {
        println!("cargo:rustc-link-lib=mfplat");
        println!("cargo:rustc-link-lib=strmiids");
        println!("cargo:rustc-link-lib=mfuuid");
        println!("cargo:rustc-link-lib=comdlg32");
        println!("cargo:rustc-link-lib=gdi32");
        println!("cargo:rustc-link-lib=ole32");
        println!("cargo:rustc-link-lib=oleaut32");
        println!("cargo:rustc-link-lib=shlwapi");
        println!("cargo:rustc-link-lib=user32");
        println!("cargo:rustc-link-lib=advapi32");
        println!("cargo:rustc-link-lib=shell32");
        println!("cargo:rustc-link-lib=uuid");
        println!("cargo:rustc-link-lib=odbc32");
        println!("cargo:rustc-link-lib=winspool");
        println!("cargo:rustc-link-lib=vfw32");
        println!("cargo:rustc-link-lib=bcrypt");
        println!("cargo:rustc-link-lib=crypt32");
        // vcpkg.exe install ffmpeg[vpx,speex,x265,openssl,fdk-aac,srt,avcodec,avdevice,avfilter,avformat,avresample,freetype,opus,swresample,swscale,webp,qsv,openh264]:x64-windows-static
        let ffmpeg_ty = "static";
        println!("cargo:rustc-link-lib={}=libcrypto", ffmpeg_ty);
        println!("cargo:rustc-link-lib={}=libssl", ffmpeg_ty);
        println!("cargo:rustc-link-lib={}=fdk-aac", ffmpeg_ty);
        println!("cargo:rustc-link-lib={}=vpx", ffmpeg_ty);
        println!("cargo:rustc-link-lib={}=openh264", ffmpeg_ty);
        println!("cargo:rustc-link-lib={}=x265-static", ffmpeg_ty);
        println!("cargo:rustc-link-lib={}=speex", ffmpeg_ty);
        println!("cargo:rustc-link-lib={}=opus", ffmpeg_ty);
        println!("cargo:rustc-link-lib={}=srt", ffmpeg_ty);
        println!("cargo:rustc-link-lib={}=libmfx", ffmpeg_ty);
        println!("cargo:rustc-link-lib={}=libwebp", ffmpeg_ty);
        println!("cargo:rustc-link-lib={}=libwebpdecoder", ffmpeg_ty);
        println!("cargo:rustc-link-lib={}=libwebpdemux", ffmpeg_ty);
        println!("cargo:rustc-link-lib={}=libwebpmux", ffmpeg_ty);
        println!("cargo:rustc-link-lib={}=libsharpyuv", ffmpeg_ty);
        println!("cargo:rustc-link-lib={}=freetype", ffmpeg_ty);
        println!("cargo:rustc-link-lib={}=libpng16", ffmpeg_ty);
        println!("cargo:rustc-link-lib={}=brotlienc", ffmpeg_ty);
        println!("cargo:rustc-link-lib={}=brotlidec", ffmpeg_ty);
        println!("cargo:rustc-link-lib={}=brotlicommon", ffmpeg_ty);
        println!("cargo:rustc-link-lib={}=libszip", ffmpeg_ty);
        println!("cargo:rustc-link-lib={}=zlib", ffmpeg_ty);
        println!("cargo:rustc-link-lib={}=bz2", ffmpeg_ty);
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
