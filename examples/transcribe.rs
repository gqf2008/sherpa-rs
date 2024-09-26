/*
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-whisper-tiny.tar.bz2
tar xvf sherpa-onnx-whisper-tiny.tar.bz2
wget https://github.com/thewh1teagle/sherpa-rs/releases/download/v0.1.0/motivation.wav -O motivation.wav
cargo run --example transcribe
*/

use eyre::{bail, Result};
use sherpa_rs::transcribe::whisper::WhisperRecognizer;
use sherpa_rs::transcribe::Transcribe;

fn read_audio_file(path: &str) -> Result<(i32, Vec<f32>)> {
    let mut reader = hound::WavReader::open(path)?;
    let sample_rate = reader.spec().sample_rate as i32;

    // Check if the sample rate is 16000
    if sample_rate != 16000 {
        bail!("The sample rate must be 16000.");
    }

    // Collect samples into a Vec<f32>
    let samples: Vec<f32> = reader
        .samples::<i16>()
        .map(|s| s.unwrap() as f32 / i16::MAX as f32)
        .collect();

    Ok((sample_rate, samples))
}

fn main() -> Result<()> {
    let wav_file = std::env::args().nth(1).unwrap();
    let (sample_rate, mut samples) = read_audio_file(wav_file.as_str())?;

    // Pad with 3 seconds of slience so vad will able to detect stop
    for _ in 0..3 * sample_rate {
        samples.push(0.0);
    }
    // Check if the sample rate is 16000
    if sample_rate != 16000 {
        bail!("The sample rate must be 16000.");
    }
    let models = std::env::current_exe()?
        .parent()
        .unwrap()
        .join("models")
        .join("sherpa-onnx-whisper-large-v3");
    //let speaker_model = format!("{}", models.join("speaker.onnx").display());
    //let vad_model = format!("{}", models.join("vad.onnx").display());
    let decoder_model = format!("{}", models.join("large-v3-decoder.int8.onnx").display());
    let encoder_model = format!("{}", models.join("large-v3-encoder.int8.onnx").display());
    let asr_token_model = format!("{}", models.join("large-v3-tokens.txt").display());

    let mut recognizer = WhisperRecognizer::new(
        decoder_model,
        encoder_model,
        asr_token_model,
        "zh".into(),
        Some(true),
        Some("directml".into()),
        Some(8),
        None,
    );
    let result = recognizer.transcribe(sample_rate, samples);
    println!("{:?}", result);
    Ok(())
}
