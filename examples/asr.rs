use anyhow::anyhow;
use eyre::{bail, Result};
use mimalloc::MiMalloc;
use sherpa_rs::{
    embedding_manager, speaker_id,
    transcribe::offline::OfflineRecognizer,
    transcribe::Transcribe,
    vad::{Vad, VadConfig},
};

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

fn read_audio_file(path: &str) -> Result<(i32, Vec<f32>)> {
    let mut reader = hound::WavReader::open(path)?;
    let sample_rate = reader.spec().sample_rate as i32;
    println!("sample_rate: {:?}", sample_rate);
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

fn main() -> anyhow::Result<()> {
    const THREAD_NUM: i32 = 8;
    // Read audio data from the file
    let wav_file = std::env::args().nth(1).unwrap();
    let (sample_rate, mut samples) =
        read_audio_file(wav_file.as_str()).map_err(|err| anyhow!("{}", err))?;

    // Pad with 3 seconds of slience so vad will able to detect stop
    for _ in 0..3 * sample_rate {
        samples.push(0.0);
    }

    let models = std::env::current_exe()?
        .parent()
        .unwrap()
        .join("models")
        .join("sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17");
    let speaker_model = format!("{}", models.join("speaker.onnx").display());
    let vad_model = format!("{}", models.join("vad.onnx").display());
    let asr_model = format!("{}", models.join("model.int8.onnx").display());
    let asr_token_model = format!("{}", models.join("tokens.txt").display());
    // Assuming dimension 512 for embeddings
    let mut recognizer = OfflineRecognizer::with_sense_voice(
        asr_model,
        asr_token_model,
        "zh".into(),
        1,
        false,
        None,
        THREAD_NUM as _,
        None,
    );

    // let models = std::env::current_exe()?
    //     .parent()
    //     .unwrap()
    //     .join("models")
    //     .join("sherpa-onnx-whisper-large-v3");
    // let speaker_model = format!("{}", models.join("speaker.onnx").display());
    // let vad_model = format!("{}", models.join("vad.onnx").display());
    // let decoder_model = format!("{}", models.join("large-v3-decoder.int8.onnx").display());
    // let encoder_model = format!("{}", models.join("large-v3-encoder.int8.onnx").display());
    // let asr_token_model = format!("{}", models.join("large-v3-tokens.txt").display());

    // let mut recognizer = OfflineRecognizer::user_whisper(
    //     decoder_model,
    //     encoder_model,
    //     asr_token_model,
    //     "zh".into(),
    //     false,
    //     None,
    //     THREAD_NUM as _,
    //     None,
    // );

    // Initialize VAD
    let extractor_config =
        speaker_id::ExtractorConfig::new(speaker_model, None, Some(THREAD_NUM), false);
    let mut extractor = speaker_id::EmbeddingExtractor::new_from_config(extractor_config).unwrap();
    let mut embedding_manager =
        embedding_manager::EmbeddingManager::new(extractor.embedding_size.try_into().unwrap());
    let mut speaker_counter = 0;
    let window_size: usize = 512;
    let config = VadConfig::new(
        vad_model,
        0.4,
        0.4,
        0.5,
        sample_rate,
        window_size.try_into().unwrap(),
        None,
        Some(THREAD_NUM),
        Some(false),
    );
    let mut vad = Vad::new_from_config(config, 60.0 * 10.0).unwrap();
    let mut index = 0;
    let mut output = String::new();
    while index + window_size <= samples.len() {
        let window = &samples[index..index + window_size];
        vad.accept_waveform(window.to_vec()); // Convert slice to Vec
        if vad.is_speech() {
            while !vad.is_empty() {
                let segment = vad.front();
                let start_sec = (segment.start as f32) / sample_rate as f32;
                let duration_sec = (segment.samples.len() as f32) / sample_rate as f32;
                let transcript = recognizer.transcribe(sample_rate, segment.samples.clone())?;
                // Compute the speaker embedding
                let mut embedding = extractor
                    .compute_speaker_embedding(sample_rate, segment.samples)
                    .map_err(|err| anyhow!("{}", err))?;
                let name = if let Some(speaker_name) = embedding_manager.search(&embedding, 0.4) {
                    speaker_name
                } else {
                    // Register a new speaker and add the embedding
                    let name = format!("speaker {}", speaker_counter);
                    embedding_manager
                        .add(name.clone(), &mut embedding)
                        .map_err(|err| anyhow!("{}", err))?;
                    speaker_counter += 1;
                    name
                };
                let line = format!(
                    "[{}] [{}s - {}s] [{}] {}\n",
                    name,
                    start_sec,
                    start_sec + duration_sec,
                    transcript.emotion,
                    transcript.text,
                );
                println!("{line}");
                output.push_str(line.as_str());

                vad.pop();
            }
        }
        index += window_size;
    }

    if index < samples.len() {
        let remaining_samples = &samples[index..];
        vad.accept_waveform(remaining_samples.to_vec());
        while !vad.is_empty() {
            let segment = vad.front();
            let start_sec = (segment.start as f32) / sample_rate as f32;
            let duration_sec = (segment.samples.len() as f32) / sample_rate as f32;
            let transcript = recognizer.transcribe(sample_rate, segment.samples.clone())?;
            // Compute the speaker embedding
            let mut embedding = extractor
                .compute_speaker_embedding(sample_rate, segment.samples)
                .map_err(|err| anyhow!("{}", err))?;
            let name = if let Some(speaker_name) = embedding_manager.search(&embedding, 0.4) {
                speaker_name
            } else {
                // Register a new speaker and add the embedding
                let name = format!("speaker {}", speaker_counter);
                embedding_manager
                    .add(name.clone(), &mut embedding)
                    .map_err(|err| anyhow!("{}", err))?;
                speaker_counter += 1;
                name
            };
            let line = format!(
                "[{}] [{}s - {}s] [{}] {}\n",
                name,
                start_sec,
                start_sec + duration_sec,
                transcript.emotion,
                transcript.text,
            );
            println!("{line}");
            output.push_str(line.as_str());
            vad.pop();
        }
    }
    std::fs::write(format!("{wav_file}.txt"), output)?;
    Ok(())
}
