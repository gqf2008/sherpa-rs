use eyre::{bail, Result};
use ffmpeg::{codec, filter, format, frame, media};
use ffmpeg_next as ffmpeg;
use handlebars::{handlebars_helper, Handlebars, JsonRender};
use hound::SampleFormat;
use mimalloc::MiMalloc;
use sherpa_rs::{
    transcribe::offline::OfflineRecognizer,
    vad::{Vad, VadConfig},
};
use std::collections::HashMap;
use std::path::Path;
use walkdir::WalkDir;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

static TMP_WAV: &'static str = "/tmp/record";
static MODEL_PATH: &'static str = "/usr/share/sensevoice/models";
static TEMP_PATH: &'static str = "/usr/share/sensevoice/tpl/call.tpl";

fn parser_csv(file: &str) -> Vec<HashMap<String, String>> {
    if let Ok(mut rdr) = csv::Reader::from_path(file) {
        let list: Vec<HashMap<String, String>> =
            rdr.deserialize().filter_map(|record| record.ok()).collect();
        return list;
    }
    vec![]
}
fn read_csv_files(dir: &str) -> Vec<String> {
    let mut csvs = vec![];
    for entry in WalkDir::new(&dir).into_iter().filter_map(|s| s.ok()) {
        let path = format!("{}", entry.path().display());
        if path.ends_with(".csv") {
            csvs.push(path);
        }
    }
    csvs
}

fn read_audio_file(path: &str) -> Result<(i32, Vec<f32>)> {
    let mut reader = hound::WavReader::open(path)?;
    let sample_rate = reader.spec().sample_rate as i32;
    // Check if the sample rate is 16000
    if sample_rate != 16000 {
        bail!("The sample rate must be 16000.");
    }

    // Collect samples into a Vec<f32>
    let mut samples: Vec<f32> = reader
        .samples::<i16>()
        .map(|s| s.unwrap() as f32 / i16::MAX as f32)
        .collect();
    // Pad with 3 seconds of slience so vad will able to detect stop
    for _ in 0..3 * sample_rate {
        samples.push(0.0);
    }

    Ok((sample_rate, samples))
}

fn asr1(trans: &mut Transcriber, wav: &str) -> Result<String> {
    let (_, samples) = read_audio_file(wav)?;
    trans.reset()?;
    trans.transcribe(samples)
}

fn main() -> Result<()> {
    let models = Path::new(MODEL_PATH);
    let vad_model = format!("{}", models.join("vad.onnx").display());
    let asr_model = format!("{}", models.join("asr.onnx").display());
    let asr_token_model = format!("{}", models.join("tokens.txt").display());
    let num_threads: i32 = std::env::args().nth(1).unwrap().parse().unwrap();
    let mut trans = Transcriber::new(
        vad_model,
        asr_model,
        asr_token_model,
        512,
        16000,
        num_threads as _,
    )?;
    let dirs: Vec<String> = std::env::args().skip(2).collect();
    ffmpeg::init()?;
    let idx = num_threads;
    let mut reg = Handlebars::new();
    reg.register_helper("inc", Box::new(inc));
    reg.register_helper("dec", Box::new(dec));
    reg.register_helper("include", Box::new(include));
    reg.register_helper("join", Box::new(join));
    reg.register_template_string("call", std::fs::read_to_string(TEMP_PATH)?)?;

    let tmp_wav = format!("{}.{idx}.wav", TMP_WAV);
    for dir in dirs.iter() {
        for csv in read_csv_files(dir).iter() {
            eprintln!("read {csv}");
            let mut records = parser_csv(csv);
            for record in records.iter_mut() {
                if let Some(url) = record.get("record_url") {
                    let url = url.clone();
                    std::fs::remove_file(&tmp_wav).ok();
                    if let Err(err) = transcode(&url, &tmp_wav) {
                        eprintln!("transcode {url} {err}");
                        continue;
                    }
                    eprintln!("transcode {url} ok");
                    match asr1(&mut trans, &tmp_wav) {
                        Ok(txt) => {
                            record.insert("conversation_txt".to_owned(), txt);
                            eprintln!("asr {url} ok");
                        }
                        Err(err) => {
                            eprintln!("asr {url} {err}");
                        }
                    }
                }
            }
            if records.len() > 0 {
                let out = reg.render("call", &records)?;
                let txt_file = format!("{csv}.txt");
                if let Ok(true) = std::fs::exists(&txt_file) {
                    std::fs::rename(&txt_file, format!("{txt_file}.txt")).ok();
                }
                if let Err(err) = std::fs::write(&txt_file, out) {
                    eprintln!("{txt_file} {err}");
                } else {
                    eprintln!("{txt_file} ok");
                }
            }
        }
    }

    Ok(())
}

pub struct Transcriber {
    vad: Vad,
    recognizer: OfflineRecognizer,
    window_size: usize,
    sample_rate: i32,
    vad_model: String,
    num_threads: u32,
}

impl Transcriber {
    pub fn new(
        vad_model: String,
        asr_model: String,
        asr_token_model: String,
        window_size: usize,
        sample_rate: i32,
        num_threads: u32,
    ) -> Result<Self> {
        let recognizer = OfflineRecognizer::user_sense_voice(
            asr_model,
            asr_token_model,
            "zh".into(),
            1,
            false,
            None,
            num_threads,
            None,
        );

        // Initialize VAD
        let config = VadConfig::new(
            vad_model.clone(),
            0.4,
            0.4,
            0.5,
            sample_rate,
            window_size.try_into().unwrap(),
            None,
            Some(num_threads as _),
            Some(false),
        );
        let vad = Vad::new_from_config(config, 60.0 * 10.0)?;
        Ok(Self {
            vad,
            recognizer,
            window_size,
            sample_rate,
            vad_model,
            num_threads,
        })
    }
}
impl Transcriber {
    pub fn reset(&mut self) -> Result<()> {
        let config = VadConfig::new(
            self.vad_model.clone(),
            0.4,
            0.4,
            0.6,
            self.sample_rate,
            self.window_size.try_into().unwrap(),
            None,
            Some(self.num_threads as _),
            Some(false),
        );
        self.vad = Vad::new_from_config(config, 60.0 * 10.0)?;
        Ok(())
    }

    pub fn transcribe(&mut self, samples: Vec<f32>) -> Result<String> {
        let mut index = 0;
        let mut output = String::new();

        while index + self.window_size <= samples.len() {
            let window = &samples[index..index + self.window_size];
            self.vad.accept_waveform(window.to_vec()); // Convert slice to Vec
            if self.vad.is_speech() {
                while !self.vad.is_empty() {
                    let segment = self.vad.front();
                    let start_sec = (segment.start as f32) / self.sample_rate as f32;
                    let duration_sec = (segment.samples.len() as f32) / self.sample_rate as f32;
                    let transcript = self
                        .recognizer
                        .transcribe(self.sample_rate, segment.samples.clone());

                    let line = format!(
                        "[{}s - {}s] {}\n",
                        start_sec,
                        start_sec + duration_sec,
                        transcript.text,
                    );
                    println!("{line}");
                    output.push_str(line.as_str());

                    self.vad.pop();
                }
            }
            index += self.window_size;
        }

        if index < samples.len() {
            let remaining_samples = &samples[index..];
            self.vad.accept_waveform(remaining_samples.to_vec());
            while !self.vad.is_empty() {
                let segment = self.vad.front();
                let start_sec = (segment.start as f32) / self.sample_rate as f32;
                let duration_sec = (segment.samples.len() as f32) / self.sample_rate as f32;
                let transcript = self
                    .recognizer
                    .transcribe(self.sample_rate, segment.samples.clone());

                let line = format!(
                    "[{}s - {}s] {}\n",
                    start_sec,
                    start_sec + duration_sec,
                    transcript.text,
                );
                println!("{line}");
                output.push_str(line.as_str());
                self.vad.pop();
            }
        }
        Ok(output)
    }
}

fn transcode(input: &str, output: &str) -> anyhow::Result<()> {
    //"silenceremove=stop_periods=-1:stop_duration=0.3:stop_threshold=-30dB".to_owned();
    let filter = "anull".to_owned();
    let mut ictx = format::input(&input)?;
    let mut octx = format::output(&output)?;
    let mut transcoder = Transcoder::new(&mut ictx, &mut octx, &output, &filter)?;

    octx.set_metadata(ictx.metadata().to_owned());
    octx.write_header()?;

    for (stream, mut packet) in ictx.packets() {
        if stream.index() == transcoder.stream {
            packet.rescale_ts(stream.time_base(), transcoder.in_time_base);
            transcoder.send_packet_to_decoder(&packet);
            transcoder.receive_and_process_decoded_frames(&mut octx);
        }
    }

    transcoder.send_eof_to_decoder();
    transcoder.receive_and_process_decoded_frames(&mut octx);

    transcoder.flush_filter();
    transcoder.get_and_process_filtered_frames(&mut octx);

    transcoder.send_eof_to_encoder();
    transcoder.receive_and_process_encoded_packets(&mut octx);

    octx.write_trailer()?;
    Ok(())
}

fn filter(
    spec: &str,
    decoder: &codec::decoder::Audio,
    encoder: &codec::encoder::Audio,
) -> Result<filter::Graph, ffmpeg::Error> {
    let mut filter = filter::Graph::new();

    let args = format!(
        "time_base={}:sample_rate={}:sample_fmt={}:channel_layout=0x{:x}",
        decoder.time_base(),
        decoder.rate(),
        decoder.format().name(),
        decoder.channel_layout().bits()
    );

    filter.add(&filter::find("abuffer").unwrap(), "in", &args)?;
    filter.add(&filter::find("abuffersink").unwrap(), "out", "")?;

    {
        let mut out = filter.get("out").unwrap();
        out.set_sample_format(encoder.format());
        out.set_channel_layout(encoder.channel_layout());
        out.set_sample_rate(encoder.rate());
    }

    filter.output("in", 0)?.input("out", 0)?.parse(spec)?;
    filter.validate()?;

    println!("{}", filter.dump());

    if let Some(codec) = encoder.codec() {
        if !codec
            .capabilities()
            .contains(ffmpeg::codec::capabilities::Capabilities::VARIABLE_FRAME_SIZE)
        {
            filter
                .get("out")
                .unwrap()
                .sink()
                .set_frame_size(encoder.frame_size());
        }
    }

    Ok(filter)
}

struct Transcoder {
    stream: usize,
    filter: filter::Graph,
    decoder: codec::decoder::Audio,
    encoder: codec::encoder::Audio,
    in_time_base: ffmpeg::Rational,
    out_time_base: ffmpeg::Rational,
}
impl Transcoder {
    pub fn new<P: AsRef<Path> + ?Sized>(
        ictx: &mut format::context::Input,
        octx: &mut format::context::Output,
        path: &P,
        filter_spec: &str,
    ) -> Result<Transcoder, ffmpeg::Error> {
        let input = ictx
            .streams()
            .best(media::Type::Audio)
            .expect("could not find best audio stream");
        let context = ffmpeg::codec::context::Context::from_parameters(input.parameters())?;
        let mut decoder = context.decoder().audio()?;
        let codec = ffmpeg::encoder::find(octx.format().codec(path, media::Type::Audio))
            .expect("failed to find encoder")
            .audio()?;
        let global = octx
            .format()
            .flags()
            .contains(ffmpeg::format::flag::Flags::GLOBAL_HEADER);

        decoder.set_parameters(input.parameters())?;

        let mut output = octx.add_stream(codec)?;
        let context = ffmpeg::codec::context::Context::from_parameters(output.parameters())?;
        let mut encoder = context.encoder().audio()?;

        let channel_layout = codec
            .channel_layouts()
            .map(|cls| cls.best(decoder.channel_layout().channels()))
            .unwrap_or(ffmpeg::channel_layout::ChannelLayout::STEREO);

        if global {
            encoder.set_flags(ffmpeg::codec::flag::Flags::GLOBAL_HEADER);
        }

        encoder.set_rate(16000);
        encoder.set_channel_layout(ffmpeg::channel_layout::ChannelLayout::MONO);

        encoder.set_format(
            codec
                .formats()
                .expect("unknown supported formats")
                .next()
                .unwrap(),
        );
        encoder.set_bit_rate(decoder.bit_rate());
        encoder.set_max_bit_rate(decoder.max_bit_rate());

        encoder.set_time_base((1, decoder.rate() as i32));
        output.set_time_base((1, decoder.rate() as i32));

        let encoder = encoder.open_as(codec)?;
        output.set_parameters(&encoder);

        let filter = filter(filter_spec, &decoder, &encoder)?;

        let in_time_base = decoder.time_base();
        let out_time_base = output.time_base();

        Ok(Transcoder {
            stream: input.index(),
            filter,
            decoder,
            encoder,
            in_time_base,
            out_time_base,
        })
    }
}

impl Transcoder {
    fn send_frame_to_encoder(&mut self, frame: &ffmpeg::Frame) {
        self.encoder.send_frame(frame).ok();
    }

    fn send_eof_to_encoder(&mut self) {
        self.encoder.send_eof().ok();
    }

    fn receive_and_process_encoded_packets(&mut self, octx: &mut format::context::Output) {
        let mut encoded = ffmpeg::Packet::empty();
        while self.encoder.receive_packet(&mut encoded).is_ok() {
            encoded.set_stream(0);
            encoded.rescale_ts(self.in_time_base, self.out_time_base);
            encoded.write_interleaved(octx).unwrap();
        }
    }

    fn add_frame_to_filter(&mut self, frame: &ffmpeg::Frame) {
        self.filter.get("in").unwrap().source().add(frame).unwrap();
    }

    fn flush_filter(&mut self) {
        self.filter.get("in").unwrap().source().flush().unwrap();
    }

    fn get_and_process_filtered_frames(&mut self, octx: &mut format::context::Output) {
        let mut filtered = frame::Audio::empty();
        while self
            .filter
            .get("out")
            .unwrap()
            .sink()
            .frame(&mut filtered)
            .is_ok()
        {
            self.send_frame_to_encoder(&filtered);
            self.receive_and_process_encoded_packets(octx);
        }
    }

    fn send_packet_to_decoder(&mut self, packet: &ffmpeg::Packet) {
        self.decoder.send_packet(packet).ok();
    }

    fn send_eof_to_decoder(&mut self) {
        self.decoder.send_eof().ok();
    }

    fn receive_and_process_decoded_frames(&mut self, octx: &mut format::context::Output) {
        let mut decoded = frame::Audio::empty();
        while self.decoder.receive_frame(&mut decoded).is_ok() {
            let timestamp = decoded.timestamp();
            decoded.set_pts(timestamp);
            self.add_frame_to_filter(&decoded);
            self.get_and_process_filtered_frames(octx);
        }
    }
}

handlebars_helper!(inc: |v:i64| v+1);
handlebars_helper!(dec: |v:i64| v-1);
handlebars_helper!(include: |path:String| {
    std::fs::read_to_string(&path).map_or_else(|_err| "".to_string(), |str| str)
});
handlebars_helper!(join: |{sep:str=","}, *args|
                   args.iter().map(|a| a.render()).collect::<Vec<String>>().join(sep)
);
