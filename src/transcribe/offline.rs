use crate::{cstr, get_default_provider};
use std::ffi::{CStr, CString};

#[derive(Debug)]
pub struct OfflineRecognizer {
    recognizer: *mut sherpa_rs_sys::SherpaOnnxOfflineRecognizer,
}

#[derive(Debug)]
pub struct OfflineRecognizerResult {
    pub text: String,
    pub event: String,
    pub lang: String,
    pub emotion: String,
}

impl OfflineRecognizer {
    pub fn user_whisper(
        decoder: String,
        encoder: String,
        tokens: String,
        language: String,
        debug: bool,
        provider: Option<String>,
        num_threads: u32,
        bpe_vocab: Option<String>,
    ) -> Self {
        let decoder_c = cstr!(decoder);
        let encoder_c = cstr!(encoder);
        let langauge_c = cstr!(language);
        let task_c = cstr!("transcribe".to_string());
        let tail_paddings = 0;
        let tokens_c = cstr!(tokens);
        let debug = if debug { 1 } else { 0 };
        let provider = provider.unwrap_or(get_default_provider());
        let provider_c = cstr!(provider);

        let bpe_vocab = bpe_vocab.unwrap_or("".into());
        let bpe_vocab_c = cstr!(bpe_vocab);

        let whisper = sherpa_rs_sys::SherpaOnnxOfflineWhisperModelConfig {
            decoder: decoder_c.into_raw(),
            encoder: encoder_c.into_raw(),
            language: langauge_c.into_raw(),
            task: task_c.into_raw(),
            tail_paddings,
        };

        let model_config = unsafe {
            let mut model_config =
                std::mem::MaybeUninit::<sherpa_rs_sys::SherpaOnnxOfflineModelConfig>::zeroed();
            model_config.assume_init_mut().bpe_vocab = bpe_vocab_c.into_raw();
            model_config.assume_init_mut().debug = debug;
            model_config.assume_init_mut().num_threads = num_threads as _;
            model_config.assume_init_mut().provider = provider_c.into_raw();
            model_config.assume_init_mut().tokens = tokens_c.into_raw();
            model_config.assume_init_mut().whisper = whisper;
            model_config.assume_init()
        };

        let config = unsafe {
            let mut config =
                std::mem::MaybeUninit::<sherpa_rs_sys::SherpaOnnxOfflineRecognizerConfig>::zeroed();
            config.assume_init_mut().decoding_method =
                CString::new("greedy_search").unwrap().into_raw();
            config.assume_init_mut().model_config = model_config;
            // config.assume_init_mut().feat_config = sherpa_rs_sys::SherpaOnnxFeatureConfig {
            //     sample_rate: 16000,
            //     feature_dim: 512,
            // };
            config.assume_init()
        };
        let recognizer = unsafe { sherpa_rs_sys::SherpaOnnxCreateOfflineRecognizer(&config) };

        Self { recognizer }
    }

    pub fn user_sense_voice(
        model: String,
        tokens: String,
        language: String,
        use_itn: i32,
        debug: bool,
        provider: Option<String>,
        num_threads: u32,
        bpe_vocab: Option<String>,
    ) -> Self {
        let tokens_c = cstr!(tokens);
        let debug = if debug { 1 } else { 0 };
        let provider = provider.unwrap_or(get_default_provider());
        let provider_c = cstr!(provider);
        let bpe_vocab_c = cstr!(bpe_vocab.unwrap_or("".into()));

        let sense_voice = sherpa_rs_sys::SherpaOnnxOfflineSenseVoiceModelConfig {
            model: cstr!(model).into_raw(),
            language: cstr!(language).into_raw(),
            use_itn,
        };

        let model_config = unsafe {
            let mut model_config =
                std::mem::MaybeUninit::<sherpa_rs_sys::SherpaOnnxOfflineModelConfig>::zeroed();
            model_config.assume_init_mut().bpe_vocab = bpe_vocab_c.into_raw();
            model_config.assume_init_mut().debug = debug;
            model_config.assume_init_mut().num_threads = num_threads as _;
            model_config.assume_init_mut().provider = provider_c.into_raw();
            model_config.assume_init_mut().tokens = tokens_c.into_raw();
            model_config.assume_init_mut().sense_voice = sense_voice;
            model_config.assume_init()
        };
        let decoding_method_c = CString::new("greedy_search").unwrap();
        let config = unsafe {
            let mut config =
                std::mem::MaybeUninit::<sherpa_rs_sys::SherpaOnnxOfflineRecognizerConfig>::zeroed();
            config.assume_init_mut().decoding_method = decoding_method_c.into_raw();
            config.assume_init_mut().model_config = model_config;
            config.assume_init()
        };
        let recognizer = unsafe { sherpa_rs_sys::SherpaOnnxCreateOfflineRecognizer(&config) };

        Self { recognizer }
    }

    pub fn transcribe(&mut self, sample_rate: i32, samples: Vec<f32>) -> OfflineRecognizerResult {
        unsafe {
            let stream = sherpa_rs_sys::SherpaOnnxCreateOfflineStream(self.recognizer);
            sherpa_rs_sys::SherpaOnnxAcceptWaveformOffline(
                stream,
                sample_rate,
                samples.as_ptr(),
                samples.len().try_into().unwrap(),
            );
            sherpa_rs_sys::SherpaOnnxDecodeOfflineStream(self.recognizer, stream);
            let result_ptr = sherpa_rs_sys::SherpaOnnxGetOfflineStreamResult(stream);
            let raw_result = result_ptr.read();
            let text = CStr::from_ptr(raw_result.text);
            let text = text.to_str().unwrap().to_string();
            let emotion = CStr::from_ptr(raw_result.emotion);
            let emotion = emotion.to_str().unwrap().to_string();
            let lang = CStr::from_ptr(raw_result.lang);
            let lang = lang.to_str().unwrap().to_string();
            let event = CStr::from_ptr(raw_result.event);
            let event = event.to_str().unwrap().to_string();
            let result = OfflineRecognizerResult {
                text,
                emotion,
                lang,
                event,
            };
            // Free
            sherpa_rs_sys::SherpaOnnxDestroyOfflineRecognizerResult(result_ptr);
            sherpa_rs_sys::SherpaOnnxDestroyOfflineStream(stream);
            return result;
        }
    }
}

unsafe impl Send for OfflineRecognizer {}
unsafe impl Sync for OfflineRecognizer {}

impl Drop for OfflineRecognizer {
    fn drop(&mut self) {
        unsafe {
            sherpa_rs_sys::SherpaOnnxDestroyOfflineRecognizer(self.recognizer);
        }
    }
}
