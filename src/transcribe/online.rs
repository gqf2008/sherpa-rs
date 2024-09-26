use crate::{cstr, get_default_provider};
use serde::Deserialize;
use std::ffi::{CStr, CString};

use super::Transcribe;

#[derive(Debug)]
pub struct OnlineRecognizer {
    recognizer: *mut sherpa_rs_sys::SherpaOnnxOnlineRecognizer,
    stream: *mut sherpa_rs_sys::SherpaOnnxOnlineStream,
    // display: *const sherpa_rs_sys::SherpaOnnxDisplay,
}

#[derive(Deserialize, Debug)]
pub struct OnlineRecognizerResult {
    // pub text: *const ::std::os::raw::c_char,
    // pub tokens: *const ::std::os::raw::c_char,
    // pub tokens_arr: *const *const ::std::os::raw::c_char,
    // pub timestamps: *mut f32,
    // pub count: i32,
    // #[doc = " Return a json string.\n\n The returned string contains:\n   {\n     \"text\": \"The recognition result\",\n     \"tokens\": [x, x, x],\n     \"timestamps\": [x, x, x],\n     \"segment\": x,\n     \"start_time\": x,\n     \"is_final\": true|false\n   }"]
    // pub json: *const ::std::os::raw::c_char,
    /** Return a json string.
     *
     * The returned string contains:
     *   {
     *     "text": "The recognition result",
     *     "tokens": [x, x, x],
     *     "timestamps": [x, x, x],
     *     "segment": x,
     *     "start_time": x,
     *     "is_final": true|false
     *   }
     */
    pub text: String,
    pub tokens: Vec<String>,
    pub timestamps: Vec<f32>,
    pub segment: f32,
    pub start_time: f32,
    pub is_final: bool,
}

impl OnlineRecognizer {
    pub fn with_zipformer(
        decoder: String,
        encoder: String,
        joiner: String,
        tokens: String,
        debug: bool,
        num_threads: u32,
        decoding_method: Option<String>,
        provider: Option<String>,
        hotwords_file: Option<String>,
        hotwords_score: Option<f32>,
    ) -> Self {
        let decoder_c = cstr!(decoder);
        let encoder_c = cstr!(encoder);
        let joiner_c = cstr!(joiner);
        let tokens_c = cstr!(tokens);
        let debug = if debug { 1 } else { 0 };
        let provider_c = cstr!(provider.unwrap_or(get_default_provider()));
        let decoding_method = cstr!(decoding_method.unwrap_or("greedy_search".to_string()));

        let model_config = unsafe {
            let mut model_config =
                std::mem::MaybeUninit::<sherpa_rs_sys::SherpaOnnxOnlineModelConfig>::zeroed();
            model_config.assume_init_mut().debug = debug;
            model_config.assume_init_mut().num_threads = num_threads as _;
            model_config.assume_init_mut().provider = provider_c.into_raw();
            model_config.assume_init_mut().tokens = tokens_c.into_raw();

            model_config.assume_init_mut().transducer =
                sherpa_rs_sys::SherpaOnnxOnlineTransducerModelConfig {
                    decoder: decoder_c.into_raw(),
                    encoder: encoder_c.into_raw(),
                    joiner: joiner_c.into_raw(),
                };
            model_config.assume_init()
        };

        let config = unsafe {
            let mut config =
                std::mem::MaybeUninit::<sherpa_rs_sys::SherpaOnnxOnlineRecognizerConfig>::zeroed();
            config.assume_init_mut().decoding_method = decoding_method.into_raw();
            config.assume_init_mut().model_config = model_config;
            config.assume_init_mut().max_active_paths = 4;
            config.assume_init_mut().enable_endpoint = 1;
            config.assume_init_mut().rule1_min_trailing_silence = 2.4;
            config.assume_init_mut().rule2_min_trailing_silence = 1.2;
            config.assume_init_mut().rule3_min_utterance_length = 300.;
            config.assume_init_mut().feat_config = sherpa_rs_sys::SherpaOnnxFeatureConfig {
                sample_rate: 16000,
                feature_dim: 80,
            };
            hotwords_file.map(|hotwords_file| {
                config.assume_init_mut().hotwords_file = cstr!(hotwords_file).into_raw();
            });
            hotwords_score.map(|hotwords_score| {
                config.assume_init_mut().hotwords_score = hotwords_score;
            });
            config.assume_init()
        };
        let recognizer = unsafe { sherpa_rs_sys::SherpaOnnxCreateOnlineRecognizer(&config) };
        let stream: *mut sherpa_rs_sys::SherpaOnnxOnlineStream =
            unsafe { sherpa_rs_sys::SherpaOnnxCreateOnlineStream(recognizer) };
        // let display: *const sherpa_rs_sys::SherpaOnnxDisplay =
        //     unsafe { sherpa_rs_sys::SherpaOnnxCreateDisplay(50) };
        Self {
            recognizer,
            stream,
            //  display,
        }
    }

    pub fn transcribe(
        &mut self,
        sample_rate: i32,
        samples: Vec<f32>,
    ) -> anyhow::Result<OnlineRecognizerResult> {
        unsafe {
            sherpa_rs_sys::SherpaOnnxOnlineStreamAcceptWaveform(
                self.stream,
                sample_rate,
                samples.as_ptr(),
                samples.len() as _,
            );
            while sherpa_rs_sys::SherpaOnnxIsOnlineStreamReady(self.recognizer, self.stream) > 0 {
                sherpa_rs_sys::SherpaOnnxDecodeOnlineStream(self.recognizer, self.stream);
            }
            let result_ptr =
                sherpa_rs_sys::SherpaOnnxGetOnlineStreamResult(self.recognizer, self.stream);

            let raw_result: sherpa_rs_sys::SherpaOnnxOnlineRecognizerResult = result_ptr.read();
            let json = CStr::from_ptr(raw_result.json);
            let json = json.to_str().unwrap_or_default();
            let result = serde_json::from_str::<OnlineRecognizerResult>(json)?;
            sherpa_rs_sys::SherpaOnnxDestroyOnlineRecognizerResult(result_ptr);
            if sherpa_rs_sys::SherpaOnnxOnlineStreamIsEndpoint(self.recognizer, self.stream) > 0 {
                sherpa_rs_sys::SherpaOnnxOnlineStreamReset(self.recognizer, self.stream);
            }
            return Ok(result);
        }
    }
}

impl Transcribe<OnlineRecognizerResult> for OnlineRecognizer {
    fn transcribe(
        &mut self,
        sample_rate: i32,
        samples: Vec<f32>,
    ) -> anyhow::Result<OnlineRecognizerResult> {
        self.transcribe(sample_rate, samples)
    }
}

unsafe impl Send for OnlineRecognizer {}
unsafe impl Sync for OnlineRecognizer {}

impl Drop for OnlineRecognizer {
    fn drop(&mut self) {
        unsafe {
            //  sherpa_rs_sys::SherpaOnnxDestroyDisplay(self.display);
            sherpa_rs_sys::SherpaOnnxDestroyOnlineStream(self.stream);
            sherpa_rs_sys::SherpaOnnxDestroyOnlineRecognizer(self.recognizer);
        }
    }
}
