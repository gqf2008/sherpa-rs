pub mod offline;
pub mod online;
pub mod whisper;

pub trait Transcribe<R> {
    fn transcribe(&mut self, sample_rate: i32, samples: Vec<f32>) -> anyhow::Result<R>;
}
