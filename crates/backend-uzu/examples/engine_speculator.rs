use std::{path::PathBuf, sync::Arc, time::Instant};

use backend_uzu::{
    session::{
        ChatSession,
        config::{DecodingConfig, RunConfig, SpeculatorConfig},
        types::{Input, Output},
    },
    speculators::prompt_lookup_speculator::PromptLookupSpeculator,
};

pub fn main() {
    let model_path = PathBuf::from(std::env::args().nth(1).unwrap());
    let prompt = "Tell me about London";

    let speculator = PromptLookupSpeculator::new();
    let decoding_config =
        DecodingConfig::default().with_speculator_config(SpeculatorConfig::new(16, Arc::new(speculator)));
    let mut session = ChatSession::new(model_path, decoding_config).unwrap();
    let start = Instant::now();
    let output = session
        .run(
            Input::Text(prompt.to_string()),
            RunConfig::default().tokens_limit(2048).enable_thinking(false),
            None::<fn(Output) -> bool>,
        )
        .unwrap();
    let elapsed = start.elapsed();
    let decode_only_tps = output.stats.generate_stats.as_ref().map(|stats| stats.tokens_per_second).unwrap_or(0.0);
    let end_to_end_tps = output.stats.total_stats.tokens_count_output as f64 / elapsed.as_secs_f64();

    println!("{}", output.text.original);
    println!("{} decode-only tps", decode_only_tps);
    println!("{} end-to-end tps", end_to_end_tps);
}
