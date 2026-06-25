use std::{path::PathBuf, time::Instant};

use backend_uzu::session::{
    ChatSession,
    config::{DecodingConfig, GrammarConfig, RunConfig},
    types::{Input, Output},
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct City {
    name: String,
    description: String,
}

pub fn main() {
    let model_path = PathBuf::from(std::env::args().nth(1).unwrap());
    let prompt = "Return a JSON object for a city of London (name, description). Description must be a comperhensive overview, ~250 words";

    let grammar_config = GrammarConfig::from_json_schema_type::<City>().unwrap();
    let mut session = ChatSession::new(model_path, DecodingConfig::default()).unwrap();
    let start = Instant::now();
    let output = session
        .run(
            Input::Text(prompt.to_string()),
            RunConfig::default().tokens_limit(2048).enable_thinking(false).grammar_config(grammar_config),
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
