use std::{collections::HashMap, fs::File, path::PathBuf, string::ToString};

use minijinja::{Environment, context};
use minijinja_contrib::pycompat::unknown_method_callback;
use serde_json::Value;
use shoji::types::session::classification::{ClassificationMessage, ClassificationRole};
use tokenizers::Tokenizer;

use crate::{
    chat::{TokenizerLocation, hanashi::Error, strftime_now},
    util::tokenizer::load_tokenizer,
};

const CONFIG_FILE_NAME: &str = "config.json";

#[derive(Debug, Clone, PartialEq, ::serde::Serialize, ::serde::Deserialize)]
struct TokenCodecConfig {
    pub prompt_template: String,
    pub output_parser_regex: Option<String>,
    pub system_role_name: String,
    pub user_role_name: String,
    pub assistant_role_name: String,
    pub eos_token: Option<String>,
    pub bos_token: Option<String>,
    pub end_of_thinking_tag: Option<String>,
    pub default_system_prompt: Option<String>,
}

pub struct ClassificationEncoding {
    tokenizer: Tokenizer,
    config: TokenCodecConfig,
}

impl ClassificationEncoding {
    pub fn new(model_dir: &str) -> Result<Self, Error> {
        let model_path = PathBuf::from(model_dir);
        let config_file = File::open(model_path.join(CONFIG_FILE_NAME))
            .map_err(|_| Error::ConfigNotFound(CONFIG_FILE_NAME.to_string()))?;
        let config = serde_json::from_reader::<File, Value>(config_file)
            .map_err(|_| Error::InvalidConfig(CONFIG_FILE_NAME.to_string()))?;
        let codec_config = &config["token_codec_config"];
        let config = serde_json::from_value::<TokenCodecConfig>(codec_config.clone())
            .map_err(|_| Error::InvalidConfig(CONFIG_FILE_NAME.to_string()))?;

        let tokenizer_location = TokenizerLocation::Directory {
            path: model_dir.to_string(),
            name: None,
        };
        let tokenizer = load_tokenizer(&tokenizer_location).map_err(|_| Error::UnableToLoadTokenizer)?;

        Ok(Self {
            tokenizer,
            config,
        })
    }

    pub fn encode(
        &mut self,
        input: &[ClassificationMessage],
    ) -> Result<Vec<u32>, Error> {
        let mut environment = Environment::new();
        environment.set_unknown_method_callback(unknown_method_callback);
        environment.add_function("strftime_now", strftime_now);
        environment
            .add_template("chat_template", self.config.prompt_template.as_str())
            .map_err(|_| Error::UnableToEncodeText)?;

        let messages: Vec<HashMap<String, String>> = input
            .iter()
            .map(|message| {
                let role = match message.role {
                    ClassificationRole::Assistant => self.config.assistant_role_name.as_str(),
                    ClassificationRole::User => self.config.user_role_name.as_str(),
                };

                let mut map: HashMap<String, String> = HashMap::new();
                map.insert("role".to_string(), role.to_string());
                map.insert("content".to_string(), message.content.clone());
                map
            })
            .collect();

        let rendered_message = environment
            .get_template("chat_template")
            .unwrap()
            .render(context!(
                messages => messages,
                add_generation_prompt => false,
                bos_token => self.config.bos_token.clone(),
                eos_token => self.config.eos_token.clone(),
                enable_thinking => false,
            ))
            .map_err(|_| Error::UnableToEncodeText)?;

        let tokens = self.tokenizer.encode(rendered_message, false).map_err(|_| Error::UnableToEncodeText)?;

        Ok(tokens.get_ids().to_vec())
    }
}
