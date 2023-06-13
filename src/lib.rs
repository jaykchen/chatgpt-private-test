use dotenv::dotenv;
use http_req::{request::Method, request::Request, uri::Uri};
use serde::Deserialize;
use serde_json::Value;
use std::env;

#[no_mangle]
#[tokio::main(flavor = "current_thread")]
pub async fn custom_gpt(sys_prompt: &str, u_prompt: &str, m_token: u16) -> Option<String> {
    let system_prompt = serde_json::json!(
        {"role": "system", "content": sys_prompt}
    );
    let user_prompt = serde_json::json!(
        {"role": "user", "content": u_prompt}
    );

    match chat(vec![system_prompt, user_prompt], m_token).await {
        Ok((res, _count)) => Some(res),
        Err(_) => None,
    }
}

pub async fn chat(message_obj: Vec<Value>, m_token: u16) -> Result<(String, u32), anyhow::Error> {
    dotenv().ok();
    let api_token = env::var("OPENAI_API_TOKEN")?;

    let params = serde_json::json!({
      "model": "gpt-3.5-turbo",
      "messages": message_obj,
      "temperature": 0.7,
      "top_p": 1,
      "n": 1,
      "stream": false,
      "max_tokens": m_token,
      "presence_penalty": 0,
      "frequency_penalty": 0,
      "stop": "\n"
    });

    let uri = "https://api.openai.com/v1/chat/completions";

    let uri = Uri::try_from(uri)?;
    let mut writer = Vec::new();
    let body = serde_json::to_vec(&params)?;

    let bearer_token = format!("Bearer {}", api_token);
    let _response = Request::new(&uri)
        .method(Method::POST)
        .header("Authorization", &bearer_token)
        .header("Content-Type", "application/json")
        .header("Content-Length", &body.len())
        .body(&body)
        .send(&mut writer)?;

    let res = serde_json::from_slice::<ChatResponse>(&writer)?;
    let token_count = res.usage.total_tokens;
    Ok((res.choices[0].message.content.to_string(), token_count))
}

#[derive(Deserialize)]
pub struct ChatResponse {
    pub id: String,
    pub choices: Vec<Choice>,
    pub usage: Usage,
}

#[derive(Deserialize)]
pub struct Choice {
    pub index: u32,
    pub message: Message,
    pub finish_reason: String,
}

#[derive(Deserialize)]
pub struct Message {
    pub role: String,
    pub content: String,
}

#[derive(Deserialize)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}
