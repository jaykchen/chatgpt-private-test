use async_openai::{
    config::OpenAIConfig,
    types::{
        ChatCompletionRequestMessage, CreateChatCompletionRequestArgs, CreateEmbeddingRequestArgs,
        CreateEmbeddingResponse, Role,
    },
    Client,
};
use qdrant_client::prelude::{Payload, QdrantClient, QdrantClientConfig};
use qdrant_client::qdrant::{
    point_id::PointIdOptions, vectors::VectorsOptions, vectors_config::Config, CreateCollection,
    Distance, PointId, PointStruct, SearchPoints, Vector, VectorParams, Vectors, VectorsConfig,
};
use rand::Rng;
use serde::Serialize;
use std::default::Default;
use tiktoken_rs::cl100k_base;

pub async fn search_with_openai(
    query: &str,
    collection_name: &str,
    api_key: &str,
) -> anyhow::Result<()> {
    let config = OpenAIConfig::new().with_api_key(api_key);
    let openai_client = Client::with_config(config);

    let config = QdrantClientConfig::from_url("http://127.0.0.1:6334");
    let qdrant_client = QdrantClient::new(Some(config))?;

    let request = CreateEmbeddingRequestArgs::default()
        .model("text-embedding-ada-002")
        .input(query)
        .build()
        .unwrap();

    let response: CreateEmbeddingResponse = openai_client.embeddings().create(request).await?;
    let question_vector = response.data[0].clone().embedding;

    let search_result = qdrant_client
        .search_points(&SearchPoints {
            collection_name: collection_name.into(),
            vector: question_vector,
            filter: None,
            limit: 10,
            with_vectors: None,
            with_payload: None,
            params: None,
            score_threshold: None,
            offset: None,
            ..Default::default()
        })
        .await?;
    dbg!(search_result);

    Ok(())
}

pub async fn load_text() -> anyhow::Result<()> {
    let bpe = cl100k_base().unwrap();

    // let s = include_str!("book.txt");
    let s = "book.txt";

    let chunked_text = bpe
        .encode_ordinary(&convert(s))
        .chunks(4500)
        .map(|c| bpe.decode(c.to_vec()).unwrap())
        .collect::<Vec<String>>();

    let mut ids_vec = (0..10000u64).into_iter().rev().collect::<Vec<u64>>();

    for chunk in chunked_text {
        // if let Ok(segment) = segment_text(&chunk, api_key).await {
        //     for seg in &segment {
        //         println!("{}\n", seg);
        //     }
        // }
    }

    Ok(())
}

pub async fn init_collection(collection_name: &str) -> anyhow::Result<()> {
    let config = QdrantClientConfig::from_url("http://127.0.0.1:6334");
    let qdrant_client = QdrantClient::new(Some(config))?;

    qdrant_client
        .create_collection(&CreateCollection {
            collection_name: collection_name.into(),
            vectors_config: Some(VectorsConfig {
                config: Some(Config::Params(VectorParams {
                    size: 1536,
                    distance: Distance::Cosine.into(),
                    hnsw_config: None,
                    quantization_config: None,
                    on_disk: None,
                })),
            }),
            ..Default::default()
        })
        .await?;

    Ok(())
}

pub async fn upload_embeddings(
    inp: Vec<String>,
    ids_vec: &mut Vec<u64>,
    collection_name: &str,
    api_key: &str,
) -> anyhow::Result<()> {
    let config = OpenAIConfig::new().with_api_key(api_key);
    let openai_client = Client::with_config(config);
    let config = QdrantClientConfig::from_url("http://127.0.0.1:6334");
    let qdrant_client = QdrantClient::new(Some(config))?;

    let request = CreateEmbeddingRequestArgs::default()
        .model("text-embedding-ada-002")
        .input(&inp)
        .build()?;

    let response: CreateEmbeddingResponse = openai_client.embeddings().create(request).await?;
    let embeddings = response.data;
    let mut points = Vec::new();

    for (i, sentence) in inp.iter().enumerate() {
        let id = ids_vec.pop().unwrap();
        let payload: Payload = serde_json::json!({ "text": sentence.trim().to_string()})
            .try_into()
            .unwrap();
        let point = PointStruct::new(
            PointId {
                point_id_options: Some(PointIdOptions::Num(id)),
            },
            Vectors {
                vectors_options: Some(VectorsOptions::Vector(Vector {
                    data: embeddings[i].clone().embedding,
                })),
            },
            payload,
        );

        points.push(point);
    }

    qdrant_client
        .upsert_points_blocking(collection_name, points, None)
        .await?;
    Ok(())
}

pub async fn segment_text(inp: &str, api_key: &str) -> anyhow::Result<Vec<String>> {
    let config = OpenAIConfig::new().with_api_key(api_key);
    let openai_client = Client::with_config(config);

    let prompt = format!(
        r#"You are examining Chapter 1 of a book. Your mission is to dissect the provided information into short, logically divided segments to facilitate further processing afterwards. 
    Please adhere to the following steps:
    1. Break down dense paragraphs into individual sentences, with each one functioning as a distinct chunk of information. 
    2. Consider code snippets as standalone entities and separate them from the accompanying text, break down long code snippets to chunks of less than 15 lines, please respect the programming language constructs that keep a group of codes together or separate one group of codes from another. 
    3. Take into account the original source's hierarchical markings and formatting specific to a book chapter. These elements can guide the logical segmentation process.
    Keep in mind, the goal is not to summarize, but to restructure the information into more digestible, manageable units. Now, here is the text from the chapter:{inp}".
    Please reply in this format:
```
<sentence>~>_^~<sentence>~>_^~<sentence>
```"#
    );
    let system_message = ChatCompletionRequestMessage {
        role: Role::System,
        content: Some("As a dedicated assistant, your duty is to dissect the provided chapter text into clearer, bite-sized segments. To accomplish this, isolate each sentence and code snippet as independent entities. Remember, your task is not to provide a summary, but to split the original text into a texts sequence more granunlar, respecting the text's hierarchical markings and formatting as they contribute to the understanding of the text. Balance your interpretations with the original structure for an accurate representation. reply in this format: ```<sentence>~>_^~<sentence>~>_^~<sentence>```".to_string()),
        name: None,
        function_call: None,
};

    let user_message = ChatCompletionRequestMessage {
        role: Role::User,
        content: Some(prompt),
        name: None,
        function_call: None,
    };

    let request = CreateChatCompletionRequestArgs::default()
        .model("gpt-3.5-turbo-16k")
        .messages(vec![system_message, user_message])
        .max_tokens(7000_u16)
        .build()?;

    let response = openai_client
        .chat() // Get the API "group" (completions, images, etc.) from the client
        .create(request) // Make the API call in that "group"
        .await?;

    match &response.choices[0].message.content {
        Some(raw_text) => Ok(raw_text
            .split("~>_^~")
            .map(|x| x.to_string())
            .collect::<Vec<_>>()),
        None => Err(anyhow::anyhow!("Could not get the text from OpenAI")),
    }
}

struct EscapeNonAscii;

impl serde_json::ser::Formatter for EscapeNonAscii {
    fn write_string_fragment<W: ?Sized + std::io::Write>(
        &mut self,
        writer: &mut W,
        fragment: &str,
    ) -> std::io::Result<()> {
        for ch in fragment.chars() {
            if ch.is_ascii() {
                writer.write_all(ch.encode_utf8(&mut [0; 4]).as_bytes())?;
            } else {
                write!(writer, "\\u{:04x}", ch as u32)?;
                // write!(writer, "?")?;
            }
        }
        Ok(())
    }
}

pub fn convert(input: &str) -> String {
    let mut writer = Vec::new();
    let formatter = EscapeNonAscii;
    let mut ser = serde_json::Serializer::with_formatter(&mut writer, formatter);
    input.serialize(&mut ser).unwrap();
    String::from_utf8(writer).unwrap()
}

pub fn gen_ids() -> Vec<u64> {
    let mut rng = rand::thread_rng();
    let mut set = std::collections::HashSet::new();

    while set.len() < 9900 {
        set.insert(rng.gen::<u64>());
    }
    set.into_iter().collect::<Vec<u64>>()
}
