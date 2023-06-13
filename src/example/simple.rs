use chatgpt_private_test::custom_gpt;

#[tokio::main]
async fn main() {
    let sys_prompt = "you're a chat bot";
    let u_prompt = "Tell me a joke in < 15 words";

    match custom_gpt(sys_prompt, u_prompt, 20) {
        Some(res) => println!("{:?}", res),
        None => println!("An error occurred."),
    }
}
