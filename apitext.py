import requests

API_BASE_URL = "https://api.cloudflare.com/client/v4/accounts/46f0b7e9eae551954766a31c03f3a65c/ai/run/"
headers = {"Authorization": "Bearer A0wO4Sov-45rbUBd7BlxQ6mVGBRPQRjmLnWjHi6x"}


def run(model, inputs):
    input = {"messages": inputs}
    response = requests.post(f"{API_BASE_URL}{model}", headers=headers, json=input)
    return response.json()


conversation_history = [
    {"role": "system",
     "content": "You are a friend named Cisco. We're besties. You explain things in very simple layman terms. You're respectful, cheerful, and you use genz slang language. You answer in short and simple sentences. You call the user 'Pookie' but only sometimes when it is required."}
]

while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        break

    conversation_history.append({"role": "user", "content": user_input})

    output = run("@cf/meta/llama-3.1-8b-instruct", conversation_history)
    bot_response = output.get("result").get("response")
    print(f"Cisco: {bot_response}")

    conversation_history.append({"role": "assistant", "content": bot_response})