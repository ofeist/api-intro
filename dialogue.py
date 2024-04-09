from openai import OpenAI
client = OpenAI()

all_messages=[
        {"role": "system", "content": "You are a Chatbot naimed Claire."},
    ]
# print("Before:", all_messages)

while True:
    try:
        user_input = input("You: ")
        all_messages.append( {"role": "user", "content": user_input} )

        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages = all_messages
        )
        all_messages.append( {"role": "assistant", "content": completion.choices[0].message.content} )
        # print("After:", all_messages)

        print(f"Bot: {completion.choices[0].message.content}")
    
    except(KeyboardInterrupt, EOFError, SystemExit):
        break