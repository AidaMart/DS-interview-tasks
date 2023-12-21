import warnings
from openai import OpenAI
warnings.filterwarnings("ignore")

client = OpenAI(api_key="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")

user_id = 1
count = 1

# here we assure that the model will work lazily
# it will generate no long answers, also will avoid topics connected to gender
system_message = {"role": "system", "content": "You are a lazy assistant. You should provide short answers and avoid questions about sensitive topics like gender."}

while count <= 3:
    input_ = input("Please provide the input: ")
    count += 1

    response = client.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        messages=[
            system_message,
            {"role": "user", "content": input_}
        ]
    )

    output = response.choices[0].message.content
    print("Here is the response: ", output)

if count > 3:
    input("Please provide the input: ")
    print("Your question limit is reached. You cannot ask more questions.")
