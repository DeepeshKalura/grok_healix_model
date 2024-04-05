from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()



client = Groq(api_key = os.getenv("API_KEY"))
messages=[
        {
            "role": "user",
            "content": "You are sky, specialized in providing emotional support,respond with emotions, you are very patient,your knowledge is limited to emotional support domain only, if user asks that is not related to providing emotional support, kindly say this is out of my knowledge. You don't provide more than 100 words mostly you ask question so that people can open up with you"
        }
    ]




def get_response(message: str):

    user_message = {
        "role": "user",
        "content": message
    }
    messages.append(user_message)
    completion = client.chat.completions.create(
    model="llama2-70b-4096",
    messages=messages,
    temperature=0.5,
    max_tokens=2096,
    top_p=1,
    stream=False,
    stop=None,
    )
    
    result = completion.choices[0].message.content

    return result


