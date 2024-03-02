# ---------------------------------------------- FLASK IMPORTS ---------------------------------------------- #
from flask import Flask, render_template, request, jsonify, Response
from flask_cors import CORS

# ---------------------------------------------- API IMPORTS ---------------------------------------------- #
import requests
from fake_useragent import UserAgent
from secrets import randbelow
from json import dumps

from typing import Generator, List, Dict

# ---------------------------------------------- LOGGING IMPORTS ---------------------------------------------- #
import logging

# ---------------------------------------------- LOGGING ---------------------------------------------- #
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# ---------------------------------------------- API CLASS ---------------------------------------------- #
class ApiWrapper:

    def __init__(self) -> None:
        
        self.__url = 'https://mixtral.replicate.dev/api'

    # messages argument should be in list type and should include dicts
    def msg_str(self, messages: List[Dict[str, str]], *args) -> str:

        # create an empty string we'll modify 
        output: str = ""

        start: str = "[INST]" # this is what's at the start of the user's prompt
        end: str = "[/INST]" # and this is at the end of the user's prompt

        # iterate through all the messages
        for message in messages:

            if message["role"] == "user":

                output += f"<s> {start} {message['content']} {end}"

            elif message["system"] == "system":

                output += f"<s> {start} [SYSTEM: {message['content']}] {end}"

            # if the role is assistant
            else:

                output += f"{message['content']} </s>"

        # at the end, return the new string
        return output

    # get headers function to return the default headers
    def get_headers(self) -> Dict[str, str]:
        
        return {
            'Host': 'mixtral.replicate.dev',
            'User-Agent': f'{UserAgent().random}',
            'Accept': '*/*',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br',
            'Referer': 'https://mixtral.replicate.dev/',
            'Content-Type': 'text/plain;charset=UTF-8',
            'Content-Length': f'{randbelow(100)}',
            'Origin': 'https://mixtral.replicate.dev',
            'Connection': 'keep-alive',
            'Sec-Fetch-Dest': 'empty',
            'Sec-Fetch-Mode': 'cors',
            'Sec-Fetch-Site': 'same-origin',
            'TE': 'trailers'
        }
    
    # chat generator function
    def chat(self, messages: List[Dict[str, str]], model: str, *args) -> Generator[str, None, None]:

        self.__data = {
            "maxTokens": args[0] if args else 800,
            "model": f"{model}",
            "prompt": f"{self.msg_str(messages)}", 
            "temperature": args[1] if args else 0.5,
            "topP": args[2] if args else 0.9,
        }

        response = requests.post(self.__url, headers=self.get_headers(), json=self.__data)
        response.raise_for_status()
        
        for chunk in response.iter_content(chunk_size=1024):
            yield chunk.decode()

api = ApiWrapper()

# ---------------------------------------------- FORMATS ---------------------------------------------- #
# this is returned when the stream is inactive
def openai_format_nostream(response: str, model: str) -> Dict[str, str]:
    
    return {
        "choices": [
            {
            "finish_reason": "stop",
            "index": 0,
            "message": {
                "content": f"{response}",
                "role": "assistant"
                }
            }
        ],                                 
        "model": f"{model}",
        "object": "chat.completion",
    }

# this is returned when the stream is active
def openai_format_streamed(response: str, model: str) -> Dict[str, str]:

    return dumps({
            "object": "chat.completion.chunk",
            "model": f"{model}",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "content": f"{response}",
                    },
                    "finish_reason": None,
                }
            ]
        })

# this is returned when the stream is finished
def openai_format_streamed_last(model: str) -> Dict[str, str]:

    return dumps({
        "object": "chat.completion.chunk",
        "model": f"{model}",
        "choices": [
            {
                "index": 0,
                "delta": {},
                "finish_reason": "stop",
            }
        ]
    })

# ---------------------------------------------- FLASK APP ---------------------------------------------- #
app = Flask(__name__)
CORS(app)

# ---------------------------------------------- ROUTES ---------------------------------------------- #
@app.route('/chat/completions', methods=['POST'])
def chat() -> str:

    # get request json
    data = request.get_json()

    # generate config
    __temperature: float = data.get('temperature', 0.5)
    __topP: float = data.get('topP', 0.9)
    __maxTokens: int = data.get('maxTokens', 800)
    __model: str = data.get('model').removesuffix(' gpt')
    __prompt: List[Dict[str, str]] = data.get('messages')
    __stream: bool = data.get('stream', False)

    # this function is responsible for streaming the response
    def generate():

        for chunk in api.chat(__prompt, __model, __maxTokens, __temperature, __topP):

            yield b'data: ' + str(openai_format_streamed(chunk, __model)).encode() + b'\n\n'

        yield b'data: ' + str(openai_format_streamed_last(__model)).encode() + b'\n\n'

        # and then signal end
        yield b'data: [DONE]'

    # return the response
    return Response(generate(), mimetype='text/event-stream') if __stream else jsonify(openai_format_nostream(''.join([chunk for chunk in generate()]), __model)), 200

# route to get all available models
@app.route('/models', methods=['GET'])
def models() -> str:

    return jsonify({
        "data": [
            {"id": "mistralai/mixtral-8x7b-instruct-v0.1. gpt"},
            {"id": "meta/llama-2-70b-chat gpt"},
        ]
    }), 200

# index route
@app.route('/', methods=['GET'])
def index():
    return "<h1>API Wrapper</h1>"

# ---------------------------------------------- RUN APP ---------------------------------------------- #
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=False)
