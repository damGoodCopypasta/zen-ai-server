import chatbot_utils
from zendesk_utils import get_ticket_comments, get_new_tickets,post_comment
from litestar import Litestar , post, get
from litestar.config.cors import CORSConfig
from utils import convert_to_html_body
import pandas as pd
from pydantic import BaseModel
from typing import List
from dotenv import load_dotenv
load_dotenv()

cors_config = CORSConfig(allow_origins=["*"])


def parse_numbers(s):
    return [float(x) for x in s.strip("[]").split(",")]


def get_confluence_embeddings():
    convo_file = "conversations.csv"

    conversation_embeddings = pd.read_csv(
        convo_file, dtype={"embeddings": object}
    )
    conversation_embeddings["embeddings"] = conversation_embeddings[
        "embeddings"
    ].apply(lambda x: parse_numbers(x))

    return conversation_embeddings

# Class definitions (Pydantic is like Typescript :P)
class Message (BaseModel):
    role: str
    content: str


class UserQuery(BaseModel):
    messages: List[Message]
    isGPT4: bool


class ConfluencePage (BaseModel):
    title: str
    link: str

class Reply(BaseModel):
    message: str
    pages: List[ConfluencePage]


@post(path="/message")
async def chat_message(data: UserQuery) -> Reply:

    return {
        "message": "<p>Hi Freya,</p><p></p><p>Thank you for contacting us. Unfortunately, we do not have a free trial offer at the moment. However, we do have a range of sample workouts available on our My TXO by Tiff Hall YouTube channel which will help you get a better sense of the workouts currently available on the program. Here is a link: https://www.youtube.com/@MYTXO </p><p></p><p>We also have a promotion running at the moment offering a 12-month recurring subscription for $99, down from $229, when you use the code TIFF99 https://mytxo.com/register?variant=life&amp;planid=AU12MONTH-S </p><p></p><p>We hope this helps.</p><p></p><p>Big 💛 From TXO HQ!</p>"
}

    # get query string from url
    conversation_embeddings = get_confluence_embeddings()

    message = chatbot_utils.generate_answer(data.messages, conversation_embeddings)
    # convert pydantic model messages to a simple list
    return { "message": convert_to_html_body(message) }

@get(path="/tickets")
async def get_tickets() -> List[str]:
    return get_new_tickets()

@get(path="/comments")
async def get_comments(ticket: str) -> List[str]:
    return get_ticket_comments(ticket)


class Reply(BaseModel):
    ticket: str
    html_body: str


@post(path="/reply")
async def post_reply(data: Reply) -> None:
    return post_comment(data.ticket, data.html_body)

app = Litestar(
    route_handlers=[chat_message, get_tickets, get_comments, post_reply],
    cors_config=cors_config
    )









