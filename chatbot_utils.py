import os
from transformers import GPT2TokenizerFast
import openai
import numpy as np
import pandas as pd
from tqdm import tqdm

from utils import read_file
import json
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

openai.api_key =  os.getenv('OPENAI_KEY')

DOC_MODEL = 'text-embedding-ada-002'
COMPLETIONS_MODEL = "text-davinci-003"


def get_embeddings(text: str, model: str) -> list[float]:
    '''
    Calculate embeddings.

    Parameters
    ----------
    text : str
        Text to calculate the embeddings for.
    model : str
        String of the model used to calculate the embeddings.

    Returns
    -------
    list[float]
        List of the embeddings
    '''
    result = openai.Embedding.create(
      model=model,
      input=text
    )
    return result["data"][0]["embedding"]


def get_max_num_tokens():
    return 2046


def has_text(element):
    return element.string is not None and len(element.string.strip()) > 0


def collect_title_body_embeddings(pages, save_csv=True):
    collect = []
    json_data = json.loads(pages[0])
    for json_body in tqdm(json_data):
        body = f"QUESTION: {json_body['customer']}\n\n\n\nANSWER: {json_body['agent']}"
        tokens = tokenizer.encode(body)
        collect += [(body, len(tokens))]

    DOC_title_content_embeddings = pd.DataFrame(collect, columns=['body', 'num_tokens'])
    # Caculate the embeddings
    # Limit first to pages with less than 2046 tokens
    DOC_title_content_embeddings = DOC_title_content_embeddings[DOC_title_content_embeddings.num_tokens<=get_max_num_tokens()]
    DOC_title_content_embeddings['embeddings'] = DOC_title_content_embeddings.body.apply(lambda x: get_embeddings(x, DOC_MODEL))

    if save_csv:
        DOC_title_content_embeddings.to_csv('conversations.csv', index=False)

    return DOC_title_content_embeddings


def update_internal_doc_embeddings():
    data = read_file("./data/conversations.json")
    DOC_title_content_embeddings= collect_title_body_embeddings(data, save_csv=True)
    return DOC_title_content_embeddings


def vector_similarity(x, y):
    return np.dot(np.array(x), np.array(y))


def order_document_sections_by_query_similarity(query: str, doc_embeddings: pd.DataFrame):
    """
    Find the query embedding for the supplied query, and compare it against all of the pre-calculated document embeddings
    to find the most relevant sections.

    Return the list of document sections, sorted by relevance in descending order.
    """
    query_embedding = get_embeddings(query, model=DOC_MODEL)
    doc_embeddings['similarity'] = doc_embeddings['embeddings'].apply(lambda x: vector_similarity(x, query_embedding))
    doc_embeddings.sort_values(by='similarity', inplace=True, ascending=False)
    doc_embeddings.reset_index(drop=True, inplace=True)

    return doc_embeddings


def construct_prompt(query,conversation):

    MAX_SECTION_LEN = get_max_num_tokens()
    SEPARATOR = "\n* "
    separator_len = len(tokenizer.tokenize(SEPARATOR))

    chosen_convo = []
    chosen_convo_len = 0

    for section_index in range(len(conversation)):
        # Add contexts until we run out of space.
        document_section = conversation.loc[section_index]

        chosen_convo_len += document_section.num_tokens + separator_len
        if chosen_convo_len > MAX_SECTION_LEN:
            break
        if isinstance(document_section.body, str):
            chosen_convo.append(SEPARATOR + document_section.body.replace("\n", " "))

    prompt = f"""
    You are a customer support agent.
    Answer the question as truthfully as possible using the provided context below.
    If the answer is not contained within the text below, say "I don't know."
    \n\nContext: {chosen_convo}\n\n
    \n\Q: {query}\n\n
    \n\A:\n"""

    return prompt


def generate_answer(messages, conversation_embeddings):
    query = messages[-1].content
    conversation_embeddings_ordered = order_document_sections_by_query_similarity(query, conversation_embeddings)

    # Construct the prompt
    prompt = construct_prompt(query,conversation_embeddings_ordered)
    # Ask the question with the context to ChatGPT
    openai_messages = []

    for message in messages:
        openai_messages.append({"role": message.role,"content": message.content})

    openai_messages.append({"role": "system","content": prompt})
    # openai_messages.append({"role": "system","content": "The message is being sent in zendesk html_body api. Your response should be formatted as html. Line breaks should be <p></p> tags."})

    response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=openai_messages,
    # top_p=1,
    # frequency_penalty=0,
    # presence_penalty=0,
    # temperature=0.1,
    # max_tokens=300
)
    output = response['choices'][0]['message']['content']
    return output