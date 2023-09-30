from zenpy import Zenpy
from zenpy.lib.api_objects import Comment

import os
import requests
ZENDESK_API_KEY = os.getenv('ZENDESK_API_KEY')

from datetime import datetime

def calculate_time_ago(date_str):
    # Convert the date string to a datetime object
    date_format = "%Y-%m-%dT%H:%M:%SZ"
    date = datetime.strptime(date_str, date_format)

    # Get the current datetime
    current_datetime = datetime.utcnow()

    # Calculate the time difference
    time_difference = current_datetime - date

    # Calculate days, hours, and minutes ago
    days_ago = time_difference.days
    seconds_ago = time_difference.seconds
    hours_ago = seconds_ago // 3600
    minutes_ago = (seconds_ago // 60) % 60

    result = {
        "days": days_ago,
        "hours": hours_ago,
        "minutes": minutes_ago
    }

    return result


creds = {
    'email' : 'tech@tek-wellness.com',
    'token' : ZENDESK_API_KEY,
    'subdomain': 'tek-wellness'
}

# zenpy_client = Zenpy(**creds)
# conversations = []
# for ticket in zenpy_client.search(type='ticket', created_after='2023-09-01'):
#     comments = zenpy_client.tickets.comments(ticket.id)
#     cached_comment = None
#     for comment in comments:
#         is_agent = comment.author.moderator

#         if cached_comment and is_agent:
#             conversation = {
#                 "customer": cached_comment.body,
#                 "agent": comment.body,
#             }
#             conversations.append(conversation)
#         if not is_agent:
#             cached_comment = comment

# # save to json conversations.json
# with open('conversations.json', 'w') as f:
#     json.dump(conversations, f)


# a function that will return only the newest comment from a ticket:
def get_latest_comment(ticket):
    zenpy_client = Zenpy(**creds)

    comments = [x.body for x in zenpy_client.tickets.comments(ticket)]
    # get last comment
    last_comment = comments[-1]
    return last_comment


def get_new_tickets():
    zenpy_client = Zenpy(**creds)
    ticket_gen = zenpy_client.search(type='ticket', status=["new", "open"],sort_by="updated_at", sort_order='desc')
    tickets = []
    for ticket in ticket_gen:
        subject = ticket.subject
        email = ticket.submitter.email
        name = ticket.submitter.name
        status = ticket.status
        id = ticket.id
        updated_at = ticket.updated_at
        # updated_at_ago = calculate_time_ago(updated_at)
        new_ticket = {
            "id": id,
            "email": email,
            "name": name,
            "status": status,
            "age": updated_at,
            "subject": subject
        }
        tickets.append(new_ticket)
        # break
    return tickets

def get_ticket_comments(ticket):
    zenpy_client = Zenpy(**creds)
    comments = []
    for comment in zenpy_client.tickets.comments(ticket):
        body = comment.body
        author = comment.author.name
        created_at = comment.created_at
        # age = calculate_time_ago(create_at)
        comments.append({
            "body": body,
            "author": author,
            "age":created_at
        })
    # comments = [{"body": x.body, "author": author, "age": age} for x in zenpy_client.tickets.comments(ticket)]
    return comments


def post_comment(ticket, html_comment):

    signature = '<p></p><p>Big 💛 From TXO HQ!</p>'
    html_comment = html_comment.replace(signature, '')

    url = f'https://{creds["subdomain"]}.zendesk.com/api/v2/tickets/{ticket}'
    payload = {
        'ticket': {
            'status': 'solved',
            'comment': {
                'html_body': html_comment,
                'public': True
            }
        }
    }
    auth = (f'{creds["email"]}/token', creds["token"])

    response = requests.put(url, json=payload, auth=auth).json()
    return response




