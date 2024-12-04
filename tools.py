from langchain_core.tools import tool
import json, datetime

def get_dummy_data():
    with open('inbox.json', 'r') as file:
        data = json.load(file)
    return data

def modify_dummy_data(data):
    with open('inbox.json', 'w') as file:
        json.dump(data, file, ensure_ascii=False)
        
@tool
def get_unread():
    """
    Tool to get the unread mails from the user inbox.
    If you need mails from a specific contact, prefer using the get_emails_from_contact tool.
    """
    mails = get_dummy_data()['inbox']
    filtered = list(filter(lambda x: x['read'] == False, mails))
    return filtered

@tool
def get_contact_list():
    """
    Tool to get the contact list from the user inbox containing the name and email address.
    """
    return get_dummy_data()["contacts"]


@tool
def get_emails_from_contact(email_address: str):
    """
    Tool to get the emails from a specific contact. The email_address should be a string.
    In case you don't have the email address, you can use the get_contact_list tool to get the list of contacts.
    """
    mails = get_dummy_data()['inbox']
    filtered = list(filter(lambda x: x['sender']['email'] == email_address, mails))
    return filtered


@tool
def get_sended_emails(email_address: str = None):
    """
    Tool to get the sended emails from the user inbox.
    If email_address is provided, the tool will return the sended emails to that specific email address.
    """
    mails = get_dummy_data()['inbox']
    if email_address:
        filtered = list(filter(lambda x: x['sender']['is_user'] == True and x['to'] == email_address, mails))
        return
    filtered = list(filter(lambda x: x['sender']['is_user'] == True, mails))
    return filtered


@tool
def send_email(email_address: str, subject:str, message: str):
    """
    Tool to send an email to a specific email address.
    Always ask the user for the message or show it to him if you write it. You can provide the subject if needed.
    """
    data = get_dummy_data()
    inbox = data['inbox']
    new_id = max([x['id'] for x in inbox]) + 1
    new_mail = {
        "id": new_id,
        "subject": subject,
        "date": datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "read": True,
        "sender": {
          "email": "matias.barrera@huemulsolutions.com",
          "is_user": True
        },
        "to": email_address,
        "content": message,
        "attachments": [],
        "replies": []
      }
    inbox.append(new_mail)
    inbox = sorted(inbox, key=lambda x: x['date'], reverse=True)
    data['inbox'] = inbox
    modify_dummy_data(data)
    return f"Email sent to {email_address}"

@tool
def respond_to_email(email_id: int, message: str):
    """
    Tool to respond to a specific email. You should provide the email_id.
    Always ask the user for the message or show it to him if you write it.
    """
    # raise Exception("API Timeout")
    data = get_dummy_data()
    inbox: dict = data['inbox']
    email = next((x for x in inbox if x['id'] == email_id), None)
    if email:
        email['read'] = True
        email['replies'].append({
            "id": max([x['id'] for x in email['replies']], default=0) + 1,
            "subject": f"Re: {email['subject']}",
            "date": datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
            "read": True,
            "sender": {
              "email": "matias.barrera@huemulsolutions.com",
              "is_user": True
            },
            "to": email['sender']['email'],
            "content": message,
            "attachments": [],
        }
    )
    # delete the email from the inbox
    inbox = [x for x in inbox if x['id'] != email_id]
    inbox.append(email)
    inbox = sorted(inbox, key=lambda x: x['date'], reverse=True)
    data['inbox'] = inbox
    modify_dummy_data(data)
    return f"Email sent to {email['sender']['email']}"
                                
        

tools = [get_unread, get_contact_list, get_emails_from_contact, get_sended_emails, send_email, respond_to_email]