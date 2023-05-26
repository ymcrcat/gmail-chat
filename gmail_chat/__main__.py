import sys
import os
import os.path
import base64
import pickle
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from google.api_core.exceptions import BadRequest
import dateutil.parser as parser
from langchain.embeddings.openai import OpenAIEmbeddings
import pinecone
from tqdm import tqdm
import tiktoken
from langchain.text_splitter import CharacterTextSplitter


MODEL_NAME = "text-embedding-ada-002" # Name of the model used to generate text embeddings
MAX_TOKENS = 8191 # Maximum number of tokens allowed by the model
INDEX_NAME = "email-index"
TEXT_EMBEDDINGS_DIM = 1536 # Dimension of text embeddings
METRIC="cosine"

def chunk_text(text):
    # Initialize the tokenizer
    tokenizer = tiktoken.encoding_for_model('gpt-3.5-turbo')
    
    # Initialize the text splitter
    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(tokenizer.name)

    # Split the tokens into chunks of 8191 tokens
    chunks = text_splitter.split_text(text)

    # Return the chunks
    return chunks

def pinecone_init():
    pinecone_environment = os.getenv('PINECONE_ENVIRONMENT')
    if not pinecone_environment:
        raise ValueError("Pinecone environment must be provided in environment variable PINECONE_ENVIRONMENT")
    pinecone_api_key = os.getenv('PINECONE_API_KEY')
    if not pinecone_api_key:
        raise ValueError("Pinecone API key must be provided in environment variable PINECONE_API_KEY")

    pinecone.init(api_key=pinecone_api_key, pinecone_environment=pinecone_environment)

def pinecone_setup():
    """Setup Pinecone environment and create index"""    

    pinecone_init()
    if INDEX_NAME not in pinecone.list_indexes():
        print(f'Creating Pinecone index {INDEX_NAME}')
        pinecone.create_index(name=INDEX_NAME, metric=METRIC, dimension=TEXT_EMBEDDINGS_DIM)
        pinecone_init()
    else:
        print(f'Pinecone index {INDEX_NAME} already exists', INDEX_NAME)
    
    pinecone_index = pinecone.Index(INDEX_NAME)        
    print(pinecone_index.describe_index_stats())
    return pinecone_index

def get_gmail_credentials():
    """Get Gmail credentials from credentials.json file or token.pickle file"""
    
    # If modifying these SCOPES, delete the file token.pickle.
    SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']

    creds = None
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            # Load credentials from credentials.json file
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            creds = flow.run_local_server(port=52102)
        # Save the credentials for the next run
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)
    return creds

# Function to decode the message part
def decode_part(part):
    data = part['body']['data']
    data = data.replace('-', '+').replace('_', '/')
    decoded_bytes = base64.urlsafe_b64decode(data)
    return decoded_bytes.decode('utf-8')

# Function to find the desired message part
def find_part(parts, mime_type):
    for part in parts:
        if part['mimeType'] == mime_type:
            return part
    return None

def index_gmail():
    pinecone_index = pinecone_setup()
    creds = get_gmail_credentials()
    embed = OpenAIEmbeddings(model=MODEL_NAME, openai_api_key=os.getenv('OPENAI_API_KEY'))

    try:
        service = build('gmail', 'v1', credentials=creds)

        # Call the Gmail API to fetch INBOX
        results = service.users().messages().list(userId='me',labelIds = ['INBOX']).execute()
        messages = results.get('messages',[])

        if not messages:
            print('No new messages.')
        else:
            pbar = tqdm(total=len(messages), ncols=70, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}')
            message_count = 0
            for message in messages:
                msg = service.users().messages().get(userId='me', id=message['id']).execute() # fetch the message using API
                email_data = msg['payload']['headers'] # Get data from 'payload' part of dictionary

                subject = ''
                for values in email_data: 
                    name = values['name']
                    if name == 'From':
                        from_name = values['value']
                    if name == 'Subject':
                        subject = values['value']
                    if name == 'Date':
                        date_value = values['value']
                        datetime_object = parser.parse(date_value) # Parsing date from string to datetime
                
                try:
                    data = None
                    payload = msg['payload']
                    if 'parts' in payload and len(payload['parts']) > 0:
                        part = find_part(payload['parts'], 'text/plain')
                        if part:
                            data = decode_part(part)
                        else:
                            part = find_part(payload['parts'], 'text/html')
                            if part:
                                data = decode_part(part)
                    if not data:
                        raise ValueError(f"Could not find body in message {message['id']}")
          
                    content = 'From: ' + from_name + '\n' + 'Date: ' + str(datetime_object) + '\n' + 'Subject: ' + subject + '\n' + data
                    
                    # Embed email data an add to Pinecone index
                    chunks = chunk_text(content)
                    embeds = embed.embed_documents(chunks)
                    ids = [message['id'] + '-' + str(i) for i in range(len(chunks))]
                    pinecone_index.upsert(vectors=zip(ids, embeds, [{'id': message['id']} for i in range(len(chunks))]))

                except BadRequest as e:
                    print(f"\nError while adding email {message['id']} to Pinecone: {e}")
                except ValueError as e:
                    print(f"\nError while adding email {message['id']} to Pinecone: {e}")
                finally:
                    pbar.update(1)
                    message_count += 1
                    
        print(f"Successfully added {message_count} emails to Pinecone")

    except Exception as error:
        print(f'An error occurred: {error}')
        raise error

def create_index():
    pinecone_init()
    pinecone.create_index(name=INDEX_NAME, metric=METRIC, dimension=TEXT_EMBEDDINGS_DIM)
    print(f"Successfully created index {INDEX_NAME}")

def delete_index():
    pinecone_init()
    pinecone.delete_index(INDEX_NAME)
    print(f"Successfully deleted index {INDEX_NAME}")

def search(query):
    pinecone_index = pinecone_setup()
    embed = OpenAIEmbeddings(model=MODEL_NAME, openai_api_key=os.getenv('OPENAI_API_KEY'))
    embed_query = embed.embed_documents(chunk_text(query))
    response = pinecone_index.query(queries=embed_query, top_k=5)
    print (response)

def main():
    if len(sys.argv) < 2:
        sys.exit('Usage: gmail_chat index | search')

    if sys.argv[1] == 'create':
        create_index()
    if sys.argv[1] == 'index':
        index_gmail()
    if sys.argv[1] == 'delete':
        delete_index()
    elif sys.argv[1] == 'search':
        if len(sys.argv) < 3:
            sys.exit('Usage: gmail_chat search <query>')
        search(sys.argv[2])

if __name__ == '__main__':
    main()