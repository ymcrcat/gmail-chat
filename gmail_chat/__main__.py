import sys
import os
import os.path
import base64
import pickle
import cmd
import re
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from google.api_core.exceptions import BadRequest
import dateutil.parser as parser
from langchain.embeddings.openai import OpenAIEmbeddings
import pinecone
from tqdm import tqdm
import tiktoken
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone
from langchain.chat_models import ChatOpenAI
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.chains import RetrievalQA
from langchain.agents import Tool, initialize_agent


MODEL_NAME = "text-embedding-ada-002" # Name of the model used to generate text embeddings
# MAX_TOKENS = 8191 # Maximum number of tokens allowed by the model
MAX_TOKENS = 3800 # stay below the 4096 token limit for GPT-3
CHUNK_OVERLAP = 100 # Number of tokens to overlap between chunks
INDEX_NAME = "email-index"
TEXT_EMBEDDINGS_DIM = 1536 # Dimension of text embeddings
METRIC = "cosine"
GPT_MODEL = 'gpt-4'

def chunk_text(text):
    # Initialize the tokenizer
    tokenizer = tiktoken.encoding_for_model(GPT_MODEL)
    
    # Initialize the text splitter
    # text_splitter = CharacterTextSplitter.from_tiktoken_encoder(tokenizer.name)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=MAX_TOKENS,
                                                   chunk_overlap=CHUNK_OVERLAP)

    # Split the tokens into chunks of 8191 tokens
    chunks = text_splitter.split_text(text)

    # Return the chunks
    return chunks

def pinecone_init():
    pinecone_environment = os.getenv('PINECONE_ENVIRONMENT')
    if not pinecone_environment:
        sys.exit("Pinecone environment must be provided in environment variable PINECONE_ENVIRONMENT")
    pinecone_api_key = os.getenv('PINECONE_API_KEY')
    if not pinecone_api_key:
        sys.exit("Pinecone API key must be provided in environment variable PINECONE_API_KEY")

    pinecone.init(api_key=pinecone_api_key, pinecone_environment=pinecone_environment)

def pinecone_setup():
    """Setup Pinecone environment and create index"""    

    pinecone_init()
    if INDEX_NAME not in pinecone.list_indexes():
        raise ValueError(f'Pinecone index {INDEX_NAME} does not exist', INDEX_NAME)
    else:
        print(f'Pinecone index {INDEX_NAME} exists', INDEX_NAME)
    
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

def parse_date(date_string):
    try:
        date = parser.parse(date_string)
    except ValueError:
        try:
            cleaned_date_string = re.sub(r'\.\d+_\d+$', '', date_string)
            date = parser.parse(cleaned_date_string)
        except ValueError:
            date = None
    return date

# Function to decode the message part
def decode_part(part):
    if 'body' in part.keys():
        data = part['body']['data']
    else:
        return None
    data = data.replace('-', '+').replace('_', '/')
    decoded_bytes = base64.urlsafe_b64decode(data)
    return decoded_bytes.decode('utf-8')

# Function to find the desired message part
def find_part(parts, mime_type):
    for part in parts:
        if part['mimeType'] == mime_type:
            return part
    return None

message_count = 0 # Global variable to keep track of number of messages processed

def index_gmail():
    pinecone_index = pinecone_setup()
    creds = get_gmail_credentials()
    openai_api_key=os.getenv('OPENAI_API_KEY')
    if not openai_api_key:
        sys.exit("OpenAI API key must be provided in environment variable OPENAI_API_KEY")
    embed = OpenAIEmbeddings(model=MODEL_NAME, openai_api_key=openai_api_key)

    try:
        def process_email(msg):
            """Process email data and add to Pinecone index"""
            global message_count
            email_data = msg['payload']['headers']

            subject = ''
            to_name = ''
            for values in email_data: 
                name = values['name']
                if name == 'From':
                    from_name = values['value']
                if name == 'To':
                    to_name = values['value']
                if name == 'Subject':
                    subject = values['value']
                if name == 'Date':
                    date_value = values['value']
                    datetime_object = parse_date(date_value)
            
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
                    raise ValueError(f"Couldn't find body in message {msg['id']}")
        
                chunk_prefix = f"From: {from_name}\n" \
                  f"To: {to_name}\n" \
                  f"Date: {datetime_object}\n" \
                  f"Subject: {subject}\n"
                
                # Embed email data an add to Pinecone index
                chunks = [f'{chunk_prefix}{chunk}' for chunk in chunk_text(data)]
                embeds = embed.embed_documents(chunks)
                ids = [msg['id'] + '-' + str(i) for i in range(len(chunks))]
                pinecone_index.upsert(vectors=zip(ids, embeds, [{'id': msg['id'], 'text': chunks[i]} for i in range(len(chunks))]))

                message_count += 1

            except Exception as e:
                print(f"\nError while processing email {msg['id']}: {e}")

        # Define a function to get all messages recursively
        def get_all_emails(gmail, query):
            messages = []
            page_token=None
            
            while True:
                try:
                    result = gmail.users().messages().list(q=query, 
                                                        userId='me', 
                                                        maxResults=500, 
                                                        pageToken=page_token).execute()
                    messages.extend( result.get('messages', []) )
                    page_token = result.get('nextPageToken', None)
                    if not page_token:
                        break
                except HttpError as error:
                    print(f"An error occurred: {error}")
                    break
            return messages

        gmail = build('gmail', 'v1', credentials=creds)
        # A query to retrieve all emails, including archived ones
        query = "in:all"
        emails = get_all_emails(gmail, query)

        # Process and print the result
        for email in tqdm(emails, desc='Processing emails', file=sys.stdout):
            msg = gmail.users().messages().get(id=email.get('id'), userId='me', format='full').execute()
            process_email(msg)
                    
        print(f"Successfully added {message_count} emails to Pinecone.")
        print(pinecone_index.describe_index_stats())

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

def ask(query):
    openai_api_key=os.getenv('OPENAI_API_KEY')
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set")
    pinecone_index = pinecone_setup()

    embed = OpenAIEmbeddings(model=MODEL_NAME, openai_api_key=openai_api_key)
    llm = ChatOpenAI(openai_api_key=openai_api_key, 
                     model_name=GPT_MODEL,
                     temperature=0.0)

    # Answer question using LLM and email content
    text_field = "text" 
    vectorstore = Pinecone(pinecone_index, embed.embed_query, text_field)    
    qa = RetrievalQA.from_chain_type(llm=llm, 
                                     chain_type="stuff",
                                     retriever=vectorstore.as_retriever())
    result = qa.run(query)
    print(f'\n{result}')

def chat():
    openai_api_key=os.getenv('OPENAI_API_KEY')
    if openai_api_key is None:
        sys.exit("OPENAI_API_KEY environment variable is not set")
    embed = OpenAIEmbeddings(model=MODEL_NAME, openai_api_key=openai_api_key)
    pinecone_index = pinecone_setup()
    llm = ChatOpenAI(openai_api_key=openai_api_key, 
                     model_name=GPT_MODEL,
                     temperature=0.0)
    conversational_memory = ConversationBufferWindowMemory(
        memory_key='chat_history',
        k = 10,
        return_messages=True)
    text_field = "text"
    vectorstore = Pinecone(pinecone_index, embed.embed_query, text_field)
    qa = RetrievalQA.from_chain_type(llm=llm, 
                                     chain_type="refine",
                                     retriever=vectorstore.as_retriever())

    tools = [
        Tool(
            name = 'Email DB',
            func = qa.run,
            description=('useful to answer questions about emails and messages')
        )
    ]

    agent = initialize_agent(
        agent = 'chat-conversational-react-description',
        tools = tools,
        llm = llm,
        verbose = True,
        max_iterations = 5,
        early_stopping_method = 'generate',
        memory = conversational_memory
    )

    class InteractiveShell(cmd.Cmd):
        intro = 'Welcome to the Gmail Chat shell. Type help or ? to list commands.\n'
        prompt = '(Gmail Chat) '

        def do_quit(self, arg):
            "Exit the shell."
            print('Goodbye.')
            return True
        
        def emptyline(self):
            pass

        def default(self, arg):
            "Ask a question."
            try:
                result = agent.run(arg)
                print(f'\n{result}')
            except Exception as e:
                print(e)

    InteractiveShell().cmdloop()

def usage():
    sys.exit("""
    OPENAI_API_KEY, PINECONE_ENVIRONMENT, and PINECONE_API_KEY environment variables must be set.
    
    Usage: gmail_chat index | ask <query> | chat | create | delete
    """)

def main():
    if len(sys.argv) < 2:
        usage()

    if sys.argv[1] == 'create':
        create_index()
    elif sys.argv[1] == 'index':
        index_gmail()
    elif sys.argv[1] == 'delete':
        delete_index()
    elif sys.argv[1] == 'ask':
        if len(sys.argv) < 3:
            usage()
        ask(sys.argv[2])
    elif sys.argv[1] == 'chat':
        chat()
    else:
        usage()

if __name__ == '__main__':
    main()
