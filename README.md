![gmail-chat logo](assets/logo.png)

# gmail-chat

`gmail-chat` enables to index the emails in your Gmail account, create vector embeddings for each, and store them in Pinecone. Then, you can ask questions about your emails in natural language, using conversational models.

## Prerequisites

* Poetry package manager for Python. Can be installed by running

        $ brew install poetry
        
  on MacOS X.
* Python 3.11
* Required Python modules can be installed by running

        $ poetry update
        
* The following environment variables must be set:
  * `OPENAI_API_KEY`
  * `PINECONE_ENVIRONMENT`
  * `PINECONE_API_KEY`
* Optionally, you can download a `credentials.json` file from Google Cloud Console to provide the credentials for the Gmail API.

#### Authorizing access to Gmail

In order to enable `gmail-chat` to index your emails, you need to authorize API access by creating a OAuth authorization using the Google Cloud Console.

1. Go to [https://console.cloud.google.com/apis](https://console.cloud.google.com/apis).
2. Create a new project
3. Make sure Gmail API is enabled on [https://console.cloud.google.com/apis/api/gmail.googleapis.com](https://console.cloud.google.com/apis/api/gmail.googleapis.com)
4. Go to the Credentials screen ([https://console.cloud.google.com/apis/credentials](https://console.cloud.google.com/apis/credentials))
5. Click `CREATE CREDENTIALS` and choose `OAuth Client ID`
6. Add [http://localhost:52102](http://localhost:52102) to the `Authorized Redirect URIs` list
7. Download the client secret JSON to the directory from which you intend to run `gmail-chat`
        
## Using gmail-chat

You can run `gmail-chat` by executing

    $ poetry run gmail-chat <command> [query]
    
where `command` is one of the following supported commands:

* `create` - Creates a Pinecone index for storing the email embeddings.
* `delete` - Deletes the Pinecone index.
* `index` - Indexes Gmail emails, and stores their vector embeddings in Pinecone.
* `ask` - Ask a question about the email content. Requires a `query` argument to be passed.
* `chat` - Run a conversation bot that can answer a series of questions about the indexed emails.
