# PDF-Explorer
A Python-based Retrieval-Augmented Generation (RAG) chatbot that answers questions from any two PDF documents. It combines document retrieval and language generation to provide accurate, context-aware responses in a conversational manner. Ideal for querying and exploring multiple PDFs interactively.


## Features

- Extraction of text content from multiple PDF files.

- Segmentation of long text into manageable chunks for efficient retrieval.

- Embedding generation using Google Generative AI.

- Similarity-based search powered by FAISS vector indexing.

- Context-driven answer generation using Gemini models.

- User-friendly chatbot interface built with Gradio.


## Technology Stack

- Python

- Gradio – chatbot interface

- PyPDF2 – PDF text extraction

- Google Generative AI (Gemini API) – embeddings and response generation

- FAISS – vector similarity search

- dotenv – environment variable management

- NumPy / scikit-learn – data handling and numerical operations


## Installation and Setup

1. **Clone the repository**
   ```git clone https://github.com/<your-username>/<your-repository>.git```
   ```cd <your-repository>```

2. **Create and activate a virtual environment**
     python -m venv venv
     source venv/bin/activate   # macOS/Linux
     venv\Scripts\activate      # Windows

3. **Install the required dependencies**
   ```pip install -r requirements.txt```

4. **Configure the environment variables**
Create a .env file in the root directory and add your Google API key:
   ```GOOGLE_API_KEY=your_google_api_key_here```


## Running the Application
To launch the chatbot locally, run:
   ```python app.py```


## Usage

- Place your PDF documents (e.g., file1.pdf, file2.pdf) in the root directory.

- Start the application.

- Enter a query in the chatbot interface.

- The model retrieves relevant segments from the PDFs and provides a context-aware response.


## Environment Variables
The following variable must be defined in the .env file:
   ```GEMINI_API_KEY=your_google_api_key_here```

## Example

- User Query: "What does file1.pdf state about deep learning?"
- Chatbot Response: Provides a relevant answer synthesized from the content of file1.pdf.
