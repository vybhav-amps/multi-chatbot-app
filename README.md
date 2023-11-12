# Multi-Chatbot Manager
## Description

### This Streamlit application provides a platform for managing multiple chatbots. 
### Each chatbot can be independently configured with different models, settings, and data sources. 
### The application allows users to add new chatbots, select and interact with existing ones, and view their chat histories. 
### It's designed to work with LangChain, leveraging various models like GPT-3.5 and GPT-4 for conversational AI.

## Features

### Multiple Chatbots: Create and manage multiple chatbots with unique IDs.
### Customizable Settings: Configure each chatbot with different models, temperatures, and token limits.
### PDF Document Upload: Upload PDF documents as a data source for each chatbot.
### Independent Chat Histories: View and manage the chat history of each chatbot.
### Database Integration: Store chat histories in a SQLite database for persistence.
### Installation and Setup
### Clone the Repository: Clone this repository to your local machine.
### Install Requirements: Install the required Python packages using pip:
#### bash:
#### pip install streamlit dotenv sqlite3 langchain
### Environment Variables: Create a .env file in the project directory with your OpenAI API key:
#### makefile:
#### OPENAI_API_KEY=your_api_key_here
### Run the Application: Run the application using Streamlit:
#### bash:
#### streamlit run your_script_name.py

## Usage

### Add a New Chatbot: Enter a unique ID for the chatbot and click "Add Chatbot."
### Select a Chatbot: Use the dropdown menu to select an existing chatbot.
### Configure Chatbot Settings: Choose the model, set temperature, and max tokens for the selected chatbot.
### Upload a PDF Document: Optionally upload a PDF document to use as a data source for the chatbot.
### Interact with the Chatbot: Enter a question in the text input field and submit to receive a response.
### View Chat History: Scroll through the chat history for the selected chatbot.
### Clear History/Delete Chatbot: Use the provided buttons to clear the chat history or delete a chatbot.
## Code Structure
### initialize_database: Sets up the SQLite database to store chat histories.
### clear_history: Clears the chat history for a specific chatbot.
### create_vector_store_from_pdf: Processes uploaded PDF files for chatbot data sourcing.
### save_vector_store & load_vector_store: Handles saving and loading of the vector store used by chatbots.
#### main: The main function where the Streamlit UI components are defined.
#### chatbot_controls: Handles the UI and logic for individual chatbot settings and interaction.
