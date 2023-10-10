What are chatbots?
In recent years, Chatbots have become increasingly popular for automating simple conversations between users and software-platforms. Chatbots are capable of responding to user input and can understand natural language input. Python-NLTK (Natural Language ToolKit) is a powerful library that can be used to perform Natural Language Processing (NLP) tasks. In this tutorial, we will be creating a simple hardcoded chatbot using Python-NLTK.

What are the core concepts of chatbot creation?
The core concepts of chatbot creation are −

Natural Language Processing (NLP) − Chatbots use NLP to understand human language and interpret the user's intent. NLP involves techniques like tokenization, part-of-speech tagging, and named entity recognition.

Dialog Management − Dialog management is responsible for managing the flow of the conversation and maintaining context across multiple turns of the conversation.

Machine Learning − Machine learning is used to train chatbots to recognize patterns in data, make predictions, and improve over time. Techniques like supervised learning, unsupervised learning, and reinforcement learning are used in chatbot development.

APIs and Integrations − Chatbots often need to integrate with external services and APIs to provide information or complete tasks for the user.

User Experience (UX) − The user experience is critical for chatbots, as they should be easy and intuitive to use. UX considerations include designing conversational flows, choosing appropriate response types, and providing clear and helpful feedback to the user.

Prerequisites
Before we dive into the task few things should is expected to be installed onto your system −

List of recommended settings −

pip install pandas, matplotlib

It is expected that the user will have access to any standalone IDE such as VS-Code, PyCharm, Atom or Sublime text.

Even online Python compilers can also be used such as Kaggle.com, Google Cloud platform or any other will do.

Updated version of Python. At the time of writing the article I have used 3.10.9 version.

Knowledge of the use of Jupyter notebook.

Knowledge and application of virtual environment would be beneficial but not required.

It is also expected that the person will have a good understanding of statistics and mathematics.

Installation of Python-NLTK(http://www.nltk.org/install.html).

Familiarity with Text processing (Tokenization, Lemma, Stemming).

Installing Required Libraries
First, we need to install the required libraries for Developing a chatbot. NLTK, Regex, random and string libraries are required for chatbot development. To install these libraries, we can use pip command.

!pip install nltk
!pip install regex
!pip install random
!pip install string
Importing Required Libraries
After installing the necessary libraries, we need to import these libraries in our python notebook. Below is the code for importing these libraries.

import nltk
import re
import random
import string
from string import punctuation 
Preprocessing the Data
Once the required packages are installed and imported, we need to preprocess the data. Preprocessing includes removing all the unnecessary data, tokenizing the data into sentences, and removing stopwords. Stopwords are the most common words that have little or no meaning in the context of the conversation, such as ‘a’, ‘is’ etc.

# Download stopwords from nltk
nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(nltk.corpus.stopwords.words('english'))

def sentence_tokenizer(data):
   # Function for Sentence Tokenization
   return nltk.sent_tokenize(data.lower())

def word_tokenizer(data):
   # Function for Word Tokenization
   return nltk.word_tokenize(data.lower())

def remove_noise(word_tokens):
   # Function to remove stop words and punctuation
   cleaned_tokens = []
   for token in word_tokens:
      if token not in stop_words and token not in punctuation:
         cleaned_tokens.append(token)
   return cleaned_tokens 
Building a Chatbot
Now that we have performed preprocessing on the data, we are ready to build the chatbot. The flow of the chatbot can be summarized in the following steps −

Define the list of patterns and responses

Initialize an infinite while loop

Have the User Input a query

Tokenize the query and remove stop words

Match the query with one of the patterns and return a response.

# Define the Patterns and Responses
patterns = [
   (r'hi|hello|hey', ['Hi there!', 'Hello!', 'Hey!']),
   (r'bye|goodbye', ['Bye', 'Goodbye!']),
   (r'(\w+)', ['Yes, go on', 'Tell me more', 'I’m listening...']),
   (r'(\?)', ['I’m sorry, but I can’t answer that','Please ask me another question', 'I’m not sure what you mean.'])
]

# Function to generate response for the user input
def generate_response(user_input):
   # Append User Input to chat history
   conversation_history.append(user_input)
   # Generate Random response
   response = random.choice(responses)
   return response

# Main loop of chatbot
conversation_history = []
responses = [response for pattern, response in patterns]
while True:
   # User Input
   user_input = input("You: ")
   # End the Loop if the User Says Bye or Goodbye
   if user_input.lower() in ['bye', 'goodbye']:
      print('Chatbot: Goodbye!')
      break
   # Tokenize the User Input
   user_input_tokenized = word_tokenizer(user_input)
   # Remove Stop Words
   user_input_nostops = remove_noise(user_input_tokenized)
   # Process Query and Generate Response
   chatbot_response = generate_response(user_input_nostops)
   # Print Response
   print('Chatbot:', chatbot_response) 
Final program, code
import nltk
import re
import random
import string

from string import punctuation

# Download stopwords from nltk
nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(nltk.corpus.stopwords.words('english'))

def sentence_tokenizer(data):
   # Function for Sentence Tokenization
   return nltk.sent_tokenize(data.lower())

def word_tokenizer(data):
   # Function for Word Tokenization
   return nltk.word_tokenize(data.lower())

def remove_noise(word_tokens):
   # Function to remove stop words and punctuation
   cleaned_tokens = []
   for token in word_tokens:
      if token not in stop_words and token not in punctuation:
         cleaned_tokens.append(token)
   return cleaned_tokens

# Define the Patterns and Responses
patterns = [
   (r'hi|hello|hey', ['Hi there!', 'Hello!', 'Hey!']),
   (r'bye|goodbye', ['Bye', 'Goodbye!']),
   (r'(\w+)', ['Yes, go on', 'Tell me more', 'I’m listening...']),
   (r'(\?)', ['I’m sorry, but I can’t answer that', 'Please ask me another question', 'I’m not sure what you mean.'])
]

# Function to generate response for the user input
def generate_response(user_input):
   # Append User Input to chat history
   conversation_history.append(user_input)
   # Generate Random response
   response = random.choice(responses)
   return response

# Main loop of chatbot
conversation_history = []
responses = [response for pattern, response in patterns]
while True:
   # User Input
   user_input = input("You: ")
   # End the Loop if the User Says Bye or Goodbye
   if user_input.lower() in ['bye', 'goodbye']:
      print('Chatbot: Goodbye!')
      break
   # Tokenize the User Input
   user_input_tokenized = word_tokenizer(user_input)
   # Remove Stop Words
   user_input_nostops = remove_noise(user_input_tokenized)
   # Process Query and Generate Response
   chatbot_response = generate_response(user_input_nostops)
   # Print Response
   print('Chatbot:', chatbot_response)
Output
In this section we can see the output of the code: User input −


The user needs enter a string which is like a welcome message or a greeting, the chatbot will respond accordingly.


Based on the response the chatbot will create response









The chatbot ends the chat after the user writes bye in the input section.

Conclusion
In this tutorial, we have learned how to create a simple hardcoded Chatbot using Python-NLTK library with examples for each subsection. This chatbot can respond to user input with predefined responses. We also learned about Sentence Tokenization, Word Tokenization, removing Stop Words, and Pattern matching. These techniques will help in building more complex Chatbots.
