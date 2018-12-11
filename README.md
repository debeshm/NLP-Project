README ###################################################

Python packages that need to be installed gensim, pydot, nltk, keras, tensorflow

First dataset needs to be created.
Assuming it is available we can train and test out model.

To create dataset keep email csv file in a folder 'maildir' and run python makeData.py

To Train the model run trainer.py
To test the model run tester.py


FILE DESCRIPTIONS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



email_suggester.py-------------------------------------------

 * The code has been referred from https://github.com/asksonu/SmartReply/blob/master/Smart%20Reply_03_Apr_2018.py

 * The functions changed in this code are
    - load_vocab() : Changes it to load out Glove embeddings and text from the corpus that we used
    - Suggester.__init__() : Modified it to use GPU for training
    - reply() : Modified it to consider user input emails

 * The functions that we added are
    - create_reply_clusters() : Creates clusters for embeddings for reply text using KMeans
    - get_reply_embeddings() : Generates embeddings for the reply text
--------------------------------------------------------------

trainer.py----------------------------------------------------

 * This file contains the code that creates an object of Suggester class in email_suggester.py
   trains the model.

--------------------------------------------------------------

tester.py----------------------------------------------------

 * This file contains the code that creates an object of Suggester class in email_suggester.py
   and loads the pre trained model and tests the model

--------------------------------------------------------------

summarize.py---------------------------------------------------

 * This code contains a basic summarization algorithm for large texts.
 * To run this execute the command python summarize.py in the current directory.
 * This function takes a JSON file which contains email-response pairs and summarizes them

---------------------------------------------------------------

helpers.py-----------------------------------------------------

 * This code contains helper functions that helps in parsing extracting the basic components of an email
   such as body, sent to, send from.
 * we consider only the subject and the body of the email.
 * The emails are returned in a JSON format where the subject is the key and under the key the emails are present.

---------------------------------------------------------------

makeData.py----------------------------------------------------

 * Does basic pre procesing of emails
 * This file contains the code that reads ENRON dataset from a csv file and uses the 
   helper functions in helper.py to create the JSON data

----------------------------------------------------------------

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~