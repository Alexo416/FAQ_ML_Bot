"""

This file is where the logic of the bot is handled, this will read answers.txt and questions.txt file to store the utterances and intents. Intent matching will be used to read the users utterances and based on a score the bot will respond with the appropriate response, otherwise we will see if spaCy can handle the response, if not we will see if the user is angry, if that fails we will use OpenAI to generate the appropriate response back to the user.

"""

###Imports
import regex as re
import spacy
from spacy.matcher import Matcher
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import openai
from joblib import load

##Variables to be used throughout application
nlp = spacy.load("en_core_web_lg")
matcher = Matcher(nlp.vocab)
vectorizer = TfidfVectorizer(token_pattern=r"(?u)(\b\w+\b)", stop_words='english')
clf = load("clf.joblib")
corpus_vectorization = load("vectorizer.joblib")
openai.api_key = "REDACTED"



### Main
def file_input(filename):
    """Grabs each line in the .txt file and stores it into an array
        filename: the name of the file
    """
    lines = []
    with open(filename) as file:
        for line in file:
            lines.append(line.strip())
    return lines



def load_FAQ_data():
    """This method returns a list of questions and answers. The
    lists are parallel, meaning that intent n pairs with response n."""

    #uses the file_input method to store questions and answers
    questions = file_input("questions.txt") # load questions to be vectorized
    answers = file_input("answers.txt")

    #returns utterances and intents
    return questions, answers

def understand(utterance):
    """
    Once the method receives an utterance a match will be attempted to see which intent will be generated.
    If no intent is found a -1 will be generated.
    utterance: the response the user has uttered
    """
    max = 0
    global intents # declare that we will use a global variable

    #vectorize words and get the cosine similarity
    vectors = vectorizer.fit_transform(intents)
    new_vector = vectorizer.transform([utterance])
    similarities = cosine_similarity(new_vector, vectors)[0]

    for intent in intents:
        for i in range(len(similarities)):
            if similarities[i] > similarities[max]:
                max = i

    #Best score found to return appropriate responses to users
    if (similarities[max] >= 0.49):
        utterance = responses[max]

    try:
        return responses.index(utterance) #returns index of matching string
    except ValueError:
        return -1




def generate(intent, utterance):
    """Returns the response back to the user. If intent is not found spaCy will kick in, in order to appropriately respond back to the user, if spaCy fails then we will see if the user is angry and respond appropriately, otherwise we will use OpenAI to generate a response back to the user
    """

    global responses # declare that we will use a global variable

    if intent == -1: # if no match is made the failed response will head for processing



        #transform utterance and see if customer is angry based on sentiment corpus
        vector = corpus_vectorization.transform([utterance])
        prediction = clf.predict(vector)


        pattern = [{"LEMMA": {"IN": ["example"]}},
            {"LEMMA": {"IN": ["of", "like"]}},
            {"POS":"ADJ"},
            {"LEMMA": {"IN": ["film", "movie"]}}
                ]
        matcher.add("regionGenreNotFound",[pattern])

        pattern = [{"LEMMA":{"IN": ["movie", "film"]}},
                {"LEMMA":"about"},
                {"POS":"NOUN"}

        ]
        matcher.add("filmsNotFound",[pattern])

        pattern = [{"POS":"VERB"},
                    {"POS": "NOUN"},
                    {"POS":"ADP"}
        ]
        matcher.add("whyCantFindRegion",[pattern])

        pattern = [{"LOWER":"what"},
                    {"LOWER":"limits"},
        ]
        matcher.add("whatLimits",[pattern])

        pattern = [{"LOWER":"why"},
                    {"LOWER": "not"}
        ]
        matcher.add("whyNot",[pattern])


        #spaCy variables
        doc = nlp(utterance)
        matches = matcher(doc)


        for match_id, start, end in matches:
            string_id = nlp.vocab.strings[match_id]
            span = doc[start:end]
            if string_id == "regionGenreNotFound": #if string_id matches the appropriate pattern
                return "Sorry I cannot find any " +doc[start:end].text + ". You can try asking me about the type of genres that I know of instead."
            if string_id == "filmsNotFound":
                return "Ive never seen " +doc[start:end].text + " but I certainly would love to see that!"
            if string_id == "whyCantFindRegion":
                return "I have my limits :)"

            if string_id == "whatLimits":
                return "I dont want to provide you with an answer I do not know about yet, maybe later. To save one from a mistake is a gift of paradise. - Stilgar (Dune)"

            if string_id == "whyNot":
                return "Because I am just a simple bot that has its limits"

        for item in doc.ents:
            if item.label_ == 'GPE':
                return "Ive never been to " + item.text


        #if we have detected that the user is angry return response else if we have detected that the user is happy return a good response
        if prediction[0] == 0:
            return "Thats mean, if you would like to speak to a human about movies please call our number at 1-866-664-5696"


        #if we cannot answer anything at all we will use OpenAI to respond back to the user instead, temperature has been set to 0.3 to create a smaller range of responses compared to just a single response
        response = openai.Completion.create(engine="text-davinci-003", prompt=utterance, max_tokens=64, temperature=0.3)
        return response["choices"][0]["text"]


 #if no issues are found return the appropriate response back to the user
    return responses[intent]


## Load the questions and responses
intents, responses = load_FAQ_data()

## Main Program that runs in the shell

def chat():
    print()
    utterance = ""
    while True:
        utterance = input(">>> ")
        intent = understand(utterance)
        response = generate(intent, utterance)
        print(response)
        print()

        # if the utterance generates a response that signals the bot to quit the bot will exit
        if (response == "Bye for now!" or response =="Exiting..."):
            break;