# FAQ_ML_Bot :robot:

This is a machine learning powered FAQ Bot for Film Genres that incorporates the following

 * TfidVectorizer for intent matching
 * spaCy for Named Entity Recognition and extracting Noun Chunks as a fallback strategy
 * OpenAI gpt-3 transformer if there are no intents found for the users utterance
 * Pickled classifier object to classify the users tone in order to issue a response if the user is upset
 * This can also function as a Discord bot with a slight modification
