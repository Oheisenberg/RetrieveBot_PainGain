from collections import Counter
from process_functions_for_EB import pos_tag, preprocess_sentence, compare_overlap, compute_similarity, extract_nouns
from responses_for_EB import responses, blank_spot
import spacy
import en_core_web_sm

nlp = spacy.load("en_core_web_sm")

exit_commands = ("end", "bye", "goodbye", "exit", "innabit", "no")

class ChatBot:
    def make_exit(self, user_message):
        for exits in exit_commands:
            if exits in user_message.lower():
                print("See ya later")
                return True
    
    def chat(self):
        user_message = input("Hi there, I am PainGain the Exercise Bot here to help you with any of your gym related queries!\n")
        while not self.make_exit(user_message):
          user_message = self.respond(user_message)

    def find_intent_class(self, responses, user_message):
        bow_user_message = Counter(preprocess_sentence(user_message))
        preprocess_responses = [Counter(preprocess_sentence(response)) for response in responses]
        similarity_words = [compare_overlap(bow_user_message, doc) for doc in preprocess_responses]
        response_index = similarity_words.index(max(similarity_words))
        return responses[response_index]
    
    def find_entity(self, user_message):
        tagged_user_message = pos_tag(preprocess_sentence(user_message))
        message_nouns = extract_nouns(tagged_user_message)
        tokens = nlp(" ".join(message_nouns))
        category = nlp(blank_spot)
        nlp_result = compute_similarity(tokens, category)
        nlp_result.sort(key=lambda x: x[2])
        if len(nlp_result) < 1:
            return blank_spot
        else:
            return nlp_result[-1][0]
        
    def respond(self, user_message):
        best_response = self.find_intent_class(responses, user_message)
        entity_category = self.find_entity(user_message)
        print(best_response.format(entity_category))
        input_message = input("Is there anything else I can help with?\n")
        return input_message

    
PainGain = ChatBot()

PainGain.chat()
