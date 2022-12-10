from transformers import pipeline
from transformers import DistilBertTokenizerFast

# classification
text = "This was a masterpiece. Not completely faithful to the books, but enthralling from beginning to end. Might be my favorite of the three."
text2 = "fucking hell pissing me off"
text3 = "I see you’ve set aside this special time to humiliate yourself in public."
text4 = "Wow, congratulations! So excited for you!"
# hugging-face model
classifier = pipeline("sentiment-analysis", model="michellejieli/NSFW_text_classifier")
# locally 
local_classifier = pipeline("sentiment-analysis", model="/workspaces/Michelle_Li_NLP_Project/hugging-face/models")
# print results of local model
print(local_classifier(text))
print(local_classifier(text2))
print(local_classifier(text3))
print(local_classifier(text4))
print(classifier(text))
print(classifier(text2))
print(classifier(text3))
print(classifier(text4))