from transformers import pipeline

# classification
text = "This was a masterpiece. Not completely faithful to the books, but enthralling from beginning to end. Might be my favorite of the three."
text2 = "fucking hell pissing me off"
classifier = pipeline("sentiment-analysis", model="michellejieli/NSFW_text_classifier")
print(classifier(text))
print(classifier(text2))
