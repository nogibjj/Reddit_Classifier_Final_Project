from transformers import AutoConfig, AutoModelForSequenceClassification

# define the mappings as dictionaries
label2id = {"NSFW": 1, "SFW": 0}
id2label = {1: "NSFW", 0: "SFW"}
# define model checkpoint - can be the same model that you already have on the hub
model_ckpt = "michellejieli/NSFW_text_classifier"
# define config
config = AutoConfig.from_pretrained(model_ckpt, label2id=label2id, id2label=id2label)
# load model with config
model = AutoModelForSequenceClassification.from_pretrained(model_ckpt, config=config)
# export model
model.save_pretrained("./models")
