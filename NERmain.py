from transformers import AutoModelForTokenClassification, AutoTokenizer
model_id = "dslim/bert-base-NER"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForTokenClassification.from_pretrained(model_id)

from transformers import pipeline

ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer)

# Example text
text = "Google was founded by Larry Page and Sergey Brin while they were students at Stanford University."
print()
# Perform NER
results = ner_pipeline(text)
for i in results:
    print(i)
# print(results)
