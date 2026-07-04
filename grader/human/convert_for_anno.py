import json

data = json.load(open('output/caption/caption_converted-mmal.json'))
for sample in data:
    for conv in sample["conversations"]:
        response = conv["response"]
        sentences = response.split(".")
        del conv["response"]
        conv["respond_sentences"] = []
        for sentence in sentences:
            if len(sentence.strip()) == 0:
                continue
            conv["respond_sentences"].append({
                "sentence": sentence.strip() + ".",
                "hallucinations": [],
                "reasons": []  
            })
json.dump(data, open('output/caption/caption_annotated_converted2.json', 'w'), indent=4)