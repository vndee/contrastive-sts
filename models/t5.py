from transformers import T5Tokenizer, T5ForConditionalGeneration


class TextGenerator:
    def __init__(self):
        self.tokenizer = T5Tokenizer.from_pretrained('t5-large')
        self.model = T5ForConditionalGeneration.from_pretrained('t5-large').to('cuda')

    def __call__(self, text):
        input_ids = self.tokenizer.encode(f"summarize: {text}", return_tensors="pt").to('cuda')
        outputs = self.model.generate(input_ids)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)


text_generator = TextGenerator()
res = text_generator('Officials at Brandeis said this was an "extremely heartrending" time for the campus.')
print(res)
