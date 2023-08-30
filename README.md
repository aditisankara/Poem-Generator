# Poem-Generator

Text preprocessing - 
First, the poems are imported from text files and converted into a flat list. 
The class shown below encapsulates the tokenization functionality provided by a Transformers tokenizer. The encodes() method tokenizes input data and converts it into token IDs, while the decodes() method converts token IDs back into a human-readable string.
We then create a transformed list (tls) using the TfmdLists class, applying the TransformersTokenizer transformation to each element in the all_ballads list. 
The HuggingFace model returns a tuple in outputs, with the actual predictions and some additional activations. To work inside the fastai training loop, we will need to drop those using a Callback: we use those to alter the behavior of the training loop.

Model Definition and training -
The “Learner”, i.e., the model, is then defined as follows
learn = Learner(dls, model, loss_func=CrossEntropyLossFlat(), cbs=[DropOutput], metrics=Perplexity()).to_fp16()

The lr_find() method of the the fastai library is used to find an appropriate learning rate, which is then used to train the model as follows - 
learn.fit_one_cycle(10, 1e-4)

After training the model for 10 epochs, we arrive at a perplexity of 19.9109. 

Results
The caption generated is fed as a “prompt”, i.e., an input to the GPT-2 model trained on poetic text.
prompt = caption 
prompt_ids = tokenizer.encode(prompt)
inp = tensor(prompt_ids)[None].cuda()

We generate a poem with a maximum length of 100 words. 
preds = learn.model.generate(inp, max_length=100, num_beams=5, no_repeat_ngram_size=2, early_stopping=True, pad_token_id=50256)
print("Output:\n" + 100 * '-')
print(tokenizer.decode(preds[0].cpu().numpy(), skip_special_tokens=True))

The output is as follows -
Two dogs are playing with each other, 
And one of them is lying on the floor 
With a broken arm. 
"He's a dog," I say, "but he's got a brain." 
He replies: "I don't think so. I mean, look at him! Look at his eyes! They look like a child's eyes; They're full of hope! 'Twas the end of the world's desire 
For me when I was a boy.
