import discord
from transformers import pipeline
import wikipedia
import re
client = discord.Client()
from transformers import AutoTokenizer, AutoModelWithLMHead

tokenizer = AutoTokenizer.from_pretrained("airKlizz/t5-base-multi-en-wiki-news")
model = AutoModelWithLMHead.from_pretrained("airKlizz/t5-base-multi-en-wiki-news")

# summarizer = pipeline("summarization")
@client.event
async def on_ready():
    print('We have logged in as {0.user}'.format(client))

@client.event
async def on_message(message):
    if message.author == client.user:
        return
    # if message.content.startswith('$hello'):

    # get topics to be searched here

    articles = wikipedia.search(message.content)
    if articles is not None:
        context = wikipedia.page(title=articles[0])

        text = re.sub(r'==.*?==+', '', context.content)
        text = text.replace('\n', '')
        # await message.channel.send(context.summary)
        # T5 uses a max_length of 512 so we cut the article to 512 tokens.
        inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=512)

        # try question extraction on content here

        output = model.generate(inputs, max_length=512, min_length=96, length_penalty=2.0, num_beams=4, early_stopping=True)
        generated = tokenizer.decode(output[0])
        # print(generated)
        # generated = summarizer(context.content, max_length=512, min_length=30, do_sample=False)[0]
        await message.channel.send(generated)
    else:
        await message.channel.send("I could not find anything on that subject.")

client.run('NjM5NjU0ODEwMTE0NzE5NzQ1.Xbua9g._E8sPTCfIeK06R6aYENVxiRv-Xs')
