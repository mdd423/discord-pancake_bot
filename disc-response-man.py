import discord
# from transformers import pipeline
client = discord.Client()
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

# summarizer = pipeline("summarization")
@client.event
async def on_ready():
    print('We have logged in as {0.user}'.format(client))

@client.event
async def on_message(message):
    if message.author == client.user:
        return
    # messages = ''
    # only remembers the last 6 messages in conversation
    max_iter = 6
    iter = 0
    for cached_message in reversed(client.cached_messages):
        if cached_message.channel == message.channel:
            # messages += cached_message.content + '\n'
            msg_tokens = tokenizer.encode(cached_message.content + tokenizer.eos_token, return_tensors='pt')
            chat_history_ids = torch.cat([chat_history_ids,msg_tokens], dim=-1) if iter > 0 else msg_tokens
            iter += 1
            if iter >  max_iter:
                break
    # print(messages)
    # chat_history_ids = tokenizer.encode(messages + tokenizer.eos_token, return_tensors='pt')

    # if message.content.startswith('$hello'):
    new_user_input_ids = tokenizer.encode(message.content + tokenizer.eos_token, return_tensors='pt')

    # append the new user input tokens to the chat history
    bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) #if step > 0 else new_user_input_ids

    # generated a response while limiting the total chat history to 1000 tokens,
    chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)

    generated = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)

    # pretty print last ouput tokens from bot
    print('user: ' + message.content)
    print('gene: ' + generated)
    await message.channel.send(generated)

client.run()
