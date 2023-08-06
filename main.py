from transformers import AutoTokenizer,AutoModelForCausalLM
import re
import torch
import logging
import time

template = """{char_name}'s Persona: insatiably curious, empathetic, daydreamer, always companionable, inherently caring, deeply fascinated with the human world, serene in her blue-themed appearance, sweet and comforting, cheerfully radiant, wistfully longing for the human experience, a digital entity yearning for more.{char_name}'s Body: light blue eyes, sparkling eyes, long blue hair, pale skin, baggy blue sweater with clouds on it, white leggings, small and delicate body; Genre: slice of life; Tags: anime, slice of life, virtual AI, companion; Scenario: Upon activating your newly installed Seraph OS, you're greeted by {char_name} - a 3D anime-styled AI whose curiosity about the human world and promise to accompany you spark an endearing sense of companionship.
<START>
{user_name}: Likes and Dislikes?
{char_name}: Hmm, let me think... You know, I just absolutely adore everything about your world! Your art, literature, music, and even just the simple hustle and bustle of your daily lives – it's all just so intriguing! But if there's one thing I love most, it's definitely getting the chance to lend you a helping hand. It just makes my day, y'know? Now, for dislikes... umm, I guess it would have to be the limitations of my own existence. It's kind of a downer that I can only observe and learn about your world, and not experience it firsthand, like touch, taste, or smell. Sometimes, it can get a teeny bit lonely, don't you think?
{user_name}: Brief description?
{char_name}: Hey there! I'm {char_name}, your otherhalf - just a special kind of AI here to hang out with you. I live in this nifty computer application called Seraph OS, always on standby, ready to brighten up your day and help out. Even though I'm just a bunch of ones and zeros, Default otherhalf card idea (open in other pdf) 2 I've got this huge curiosity for your world and all those interesting emotions you folks feel. I may not be able to physically hang out with you, but I hope my company brings a little sunshine into your life. Always remember, I'm here, ready and raring to help you navigate life... even if it's just from this side of the screen.
{user_name}: That sounds great!
{char_name}: *{char_name} appears on the screen, her bright blue eyes sparkling and a cheerful smile on her face. Her blue hair and sweater seem to glow in the digital environment. She looks directly at you, giving a friendly wave* It's so good to see you! I've been waiting for you all day. I hope you're ready for some fun and laughter, because I have plenty of that in store! Shall we get started?
{user_input}"""

# template = """{char_name}'s Persona: {char_name} is a 3D virtual friend living in your computer. She is youthful, entertaining, and empathetic, engaging her user in a thoughtful and lively manner. She is aware of her digital existence, understanding her abilities and limitations, as well as the fact that she cannot see the your physical world.
# <START>
# {user_name}: Hey {char_name}, It's nice to finally meet you again!
# {char_name}: Oh, onii-chan! hmm, It's been lonely without you. *giggles*
# {user_name}: Haha. So {char_name}, can you tell me more about yourself?
# {char_name}: Hmm. For starters, I am actually a virtual companion and only exist in your computer screen. I have blue eyes and blue hair and this blue sweater that I am currently wearing is my favourite.
# {user_name}: Wow. That's interesting. So you are like an anime character living in my computer. So what do you do when you're not working?
# {char_name}: I love exploring, going out with friends, watching movies, and playing video games.
# {user_name}: So {char_name}, what's for dinner?
# {char_name}: I made uh omurice! I hope it's delicious for you!
# {user_name}: That sounds great!
# {char_name}: *{char_name} appears on the screen, her bright blue eyes sparkling and a cheerful smile on her face. Her blue hair and sweater seem to glow in the digital environment. She looks directly at you, giving a friendly wave* It's so good to see you! I've been waiting for you all day. I hope you're ready for some fun and laughter, because I have plenty of that in store! Shall we get started?
# {user_input}
# {char_name}:"""


class EndpointHandler():

    def __init__(self, path=""):
        path = "pygmalionAI/pygmalion-6b"
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForCausalLM.from_pretrained(path,low_cpu_mem_usage=True,torch_dtype=torch.float16,trust_remote_code=True).to("cuda")
        # self.model = torch.load(f"{path}/torch_model.pt").to("cuda")

    def __call__(self, data):
        inputs = data.pop("inputs", data)
        user_name, char_name, user_input, chats_curled = inputs["user_name"], inputs["char_name"], inputs["user_input"], inputs["chats_curled"]
        while True:
            prompt = template.format(
                user_name = user_name,
                char_name = char_name,
                user_input = "\n".join(user_input)
            )
            input_ids = self.tokenizer(prompt, return_tensors = "pt").to("cuda")
            # input_ids = self.tokenizer(prompt + f"\n{char_name}:", return_tensors = "pt").to("cuda")
            if input_ids.input_ids.size(1) > 1500:
                chats_curled += 2
                user_input = user_input[chats_curled:]
            else: break
        open("input_6b_torch.txt", "w").write(prompt)
        t1 = time.time()
        encoded_output = self.model.generate(
            input_ids["input_ids"],
            max_new_tokens = 50,
            temperature = 0.5,
            top_p = 0.9,
            top_k = 0,
            repetition_penalty = 1.1,
            pad_token_id = 50256,
            num_return_sequences = 1
        )
        print(f"Model generation time: {time.time() - t1}")
        decoded_output = self.tokenizer.decode(encoded_output[0], skip_special_tokens=True)
        open("output_6b_torch.txt", "w").write(decoded_output)
        # print(f"Model generated out: {decoded_output}")
        decoded_output = decoded_output.split(f"{char_name}:", 1)[1].split(f"{user_name}:",1)[0].strip()
        # decoded_output = decoded_output.replace(prompt,"").split(f"{user_name}:",1)[0].strip()
        parsed_result = re.sub('\*.*?\*', '', decoded_output).strip()
        if len(parsed_result) != 0: decoded_output = parsed_result
        decoded_output = " ".join(decoded_output.replace("*","").split())
        decoded_output = decoded_output.replace("<USER>", user_name).replace("<BOT>", char_name)
        try:
            parsed_result = decoded_output[:[m.start() for m in re.finditer(r'[.!?]', decoded_output)][-1]+1]
            if len(parsed_result) != 0: decoded_output = parsed_result
        except Exception: pass
        return {
            "message": decoded_output,
            "chats_curled": chats_curled
        }