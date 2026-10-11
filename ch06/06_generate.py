import os
import sys
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.append('.')

from storybot.model import GPT
from storybot.tokenizer import BPETokenizer
from storybot.utils import get_device, generate

device = get_device()
tokenizer = BPETokenizer.load_from('storybot/merge_rules.pkl')
model = GPT.load_from('storybot/model_pretrain.pt', device = device)

prompt = "<|endoftext|>"
max_new_tokens = 300
temperature = 1.0
num_samples = 3

for i in range(num_samples):
    story = generate(model, tokenizer, prompt, max_new_tokens, temperature)
    print(story)