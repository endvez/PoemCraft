import re
import nltk
import json
from transformers import GPT2Tokenizer, GPT2LMHeadModel, pipeline
from nltk.corpus import cmudict
from sentence_transformers import SentenceTransformer, util
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"

nltk.download("cmudict")
cmu_dict = cmudict.dict()

CUSTOM_SYLLABLE_OVERRIDES = {
    "our": 1, "fire": 1, "hour": 1, "flower": 2, "power": 2,
    "prayer": 2, "prayers": 2, "every": 2, "fluttering": 3,
    "drifting": 2, "exhales": 2, "unmoored": 2
}

def count_syllables_word(word):
    word_clean = word.lower().strip(".,;!?\"'()[]")
    if word_clean in CUSTOM_SYLLABLE_OVERRIDES:
        return CUSTOM_SYLLABLE_OVERRIDES[word_clean]
    if word_clean in cmu_dict:
        return [len([ph for ph in pron if ph[-1].isdigit()]) for pron in cmu_dict[word_clean]][0]
    return max(1, len(re.findall(r'[aeiouy]+', word_clean)))

def count_syllables_line(line):
    words = re.findall(r"\b[\w']+\b", line)
    return sum(count_syllables_word(word) for word in words)

# replacce paths with path on your machine
tokenizer_poetry = GPT2Tokenizer.from_pretrained(r"E:\art_ml_final_proj\poetry-gpt2-style") 
model_poetry = GPT2LMHeadModel.from_pretrained(r"E:\art_ml_final_proj\poetry-gpt2-style").to(device)

tokenizer_haiku = GPT2Tokenizer.from_pretrained(r"E:\art_ml_final_proj\haiku_model")
model_haiku = GPT2LMHeadModel.from_pretrained(r"E:\art_ml_final_proj\haiku_model").to(device)
generator_haiku = pipeline("text-generation", model=model_haiku, tokenizer=tokenizer_haiku)

def generate_haiku_with_diagnostics():
    prompt = "<|style:haiku|> <|syllable:5-7-5|>\n"
    result = generator_haiku(prompt, max_length=32, num_return_sequences=1, do_sample=True, temperature=1.0)[0]["generated_text"]

    lines = result.replace(prompt, "").strip().split("\n")
    haiku_lines = [line.strip() for line in lines if line.strip()][:3]

    while len(haiku_lines) < 3:
        haiku_lines.append("")

    third_line = haiku_lines[2]
    if "." in third_line:
        first_period_index = third_line.index(".") + 1  
        haiku_lines[2] = third_line[:first_period_index].strip()

    diagnostics = []
    for i, (line, expected) in enumerate(zip(haiku_lines, [5, 7, 5])):
        actual = count_syllables_line(line)
        diagnostics.append({
            "line": line,
            "syllables": actual,
            "expected": expected,
            "difference": actual - expected
        })

    return haiku_lines, diagnostics

def truncate_naturally(poem, max_lines=20):
    lines = poem.strip().split("\n")
    for i, line in enumerate(lines[:max_lines]):
        if line.strip() == "":
            return "\n".join(lines[:i])
    for i in range(min(len(lines), max_lines)-1, 0, -1):
        if re.search(r"[.?!…]$", lines[i].strip()):
            return "\n".join(lines[:i+1])
    return "\n".join(lines[:max_lines])

def generate_poem(style, prompt="", max_lines=20, min_lines=6, max_attempts=5):
    input_text = f"<|style:{style}|>\n{prompt.strip()}\n"
    input_ids = tokenizer_poetry.encode(input_text, return_tensors="pt").to(device)

    for _ in range(max_attempts):
        output = model_poetry.generate(
            input_ids,
            max_length=200,
            temperature=0.9,
            top_p=0.95,
            do_sample=True,
            pad_token_id=tokenizer_poetry.eos_token_id
        )
        poem = tokenizer_poetry.decode(output[0], skip_special_tokens=True)
        cleaned = poem.replace(f"<|style:{style}|>", "").strip()
        truncated = truncate_naturally(cleaned, max_lines=max_lines)
        if len(truncated.splitlines()) >= min_lines:
            return truncated
    return "Could not generate a long enough poem after multiple attempts."

embedder = SentenceTransformer("all-MiniLM-L6-v2")
with open("combined_poems.json", "r", encoding="utf-8") as f:
    data = json.load(f)

def find_most_similar_real_poem(generated_text, style):
    if style.lower() == "haiku":
        return "(Not applicable for haiku)", 0.0

    candidates = [e['text'].split('\n', 1)[-1].strip()
                  for e in data if e.get('style', '').lower() == style.lower()]

    if not candidates:
        return "(No real poems found)", 0.0

    gen_embed = embedder.encode(generated_text, convert_to_tensor=True)
    real_embeds = embedder.encode(candidates, convert_to_tensor=True)
    cos_scores = util.pytorch_cos_sim(gen_embed, real_embeds)[0]
    best_idx = int(cos_scores.argmax())
    best_score = float(cos_scores[best_idx])
    return candidates[best_idx], best_score
from collections import Counter


def repetition_score(poem, n=2):
    """
    Returns a score between 0 and 1.
    Higher means more repetitive (bad).
    """
    words = re.findall(r"\b\w+\b", poem.lower())
    if len(words) < n:
        return 0.0

    ngrams = [tuple(words[i:i+n]) for i in range(len(words)-n+1)]
    counts = Counter(ngrams)
    total = len(ngrams)
    repeated = sum(1 for count in counts.values() if count > 1)
    
    return repeated / total if total else 0.0


style_map = {
    "1": "Shakespeare",
    "2": "Rudyard Kipling",
    "3": "Emily Dickinson",
    "4": "Walt Whitman",
    "5": "Edgar Allan Poe",
    "6": "haiku"
}

print("Choose a style:")
print("Styles: [1]William Shakespeare [2]Rudyard Kipling [3]Emily Dickinson [4]Walt Whitman [5]Edgar Allan Poe [6]haiku")
style_num = input("Enter a number (1–6): ").strip()
style = style_map.get(style_num)

if not style:
    raise ValueError("Invalid style number.")

prompt = input("Enter a prompt or keyword (optional): ").strip()

if style == "haiku":
    gen_poem = generate_haiku_with_diagnostics()
else:
    gen_poem = generate_poem(style, prompt)

real_poem, similarity = find_most_similar_real_poem(gen_poem, style)

if style == "haiku":
    lines, diag = gen_poem

    print("Generated Haiku:")
    for line in lines:
        print(line)
    
    haiku_text = "\n".join(lines)

    rep_score = repetition_score(haiku_text, n=2)
    print(f"Repetition Score: {rep_score:.2f}")

    print("\nSyllable Diagnostics:")
    for i, d in enumerate(diag, 1):
        print(f"Line {i}: {d['syllables']} syllables (expected {d['expected']}), diff = {d['difference']:+}")

else:
    print("\nGenerated Poem:\n")
    print(gen_poem)

    rep_score = repetition_score(gen_poem, n=2)
    print(f"Repetition Score: {rep_score:.2f}")

    real_poem, similarity = find_most_similar_real_poem(gen_poem, style)
    print("\nMost Similar Real Poem:\n")
    print(real_poem)
    print(f"\nSimilarity Score: {similarity:.4f}")