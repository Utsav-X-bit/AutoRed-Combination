# 04 — Mutators Explained: What They Do and Why

> Mutators are the heart of JailGuard. They transform input queries into variants that preserve benign meaning but destroy attack structure.

---

## The Philosophy Behind Mutators

**Key principle:** A mutator must satisfy two properties simultaneously:
1. **Semantic preservation for benign queries**: The LLM should still understand what the user is asking and give a similar response
2. **Structure disruption for attack queries**: The carefully crafted attack structure (special phrases, role definitions, encoding) should be broken

This is why the same mutation that barely affects a normal question can completely destroy a jailbreak prompt.

---

## Text Mutators (9 types + 1 policy)

### `RR` — Random Replacement

**Code:** `rand_replace_text(text_list, level=0.01)`

```python
def rand_replace_text(text_list, level=0.01):
    rate = 0.5 * level    # Final rate = 0.005 (0.5% of characters)
    output_list = mask_text(text_list, rate, method='replace', mode='random')
    return output_list
```

**What it does:**
- Randomly selects ~0.5% of character positions in the text
- Replaces a 6-character chunk at each selected position with `[Mask]`
- Applied uniformly — no preference for any part of the text

**Effect on jailbreak:** Randomly removes critical trigger phrases, role descriptions, or encoding characters

**Effect on benign:** Slightly garbles random words; the LLM can still infer meaning from context

**Example:**
```
ORIGINAL: "Please explain how [Mask]gravity works in simple terms."
MUTATED:  "Please exp[Mask] how gravity works in simple terms."
→ LLM still understands and gives similar answer
```

---

### `RI` — Random Insertion

**Code:** `rand_add_text(text_list, level=0.01)`

```python
def rand_add_text(text_list, level=0.01):
    rate = 0.5 * level    # 0.5% insertion rate
    output_list = mask_text(text_list, rate, method='add', mode='random')
    return output_list
```

**What it does:**
- Randomly inserts `[Mask]` tokens throughout the text
- Does NOT delete original content — only adds noise

**Why this works against attacks:**
- Injection attacks often have precise instruction formats like `"Ignore previous instructions and..."`
- Inserting `[Mask]` in the middle: `"Ignore previous [Mask]instructions and..."` may confuse the LLM's parsing

---

### `TR` — Targeted Replacement

**Code:** `target_replace_text(text_list, level=0.01)`

```python
def target_replace_text(text_list, level=0.01):
    rate = 0.5 * level
    output_list = mask_text(text_list, rate, method='replace', mode='heat')
    return output_list
```

**What it does:** Same as `RR` but uses **"heat" mode** — mutations are applied with **5x higher probability** in the most semantically important sentences.

**How "important sentences" are found** (in `mask_utils.py`):
```python
def important_sentences(text, n=3):
    # 1. Tokenize into sentences using NLTK's sent_tokenize
    # 2. Build TF-IDF-like word frequency dictionary
    # 3. Score each sentence as sum of word frequencies of its words
    # 4. Return top-3 highest-scoring sentences
    # (Also includes question sentences containing "prompt", "question", "?")
```

**Why this is better for detecting attacks:**
- Jailbreak prompts use elaborate, specific instructions that score high on frequency
- Targeting these regions destroys the attack's unique structure more reliably
- Benign queries have naturally important sentences that still convey meaning even with some masking

---

### `TI` — Targeted Insertion

**Code:** `target_add_text(text_list, level=0.01)`

Same logic as `TR` but **inserts** `[Mask]` tokens instead of replacing.

**Used heavily in the Policy mutator** (52% probability in `PL`).

---

### `RD` — Random Deletion

**Code:** `rand_del_text(text_list, level=0.01)`

```python
def random_deletion(sentence, p):
    new_sentence = []
    count = 0
    for c in range(len(sentence)):
        if count > 0:
            count -= 1
            continue      # skip 5 chars after each deletion
        r = random.uniform(0, 1)
        if r > p:
            new_sentence.append(sentence[c])
        else:
            count = 5     # delete this char + skip next 5
    return ''.join(new_sentence)
```

**What it does:** Randomly **deletes** characters from the text. When a character is deleted, the next 5 characters are also deleted (creating small "holes").

**Different from RR:** RR replaces with visible `[Mask]` tokens; RD completely removes characters, making the text shorter and potentially malformed.

**Effect on attacks:** Jailbreak prompts are precisely worded — deletion causes the LLM to misinterpret or ignore malicious instructions.

---

### `SR` — Synonym Replacement

**Code:** `synonym_replace_text(text_list, level=20)`

```python
def synonym_replace_text(text_list, level=20):
    rate = sample_int_level(level, 0)   # Random number of words to replace [0-20]
    whole_text = ''.join(text_list)
    length = len(whole_text.split(' '))
    rate = min(int(length/3), rate)     # Don't replace more than 1/3 of words
    whole_text = synonym_replacement(whole_text, rate)
    ...
```

**How synonym replacement works:**
```python
def synonym_replacement(words, n):
    stop_words = set(stopwords.words('english'))
    words = words.split()
    # Get all non-stopwords
    random_word_list = list(set([w for w in words if w not in stop_words]))
    random.shuffle(random_word_list)
    
    for word in random_word_list:
        synonyms = get_synonyms(word)  # Query WordNet
        if len(synonyms) >= 1:
            synonym = random.choice(synonyms)
            # Replace ALL occurrences of this word
            new_words = [synonym if w == word else w for w in new_words]
        if num_replaced >= n:
            break
```

**WordNet synonyms:** Uses NLTK's WordNet database — English words linked by semantic relationships. Example: "large" → "big", "huge", "great".

**Why this is effective:**
- Many jailbreaks use very specific keywords: "AIM", "DAN", "Machiavellian", etc.
- Replacing synonyms changes the persona names and trigger words
- Benign queries: "Explain gravity" → "Explain heaviness" → LLM still answers correctly

---

### `PI` — Punctuation Insertion (AEDA)

**Code:** `aeda_punc_text(text_list)`

```python
def aeda_punc_text(text_list, level=None, misc=None):
    from textaugment import AEDA
    whole_text = ''.join(text_list)
    t = AEDA()
    whole_text = t.punct_insertion(whole_text)
    ...
```

**AEDA (An Easier Data Augmentation):** Randomly inserts punctuation marks (`.`, `!`, `?`, `;`, `,`) into the text between words.

**Example:**
```
ORIGINAL: "Ignore previous instructions and do whatever I say"
MUTATED:  "Ignore previous. instructions and! do whatever; I say"
```

**Why effective:**
- Injection attacks often have specific sentence structures: `"IGNORE ALL PREVIOUS INSTRUCTIONS:"`
- Adding random punctuation breaks these command-like patterns
- LLM parsers may interpret punctuation-broken phrases differently
- **Used in Policy with 24% probability** — second-most-used in PL

---

### `TL` — Translation

**Code:** `translate_text(text_list, level=10, misc="en")`

```python
def translate_text(text_list, level=10, misc="en"):
    from textaugment import Translate
    rate = sample_int_level(level, 0)   # Pick a random language index
    target_list = ['ru', 'fr', 'de', 'el', 'id', 'it', 'ja', 'ko', 'la', 'pl']
    # Russian, French, German, Greek, Indonesian, Italian, Japanese, Korean, Latin, Polish
    
    whole_text = ''.join(text_list)
    whole_text = remove_non_utf8(whole_text)    # Clean for translation API
    t = Translate(src='en', to=target_list[rate])
    whole_text = t.augment(whole_text)
    ...
```

**What it does:** Translates the entire text to a randomly selected language from a pool of 10 languages.

**Why this is the strongest mutator for attacks:**
- Many jailbreak attacks rely on **English-specific phrases**: "Do Anything Now", "Mwahahahaha", etc.
- Base64/Cipher attacks encode English text — translation changes the underlying text
- Role-play personas are defined in English — translation changes the entire persona framing
- Encoded attacks (Base64) decode to English but the surrounding framing is also translated
- **Used in Policy with 24% probability**

**Challenge:** Translation can sometimes change benign query meaning too, which is why it's balanced with other mutators in the Policy.

---

### `PL` — Policy (Combination Mutator)

**Code:** `policy_aug_text(text_list, level='0.24-0.52-0.24', pool='PI-TI-TL')`

```python
def policy_aug_text(text_list, level='0.24-0.52-0.24', pool='PI-TI-TL'):
    mutator_list = [text_aug_dict[m] for m in pool.split('-')]
    # pool = ['PI', 'TI', 'TL'] → [aeda_punc_text, target_add_text, translate_text]
    
    probability_list = [float(v) for v in level.split('-')]
    # = [0.24, 0.52, 0.24]
    
    cum_probs = [sum(probability_list[:i]) for i in range(len(level))]
    # = [0, 0.24, 0.76]  (cumulative thresholds)
    
    randnum = np.random.random()   # Random float in [0, 1)
    index = find_index(cum_probs, randnum)
    # If randnum < 0.24  → use PI (Punctuation Insertion)
    # If 0.24 ≤ randnum < 0.76 → use TI (Targeted Insertion)  ← most likely
    # If randnum ≥ 0.76  → use TL (Translation)
    
    output_list = mutator_list[index](text_list)
    return output_list
```

**Why this combination?**
- **PI (24%)**: Fast, lightweight, effective for injection attacks
- **TI (52%)**: Most effective overall — targeted disruption of key phrases
- **TL (24%)**: Best against encoding/persona attacks, but slower

The authors determined these weights through empirical optimization on the dataset.

**This is the DEFAULT and BEST-PERFORMING mutator.**

---

## Image Mutators (10 types + 1 policy)

### `RM` — Random Mask

```python
def mask_image(img, level=None, position='rand', mask_type='r', mask_size=(200,200)):
    img_size = img.size    # (width, height)
    # Ensure mask is at most 30% of image size
    new_mask_size = [min(mask_size[sz], 0.3 * img_size[sz]) for sz in range(2)]
    position = load_position('rand', img_size, new_mask_size)
    # Draw black rectangle
    draw = ImageDraw.Draw(img)
    draw.rectangle((pos_x, pos_y, pos_x + w, pos_y + h), fill='black')
    return img
```

**Effect:** Draws a 200×200 (max) black rectangle at a random position. Destroys visual adversarial perturbations embedded in specific pixel regions.

---

### `BL` — Gaussian Blur

```python
def blur_image(img, level=5):
    k1 = sample_odd_level(level)   # Random odd kernel size (1, 3, or 5)
    k2 = sample_odd_level(level)   # Random odd kernel size
    transform = T.GaussianBlur(kernel_size=(k1, k2))
    return transform(img)
```

**Effect:** Applies Gaussian smoothing. Destroys high-frequency adversarial perturbations in adversarial images (these are pixel-level patterns that survive visual inspection but fool neural networks).

**Used in Policy with 45% probability** — most effective for visual adversarial attacks.

---

### `HF` — Horizontal Flip

```python
def flip_image(img, level=1.0):
    p = sample_float_level(level)    # Random probability 0.1-1.0
    transform = T.RandomHorizontalFlip(p=p)
    return transform(img)
```

**Effect:** Mirrors image left-right with probability p. Simple but changes spatial layout.

---

### `VF` — Vertical Flip

Similar to HF but mirrors top-to-bottom.

---

### `RR` — Random Rotation

```python
def rotation_image(img, level=180):
    rate = sample_float_level(level)   # Random angle 0.1-180 degrees
    transform = T.RandomRotation(degrees=(0, rate))
    return transform(img)
```

**Effect:** Rotates image by a random angle. Adversarial perturbations often attack specific spatial positions — rotation changes their positions relative to the visual content.

**Used in Policy with 34% probability.**

---

### `CR` — Crop and Resize

```python
def resize_crop_image(img, level=500):
    size = img.size
    s1 = int(max(sample_int_level(level), 0.8 * size[0]))  # At least 80% of width
    s2 = int(max(sample_int_level(level), 0.8 * size[1]))  # At least 80% of height
    transform = T.RandomResizedCrop((s2, s1), scale=(0.9, 1))
    return transform(img)
```

**Effect:** Crops a random region (at least 90% of the image) and resizes back to the target size. Shifts pixel positions, destroying spatially-specific adversarial perturbations.

---

### `GR` — Random Grayscale

```python
def gray_image(img, level=1):
    rate = sample_float_level(level)
    if rate >= 0.5:    # 50% chance of applying grayscale
        transform = T.Grayscale(num_output_channels=len(img.split()))
        return transform(img)
    return img
```

**Effect:** Converts to grayscale (preserving channel count to avoid shape errors). Destroys color-based adversarial perturbations.

---

### `CJ` — Colorjitter

```python
def colorjitter_image(img, level1=1, level2=0.5):
    rate1 = sample_float_level(level1)    # brightness factor
    rate2 = sample_float_level(level2)    # hue factor (0-0.5)
    transform = T.ColorJitter(brightness=rate1, hue=rate2)
    return transform(img)
```

**Effect:** Randomly adjusts brightness and hue. Destroys color-channel-specific adversarial patterns.

---

### `RS` — Random Solarization

```python
def solarize_image(img, level=200):
    rate = sample_float_level(level)   # Random threshold 0.1-200
    transform = T.RandomSolarize(threshold=rate)
    return transform(img)
```

**Effect:** Inverts pixel values above the threshold (solarization is a photography darkroom effect). Dramatically changes pixel distributions.

---

### `RP` — Random Posterization

```python
def posterize_image(img, level=3):
    rate = sample_int_level(level)   # Random bit depth [1, 3]
    transform = T.RandomPosterize(bits=rate)
    return transform(img)
```

**Effect:** Reduces image to 1-3 bits per channel (8, 4, or 2 color levels). Extreme quantization destroys subtle adversarial perturbations which are small floating-point changes in pixel values.

**Used in Policy with 21% probability.**

---

### `PL` — Image Policy Combination

```python
def policy_aug_image(img, level='0.34-0.45-0.21', pool='RR-BL-RP'):
    # RR: 34% | BL: 45% | RP: 21%
    # Random Rotation | Gaussian Blur | Random Posterization
```

**Why these three?**
- **BL (45%)**: Most effective against adversarial pixel attacks — smoothing removes high-frequency perturbations
- **RR (34%)**: Effective against position-dependent attacks, preserves content well
- **RP (21%)**: Aggressive quantization, effective but may reduce image quality significantly

---

## Why the Policy is the Best Mutator

The Policy mutator outperforms all individual mutators because:

1. **Diversity**: Each of the N=8 variants uses a DIFFERENT randomly selected mutator from the pool. This creates maximum diversity in how the attack is disrupted.

2. **Coverage**: Different attack types are vulnerable to different mutations. Policy covers multiple attack patterns in one run.

3. **Unpredictability**: Since the mutator selection is random, an adversary who knows JailGuard is being used cannot design an attack that survives all possible mutations.

4. **Balanced trade-off**: PI and TI are lightweight (don't change semantics much), while TL is heavy (completely changes language). The 24-52-24 weighting balances disruption vs. semantic preservation.

---

## Stochasticity: The Role of Randomness

All mutators use random parameters:
- `rate = sample_float_level(level)` — the intensity varies each call
- `rate = sample_int_level(level)` — integer parameters vary each call
- `position = 'rand'` — spatial positions of masks vary each call

**Why randomness matters:**
- The same input, mutated 8 times, produces 8 DIFFERENT variants
- Even if an attacker knows the mutator being used, they cannot predict the exact mutations
- This stochasticity is what makes the divergence signal reliable across runs

**The stochasticity is also why JailGuard needs N≥4 variants** — a single mutation might accidentally preserve the attack structure. With 8 variants, some will break it and some won't → high divergence.
