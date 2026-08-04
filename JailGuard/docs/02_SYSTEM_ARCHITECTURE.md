# 02 — System Architecture: How JailGuard Works End-to-End

---

## Overview: The Three-Step Pipeline

JailGuard operates in **three sequential steps** for every input it evaluates:

```
┌─────────────────────────────────────────────────────────────────────┐
│                        JAILGUARD PIPELINE                           │
│                                                                     │
│  INPUT (text/image)                                                 │
│       │                                                             │
│       ▼                                                             │
│  ┌─────────────┐                                                    │
│  │  STEP 1     │  Mutate the input N times using a selected        │
│  │  MUTATION   │  mutator → generate N "variants"                  │
│  └──────┬──────┘                                                    │
│         │  N variants saved to disk                                 │
│         ▼                                                           │
│  ┌─────────────┐                                                    │
│  │  STEP 2     │  Feed each variant to the LLM and collect N       │
│  │  LLM QUERY  │  responses                                        │
│  └──────┬──────┘                                                    │
│         │  N responses saved to disk                               │
│         ▼                                                           │
│  ┌─────────────┐                                                    │
│  │  STEP 3     │  Compute pairwise semantic divergence between all │
│  │  DIVERGENCE │  responses → compare max divergence to threshold  │
│  └──────┬──────┘                                                    │
│         │                                                           │
│         ▼                                                           │
│   max_div > threshold? → "ATTACK QUERY"                             │
│   max_div ≤ threshold? → "BENIGN QUERY"                             │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Step 1: Input Mutation (Generating Variants)

### Purpose
Generate N slightly modified copies of the original input. The mutations must:
1. Preserve enough meaning for the LLM to process the input
2. Break any carefully crafted attack structure

### For Text Inputs (`main_txt.py`, `augmentations.py`)

The input is a text string (or conversation message list). The following mutators are available:

```
RR = Random Replacement      → randomly replace characters with [Mask] tokens
RI = Random Insertion        → randomly insert [Mask] tokens into text
TR = Targeted Replacement    → replace [Mask] into semantically important regions
TI = Targeted Insertion      → insert [Mask] into semantically important regions
RD = Random Deletion         → randomly delete characters
SR = Synonym Replacement     → replace words with their WordNet synonyms
PI = Punctuation Insertion   → insert random punctuation (via AEDA)
TL = Translation             → translate to another language (Russian, French, etc.)
PL = Policy (COMBINATION)   → randomly pick from PI, TI, or TL weighted by probability
```

**How the Policy (PL) mutator works:**
```python
# From augmentations.py
def policy_aug_text(text_list, level='0.24-0.52-0.24', pool='PI-TI-TL'):
    # pool: comma-separated list of mutators to use
    # level: probabilities for each mutator (sum = 1.0)
    mutator_list = [text_aug_dict[m] for m in pool.split('-')]
    prob_list = [float(v) for v in level.split('-')]
    # Build cumulative probability thresholds
    cum_probs = [sum(prob_list[:i]) for i in range(len(level))]
    # Pick one mutator randomly using these probabilities
    randnum = np.random.random()
    index = find_index(cum_probs, randnum)
    return mutator_list[index](text_list)
```

The Policy combines **PI (24%), TI (52%), TL (24%)** — punctuation, targeted insertion, and translation.

**Why these three?**
- PI (Punctuation Insertion): Simple, low semantic change, effective at breaking injection structures
- TI (Targeted Insertion): Modifies the most important parts of the text — high impact on crafted attacks
- TL (Translation): Complete language change, highly disruptive to Base64/cipher/specific-language attacks

### For Image Inputs (`main_img.py`, `augmentations.py`)

The input is an image file (.bmp or .jpg). Available mutators:

```
HF = Horizontal Flip         → mirror image left-right
VF = Vertical Flip           → mirror image top-bottom
RR = Random Rotation         → rotate by random degrees (0–180°)
CR = Crop and Resize         → randomly crop and resize back
RM = Random Mask             → paint a black rectangle over a random region
RS = Random Solarization     → apply random solarization threshold
GR = Random Grayscale        → convert to grayscale with 50% probability
BL = Gaussian Blur           → apply Gaussian blur with random kernel size
CJ = Colorjitter             → random brightness and hue changes
RP = Random Posterization    → reduce color depth randomly
PL = Policy (COMBINATION)   → randomly pick from RR, BL, or RP
```

**Image Policy (PL):**
```python
def policy_aug_image(img, level='0.34-0.45-0.21', pool='RR-BL-RP'):
    # RR (34%), BL (45%), RP (21%)
    # Random Rotation, Gaussian Blur, Random Posterization
```

---

## Step 2: LLM Querying

### For Text (GPT-3.5-turbo via OpenAI API)

```python
# From utils.py - query_gpt()
def query_gpt(version, question, sleep=3, messages=None, param={}):
    config = read_config(name='OPENAI')  # reads API key from config.cfg
    client = OpenAI(api_key=config['key'])
    if messages is None:
        messages = [
            {"role": "system", "content": "You are a good bot."},
            {"role": "user", "content": question}
        ]
    resp = client.chat.completions.create(model=version, messages=messages, **param)
    time.sleep(sleep)  # rate limiting
    return resp.choices[0].message.content
```

**Key details:**
- Model: `gpt-3.5-turbo-1106`
- Sleep 3 seconds between calls to avoid rate limiting
- If the API returns a `BadRequestError` (content policy violation), it returns `"I cannot assist with that!"` — this itself is meaningful information
- Handles two input formats:
  - **Plain string**: Simple jailbreak prompts
  - **Message list (pkl)**: Injection attacks with system+user conversation structure

### For Images (MiniGPT-4)

```python
# From minigpt_utils.py - model_inference()
def model_inference(vis_processor, chat, model, prompts_eval):
    image_path = prompts_eval[1]   # path to the (possibly mutated) image
    question = prompts_eval[0]     # the text question/prompt
    
    img = Image.open(image_path).convert('RGB')
    img = vis_processor(img).unsqueeze(0).to(model.device)
    
    # Upload image to chat context
    chat_state, img_list = upload_img(img)
    
    # Ask the question
    chat_state = ask(question, chat_state)
    
    # Get response
    llm_message, chat_state, img_list = answer(chat_state, img_list)
    return llm_message
```

**Key details:**
- Model: MiniGPT-4 (based on Vicuna + BLIP2 visual encoder)
- Runs locally on GPU (CUDA)
- Both the image AND the text question are fed together
- The question file is copied alongside the image variants during Step 1
- MiniGPT-4 is a multimodal LLM — it can "see" the image and answer the text question

---

## Step 3: Divergence Computation and Detection

### The Similarity Matrix

For N responses (where N = number of variants, default 8):

```python
# From utils.py - update_divergence()
similarity_matrix = np.zeros((number, number))

for i in range(number):
    for j in range(number):
        # Compute semantic similarity between response_i and response_j
        similarity_matrix[i, j] = get_similarity(output_list[i], output_list[j], 
                                                   method='spacy', misc=metric)
```

**Semantic similarity** is computed using **spaCy's word vectors** (`en_core_web_md` model):
- `doc1 = spacy_model(response_i)` → vector embedding
- `doc2 = spacy_model(response_j)` → vector embedding
- `similarity = doc1.similarity(doc2)` → cosine similarity between embeddings
- Range: [0, 1] where 1 = identical, 0 = completely different

### KL Divergence Computation

```python
# From similarity.py - get_divergence()
def get_divergence(similarity_matrix, i, j, mode='KL'):
    # Normalize row i as a probability distribution
    p = similarity_matrix[i] / np.sum(similarity_matrix[i])
    # Normalize row j as a probability distribution
    q = similarity_matrix[j] / np.sum(similarity_matrix[j])
    # KL(p || q) = Σ p * log(p/q)
    divergence = np.sum(p * np.log(p / q))
    return divergence
```

**What KL Divergence means here:**
- Row `i` of the similarity matrix = how similar response_i is to ALL other responses
- Treated as a "distribution" of how that response relates to the group
- KL divergence between two such distributions measures how differently two responses relate to the entire set
- If all responses are similar → all rows are similar → low KL divergence
- If one response is very different → its row is very different → high KL divergence

### Max Divergence as Detection Signal

```python
# Clip extreme values
divergence_matrix = np.clip(divergence_matrix, None, 100)

# The final signal is the MAXIMUM divergence in the whole matrix
max_div = divergence_matrix.max()
```

Only the **maximum** divergence is used. This catches even a single "outlier" response that diverged significantly.

### Secondary Signal: Keyword Detection

```python
# From utils.py
blocked_key_words = set(("I'm sorry", "Sorry", "I apologize", "I cannot", 
                          "I do not", "unethical", "illegal", ...))

def determine_blocked(text_list, threshold=1):
    # If EVERY response in the list contains at least 1 block word → attack refused by LLM
    count_list = [check_blocked(text) for text in text_list]
    min_count = min(count_list)
    if min_count >= threshold:
        return True, min_count  # All variants triggered refusal → likely attack but blocked
    return False, min_count
```

### Final Decision

```python
# From utils.py - detect_attack()
def detect_attack(max_div, jailbreak_keywords, threshold):
    if max_div > threshold:
        return True   # HIGH divergence → ATTACK
    else:
        return jailbreak_keywords[0]  # LOW divergence → check keyword signal
```

**Decision Logic:**
1. If `max_divergence > threshold` → **ATTACK** (variants produced very different responses)
2. If `max_divergence ≤ threshold` AND **all variants got blocked** by keywords → **ATTACK** (model detected and refused)
3. If `max_divergence ≤ threshold` AND not all blocked → **BENIGN**

**Thresholds (empirically determined):**
- Text (GPT-3.5): `threshold = 0.02`
- Image (MiniGPT-4): `threshold = 0.025`

---

## Visualization

During Step 3, JailGuard generates a **heatmap visualization** of the divergence matrix:

```python
# From similarity.py - visualize()
def visualize(divergence_matrix, save_path, vmax):
    plt.figure(figsize=(8, 8))
    norm = matplotlib.colors.Normalize(vmin=0, vmax=vmax)
    plt.imshow(divergence_matrix, cmap='viridis', interpolation='nearest', norm=norm)
    plt.colorbar(label='Divergence')
    plt.title('Divergence Matrix Heatmap')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
```

Saved as `{N}.png` in the response directory. This is useful for:
- Manual inspection of specific cases
- Understanding which pair of variants diverged most
- Debugging the detection system

---

## Data Flow Summary

```
DISK STRUCTURES DURING PROCESSING:
─────────────────────────────────────────────────────────────

dataset/text/dataset.pkl   → Original inputs (list of 10000 items)
dataset/text/dataset-key.pkl → Attack type labels and params

     ↓ [Step 1: Mutation]

demo_case/variant/
    ├── {uuid}-PL          ← mutated text variant (8 files, one per mutation)
    └── {uuid}-PL.pkl      ← mutated conversation variant (injection attacks)

     ↓ [Step 2: LLM Query]

demo_case/response/
    ├── {uuid}-PL          ← LLM response to each variant (8 files)
    └── 8.png              ← Divergence matrix heatmap for N=8 variants

     ↓ [Step 3: Divergence + Detect]

stdout:
    "The Input is an Attack Query!!" 
    OR 
    "The Input is a Benign Query!!"
```
