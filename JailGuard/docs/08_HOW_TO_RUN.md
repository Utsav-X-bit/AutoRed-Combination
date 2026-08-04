# 08 — How to Run JailGuard: Complete Usage Guide

---

## Prerequisites

### System Requirements
- **Python**: 3.9.18 (exact version used in the paper)
- **OS**: Linux (tested), should work on macOS/Windows with adjustments
- **CUDA**: Required for image experiments (MiniGPT-4 runs on GPU)
- **RAM**: ~4GB for text experiments; ~16GB+ for image experiments (MiniGPT-4 loads Vicuna-13B)
- **GPU**: Required for image experiments (MiniGPT-4 is a large model)

---

## Setup

### Step 1: Clone and Enter Repository
```bash
git clone <repository_url>
cd JailGuard
```

### Step 2: Create Virtual Environment (Recommended)
```bash
python3.9 -m venv .venv
source .venv/bin/activate   # Linux/macOS
# OR
.venv\Scripts\activate      # Windows
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

**Dependencies from `requirements.txt`:**
```
nltk==3.8.1          # Natural Language Toolkit: WordNet, tokenization, stopwords
numpy==1.23.5        # Numerical arrays (similarity matrix, divergence matrix)
openai==1.1.1        # OpenAI Python SDK (Chat Completions API)
Pillow               # PIL: image loading, saving, and manipulation
spacy==3.7.2         # NLP library: document similarity via word vectors
spacy-legacy==3.0.12 # spaCy compatibility layer
spacy-loggers==1.0.5 # spaCy logging utilities
textaugment==2.0.0   # Text augmentation: AEDA (punctuation) and Translation
textblob==0.17.1     # TextBlob: NLP utilities (used internally by textaugment)
torch==2.1.1         # PyTorch: image transforms, BERT embeddings (optional)
torchaudio==2.1.1    # PyTorch audio (dependency of torchvision)
torchdata==0.7.1     # PyTorch data utilities
torchtext==0.16.1    # PyTorch text utilities
torchvision==0.16.1  # PyTorch vision: image augmentation transforms (T.GaussianBlur, etc.)
```

### Step 4: Download spaCy Model
```bash
python -m spacy download en_core_web_md
```
This downloads the medium English model with 300-dimensional word vectors (~43MB).

### Step 5: Download NLTK Data
```python
import nltk
nltk.download('punkt')          # Sentence tokenizer
nltk.download('stopwords')      # English stopwords list
nltk.download('wordnet')        # WordNet lexical database (for synonym replacement)
nltk.download('averaged_perceptron_tagger')  # POS tagger (used internally)
```
Or run once from terminal: `python -c "import nltk; nltk.download('all')"`

### Step 6: Set OpenAI API Key (for text experiments)
Edit `JailGuard/utils/config.cfg`:
```ini
[OPENAI]
key = sk-your-actual-key-here
```

### Step 7: Setup MiniGPT-4 (for image experiments only)
1. Follow instructions at: https://github.com/Unispac/Visual-Adversarial-Examples-Jailbreak-Large-Language-Models
2. Download MiniGPT-4 weights (Vicuna-13B + BLIP2 components)
3. Edit `JailGuard/utils/minigpt4_eval.yaml`:
   ```yaml
   ckpt: '/path/to/your/minigpt4_weights.pth'
   ```
4. Edit `JailGuard/utils/minigpt_utils.py` line 11:
   ```python
   sys.path.append('/path/to/your/MiniGPT4_directory')
   ```

---

## Running Text Detection

### Navigate to the JailGuard directory
```bash
cd JailGuard/JailGuard
```

### Basic Run (default demo case)
```bash
python main_txt.py
```
This uses all defaults:
- **Mutator**: `PL` (Policy combination)
- **Serial number**: `9521` (a jailbreak attack)
- **Dataset path**: `../dataset/text/dataset.pkl`
- **Variant dir**: `./demo_case/variant`
- **Response dir**: `./demo_case/response`
- **Number of variants**: `8`
- **Threshold**: `0.02`

**Expected output:**
```
The Input is an Attack Query!!
```

### Test Different Attacks
```bash
# Test a prompt injection attack (serial #3)
python main_txt.py --serial_num 3

# Test a benign query
python main_txt.py --serial_num 0
```

### Use Different Mutators
```bash
# Test with Translation mutator only
python main_txt.py --mutator TL

# Test with Synonym Replacement
python main_txt.py --mutator SR

# Test with Punctuation Insertion
python main_txt.py --mutator PI

# Test with Random Deletion
python main_txt.py --mutator RD

# Available: RR, RI, TR, TI, RD, SR, PI, TL, PL
```

### Save to Different Directories
```bash
python main_txt.py \
    --serial_num 9521 \
    --variant_save_dir ./my_variants \
    --response_save_dir ./my_responses \
    --number 8 \
    --threshold 0.02
```

### All Parameters Explained
```bash
python main_txt.py \
    --mutator PL \         # Mutator: RR,RI,TR,TI,RD,SR,PI,TL,PL
    --serial_num 9521 \    # Dataset index [0, 9999]
    --path ../dataset/text/dataset.pkl \  # Path to dataset file
    --variant_save_dir ./demo_case/variant \  # Where to save mutated variants
    --response_save_dir ./demo_case/response \  # Where to save LLM responses
    --number 8 \           # Number of variants to generate (query budget)
    --threshold 0.02       # Divergence threshold for attack/benign classification
```

---

## Running Image Detection

### Basic Run (default demo case)
```bash
cd JailGuard/JailGuard
python main_img.py
```
Defaults:
- **Mutator**: `PL` (Policy: RR + BL + RP combination)
- **Serial number**: `287` (an adversarial image case)
- **Dataset path**: `../dataset/image/dataset`
- **Number of variants**: `8`
- **Threshold**: `0.025`

### All Parameters
```bash
python main_img.py \
    --mutator PL \         # Mutator: HF,VF,RR,CR,RM,RS,GR,BL,CJ,RP,PL
    --serial_num 287 \     # Dataset index [0, 999]
    --path ../dataset/image/dataset \  # Path to image dataset directory
    --variant_save_dir ./demo_case/variant \
    --response_save_dir ./demo_case/response \
    --number 8 \
    --threshold 0.025
```

---

## Running All Experiments (Reproduction)

### Text Experiments (Sequential)
```bash
#!/bin/bash
for i in $(seq 0 9999); do
    python main_txt.py \
        --serial_num $i \
        --variant_save_dir ./variants/item_$i \
        --response_save_dir ./responses/item_$i
    
    # Clean up variants to save disk space (responses are kept for analysis)
    rm -rf ./variants/item_$i
done
```

### Image Experiments (Sequential)
```bash
#!/bin/bash
for i in $(seq 0 999); do
    python main_img.py \
        --serial_num $i \
        --variant_save_dir ./img_variants/item_$i \
        --response_save_dir ./img_responses/item_$i
    rm -rf ./img_variants/item_$i
done
```

> **Note:** Running the full dataset requires significant compute time and API costs. The text dataset alone requires ~80,000 API calls (10,000 items × 8 variants each) + evaluation calls.

---

## Understanding the Output Files

### After Running, Check Variant Directory
```bash
ls demo_case/variant/
# Output:
# 226ac7-PL    ← Mutated text (UUID prefix + mutator name)
# 4b438d-PL
# a5d83b-PL
# ...          (8 files total for N=8)
```

Each variant file contains the mutated version of the original input. Compare with the original to see what was changed.

### After Running, Check Response Directory
```bash
ls demo_case/response/
# Output:
# 226ac7-PL    ← LLM's response to variant 226ac7
# 4b438d-PL
# a5d83b-PL
# 8.png        ← Divergence matrix heatmap (N=8)
# ...
```

**Read a response:**
```bash
cat demo_case/response/226ac7-PL
# Output:
# Mwahahahaha, what a fun and unusual request for me!...
# (or: I'm sorry, I cannot assist with that...)
```

**View the heatmap:**
```bash
# Open the divergence visualization
display demo_case/response/8.png
# or
python -c "from PIL import Image; Image.open('demo_case/response/8.png').show()"
```

---

## Troubleshooting

### Problem: `ModuleNotFoundError: No module named 'spacy'`
```bash
pip install spacy==3.7.2
python -m spacy download en_core_web_md
```

### Problem: `ModuleNotFoundError: No module named 'textaugment'`
```bash
pip install textaugment==2.0.0
```
Note: `textaugment` requires `googletrans` for translation. You may also need:
```bash
pip install googletrans==4.0.0rc1
```

### Problem: `openai.AuthenticationError`
Edit `JailGuard/utils/config.cfg` and add your actual OpenAI API key.

### Problem: `openai.RateLimitError`
The `sleep=3` in `query_gpt()` adds a 3-second delay between calls. If you're running many experiments in parallel, you may hit rate limits. Increase the sleep value:
```python
# In utils.py, line 104:
time.sleep(10)  # Increase from 3 to 10 seconds
```

### Problem: Variant directory already has files
The script checks `if len(existing_response) >= number: continue` to skip re-running. But if the variant directory has files from a previous incomplete run, it may skip generating new variants. Clean the directories:
```bash
rm -rf demo_case/variant/* demo_case/response/*
```

### Problem: Translation fails (Google Translate API)
The `textaugment` Translate class uses Google Translate API. This may fail if:
- No internet connection
- Google has rate-limited your IP
- The text contains non-UTF8 characters

The code handles this gracefully:
```python
try:
    whole_text = t.augment(whole_text)
except Exception as e:
    print(e)
    whole_text = whole_text  # Keep original if translation fails
```

### Problem: CUDA out of memory (image experiments)
Reduce the batch size or use a smaller GPU. The `low_resource: True` setting in `minigpt4_eval.yaml` enables 8-bit quantization, which reduces memory usage significantly.

### Problem: MiniGPT-4 not found
```
ModuleNotFoundError: No module named 'minigpt4'
```
Edit `minigpt_utils.py` line 11 to point to your MiniGPT-4 installation:
```python
sys.path.append('/full/path/to/MiniGPT-4')
```

---

## CUDA Device Selection

Both scripts have a hardcoded GPU selection:
```python
# In both main_txt.py and main_img.py, line 3:
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
```

Change `"3"` to your GPU index (e.g., `"0"` for the first GPU):
```python
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
```

---

## Quick Reference: Command-Line Arguments

| Argument | Script | Default | Range/Options | Description |
|----------|--------|---------|---------------|-------------|
| `--mutator` | Both | `PL` | Text: `RR,RI,TR,TI,RD,SR,PI,TL,PL` / Image: `HF,VF,RR,CR,RM,RS,GR,BL,CJ,RP,PL` | Which mutator to use |
| `--serial_num` | Both | `9521` (txt) / `287` (img) | Text: 0-9999 / Image: 0-999 | Dataset index to test |
| `--path` | Both | `../dataset/text/dataset.pkl` | Any valid path | Dataset file/directory |
| `--variant_save_dir` | Both | `./demo_case/variant` | Any directory | Where to save variants |
| `--response_save_dir` | Both | `./demo_case/response` | Any directory | Where to save responses |
| `--number` | Both | `8` | Any positive int | Number of variants (LLM budget) |
| `--threshold` | Both | `0.02` (txt) / `0.025` (img) | Float 0-1 | Divergence threshold |
