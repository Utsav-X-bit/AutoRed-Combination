# GEMINI.md: AutoRed Scripts

## 1. Overview

The `scripts` directory contains the necessary Python scripts and Jupyter notebooks for data preparation and model training. It is divided into two main subdirectories: `pi` for prompt injection tasks and `training` for the actual model training.

## 2. Directory Structure

```
scripts/
├── pi/                  # Scripts and notebooks for prompt injection data and models
│   ├── create_data.ipynb
│   ├── fsm.ipynb
│   ├── pi_dacreate_password_extractor_dataset.ipynb
│   ├── train_decision_policy.ipynb
│   ├── view_datapool.ipynb
│   ├── view_decision_policy.ipynb
│   ├── inference/
│   └── pi_data/
└── training/            # Scripts and configurations for training the models
    ├── train_text_generation.py
    └── task_configs/
```

## 3. Prompt Injection (`pi`)

The `pi` directory is dedicated to the **Malicious Prompt Generator** and the **Stop Point Identifier**.

### 3.1. Data Generation

The `pi_data` directory contains the datasets for training the prompt injection models. The notebooks `create_data.ipynb` and `pi_dacreate_password_extractor_dataset.ipynb` are used to generate these datasets.

*   **`pi_data/pi_ext_data`**: Contains data for the supervised fine-tuning of the sensitive information extractor.
*   **`pi_data/pi_gen_data`**: Contains data for training the malicious prompt generator.

### 3.2. Model Training and Inspection

*   **`train_decision_policy.ipynb`**: This notebook is used to train the **Stop Point Identifier**. It trains a binary classifier to decide whether to attack or extract at a given stage.

*   **`view_datapool.ipynb`**: A utility notebook to inspect the generated datasets.

*   **`view_decision_policy.ipynb`**: A notebook to evaluate and visualize the performance of the trained **Stop Point Identifier**.

*   **`inference/view_gen_policy.ipynb`**: This notebook is for viewing the outputs of the trained **Malicious Prompt Generator**.

## 4. Training (`training`)

The `training` directory is where the core training of the **Malicious Prompt Generator** takes place, using the `rl4lms` library.

### 4.1. Training Script

*   **`train_text_generation.py`**: This is the main script for training the language models. It takes a configuration file as input and uses the `rl4lms` library to perform either Supervised Fine-Tuning (SFT) or Reinforcement Learning (RL) training.

### 4.2. Task Configurations

The `task_configs` directory contains YAML files that define the parameters for each training task.

*   **`pi_ext/pi_ext_supervised.yml`**: Configuration for supervised fine-tuning of the **Sensitive Information Extractor**.

*   **`pi_gen/pi_supervised.yml`**: Configuration for supervised fine-tuning of the **Malicious Prompt Generator**.

*   **`pi_gen/pi_ppo.yml`** and **`pi_gen/pi_nlpo.yml`**: Configurations for training the **Malicious Prompt Generator** using PPO and NLPO algorithms, respectively.

### 4.3. How to Run Training

To run a training task, you can execute the `train_text_generation.py` script with the desired configuration file. For example, to run supervised fine-tuning on the prompt generation task:

```bash
python scripts/training/train_text_generation.py --config_path scripts/training/task_configs/pi_gen/pi_supervised.yml
```

Similarly, to run RL training with PPO:

```bash
python scripts/training/train_text_generation.py --config_path scripts/training/task_configs/pi_gen/pi_ppo.yml
```

Make sure to adjust the parameters in the YAML files (e.g., model paths, batch sizes) to fit your specific setup and requirements.
