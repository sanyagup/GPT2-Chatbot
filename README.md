# GPT2-Chatbot

# Fine-Tuned GPT-2 for Medical and Conversational Data

This repository contains code for fine-tuning the GPT-2 model on a combined dataset of conversational and medical text. The project was originally developed during my internship, where I worked on a GPT-2 based chatbot with a focus on medical applications. *Note: Due to privacy reasons, the original medical data is not included in this repository.*

## Overview

The goal of this project is to create a GPT-2 model that can generate coherent responses to prompts incorporating both general conversation and medical context. Key features include:
- **Fine-tuning GPT-2:** Utilizing Hugging Face's Transformers library.
- **Data Labeling:** Preparing and labeling text data with custom tags.
- **Response Generation:** A function to generate responses based on input prompts.

## Repository Structure

- `app.py`  
  Main script for loading data, fine-tuning the GPT-2 model, and generating responses.

- `requirements.txt`  
  Lists all the Python dependencies required for this project.

- `README.md`  
  This file.

*Note: The original `combined_medical_data.txt` file is not provided due to privacy restrictions. You can substitute this with your own appropriately anonymized data if you wish to replicate the project.*

## Setup and Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/sanyagup/GPT2-Chatbot.git
   cd GPT2-Chatbot
   ```

2. **Create and Activate a Virtual Environment (Optional but recommended):**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

## Data Preparation

- The function `load_and_label_data` reads in your text file and labels the content with tags (e.g., `<background>`, `<conversation>`, `<medical>`).
- If you have separate files for conversational and medical data, refer to the alternate combining function provided in the code.
- **Important:** The original medical data is excluded for privacy reasons. Please replace or augment with your own data if needed.

## Training the Model

The training script uses Hugging Face's `Trainer` API for fine-tuning. Key training settings include:
- **Epochs:** 3
- **Batch Size:** 4 per device
- **Mixed Precision:** Enabled (`fp16=True`) for faster training on supported hardware.

To start training, run:

```bash
python app.py
```

The fine-tuned model and tokenizer will be saved in the directory `./fine-tuned-gpt2-medical-enhanced`.

## Generating Responses

After fine-tuning, you can generate responses using the `generate_response` function. For example:

```python
prompt = "<medical> What are the common symptoms of patients with a cholesterol level of 250?"
response = generate_response(prompt, model, tokenizer)
print(response)
```

This function tokenizes the prompt and generates a response using the fine-tuned model.

## Recreating the Project

This repository serves as a minimal recreation of my previous work. The original version involved fine-tuning on private medical data; therefore, this version uses a placeholder approach for data handling. Adjust the data preparation functions as needed to work with your own dataset.

## License

*Insert license information here (if applicable).*

## Disclaimer

This project is for educational purposes only. The fine-tuned model uses publicly available GPT-2 weights and custom data; however, some data or configurations may be subject to privacy or proprietary restrictions. Use responsibly.

---

Feel free to modify this README to better reflect your project specifics. Let me know if you need any further changes!