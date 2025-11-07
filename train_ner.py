
import json
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer, DataCollatorForTokenClassification
import numpy as np

def train_ner_model(data_path, model_output_path):
    """
    Loads processed IOB data, trains a token classification model, and saves it.
    """
    # 1. Load the processed dataset
    with open(data_path, 'r', encoding='utf-8') as f:
        processed_data = json.load(f)

    # Convert to Hugging Face Dataset object
    dataset = Dataset.from_list(processed_data)

    # 2. Create label mappings (str to int)
    # Get all unique tags from the dataset
    unique_tags = set(tag for record in processed_data for tag in record['ner_tags'])
    label_list = sorted(list(unique_tags))
    label2id = {label: i for i, label in enumerate(label_list)}
    id2label = {i: label for i, label in enumerate(label_list)}

    print(f"Found {len(label_list)} unique NER tags: {label_list}")

    # 3. Load pre-trained tokenizer
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    # 4. Preprocessing function to align tokens and labels
    def tokenize_and_align_labels(examples):
        # This function handles the fact that a single word can be split into multiple tokens
        tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)
        
        labels = []
        for i, label in enumerate(examples["ner_tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100) # Special token, ignore in loss calculation
                elif word_idx != previous_word_idx:
                    label_ids.append(label2id[label[word_idx]]) # First token of a word
                else:
                    # Subsequent tokens of the same word get the same label, but some schemes use -100
                    label_ids.append(-100)
                previous_word_idx = word_idx
            labels.append(label_ids)
        
        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    # Apply the preprocessing
    tokenized_dataset = dataset.map(tokenize_and_align_labels, batched=True)

    # 5. Split into training and evaluation sets
    train_test_split = tokenized_dataset.train_test_split(test_size=0.2)
    train_dataset = train_test_split["train"]
    eval_dataset = train_test_split["test"]

    print(f"\nTraining set size: {len(train_dataset)}")
    print(f"Evaluation set size: {len(eval_dataset)}")

    # 6. Load the pre-trained model
    model = AutoModelForTokenClassification.from_pretrained(
        "distilbert-base-uncased", 
        num_labels=len(label_list),
        id2label=id2label,
        label2id=label2id
    )

    # 7. Define Training Arguments
    training_args = TrainingArguments(
        output_dir=model_output_path,
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        save_strategy="epoch",
        load_best_model_at_end=True,
    )

    # Data collator handles dynamic padding
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    # 8. Instantiate the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # 9. Train the model
    print("\nStarting model training...")
    trainer.train()

    # 10. Save the final model
    print("\nTraining complete. Saving the best model...")
    trainer.save_model(model_output_path)
    tokenizer.save_pretrained(model_output_path)
    print(f"Model saved to {model_output_path}")

if __name__ == "__main__":
    JSON_DATA_PATH = "C:\\BrokerMsg\\training_data.json"
    MODEL_OUTPUT_PATH = "C:\\BrokerMsg\\broker-ner-model"
    train_ner_model(JSON_DATA_PATH, MODEL_OUTPUT_PATH)
