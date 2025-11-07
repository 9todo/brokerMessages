
import pandas as pd
from transformers import AutoTokenizer
import json
import numpy as np

def create_training_data(csv_path, output_path):
    """
    Loads data from a CSV file, processes it to create IOB-tagged training data
    for a token classification model, and saves it to a JSON file.
    """
    # 1. Load pre-trained tokenizer
    # Using a DistilBERT tokenizer as a good baseline.
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    # 2. Load the CSV data
    try:
        df = pd.read_csv(csv_path, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(csv_path, encoding='latin1')
    
    # Replace NaN with None for easier handling
    df = df.replace(np.nan, None, regex=True)

    # 3. Define the mapping from CSV columns to entity labels
    # We handle multiple securities by adding a suffix, but the label is the same.
    # This can be simplified later if needed.
    column_to_entity = {
        'Issuer': 'Issuer',
        'Coupon': 'Coupon',
        'Maturity': 'Maturity',
        'Bid Amount': 'Bid_Amount',
        'Bid': 'Bid_Rate',
        'Offer': 'Offer_Rate',
        'Offer amount': 'Offer_Amount',
        'Issuer.1': 'Issuer',
        'Coupon.1': 'Coupon',
        'Maturity.1': 'Maturity',
        'Bid Amount.1': 'Bid_Amount',
        'Bid.1': 'Bid_Rate',
        'Offer.1': 'Offer_Rate',
        'Offer amount.1': 'Offer_Amount',
        'Issuer.2': 'Issuer',
        'Coupon.2': 'Coupon',
        'Maturity.2': 'Maturity',
        'Bid Amount.2': 'Bid_Amount',
        'Bid.2': 'Bid_Rate',
        'Offer.2': 'Offer_Rate',
        'Offer amount.2': 'Offer_Amount',
        'Issuer.3': 'Issuer',
        'Coupon.3': 'Coupon',
        'Maturity.3': 'Maturity',
        'Bid Amount.3': 'Bid_Amount',
        'Bid.3': 'Bid_Rate',
        'Offer.3': 'Offer_Rate',
        'Offer amount.3': 'Offer_Amount',
    }

    training_data = []
    print("Starting data preparation...")

    # 4. Iterate over each row in the DataFrame
    for index, row in df.iterrows():
        chat_message = row['Chat']
        if not isinstance(chat_message, str):
            continue

        # Tokenize the entire chat message
        tokenized_chat = tokenizer(chat_message, add_special_tokens=False)
        tokens = tokenizer.convert_ids_to_tokens(tokenized_chat['input_ids'])
        iob_tags = ['O'] * len(tokens)

        # Collect all entities for the current row
        entities = []
        for col, entity_name in column_to_entity.items():
            if col in row and row[col] is not None:
                entities.append((str(row[col]), entity_name))
        
        # Sort entities by length, longest first, to handle nested entities better
        entities.sort(key=lambda x: len(x[0]), reverse=True)

        # 5. Tag entities
        for entity_text, entity_name in entities:
            # Tokenize the entity text itself
            tokenized_entity = tokenizer(entity_text, add_special_tokens=False)
            entity_tokens = tokenizer.convert_ids_to_tokens(tokenized_entity['input_ids'])

            if not entity_tokens:
                continue

            # Find the sublist of entity tokens within the main list of tokens
            for i in range(len(tokens) - len(entity_tokens) + 1):
                # Check if the sublist matches and if the spot is not already tagged
                if tokens[i:i+len(entity_tokens)] == entity_tokens and all(tag == 'O' for tag in iob_tags[i:i+len(entity_tokens)]):
                    iob_tags[i] = f'B-{entity_name}'
                    for j in range(1, len(entity_tokens)):
                        iob_tags[i+j] = f'I-{entity_name}'
                    # Break after the first match to avoid tagging duplicates
                    break
        
        record = {
            "tokens": tokens,
            "ner_tags": iob_tags,
            "deal_type": row['Type'] if 'Type' in row and row['Type'] is not None else 'secondary'
        }
        training_data.append(record)

    # 6. Save the processed data
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(training_data, f, indent=4)

    print(f"Successfully created {output_path}")
    print(f"Total records processed: {len(training_data)}")
    if training_data:
        print("\n--- Example of first processed record ---")
        print(json.dumps(training_data[0], indent=2))
        print("------------------------------------")


if __name__ == "__main__":
    # Define paths
    CSV_FILE_PATH = "C:\\BrokerMsg\\MainTrainingData.csv"
    JSON_OUTPUT_PATH = "C:\\BrokerMsg\\training_data.json"
    
    # Run the data creation process
    create_training_data(CSV_FILE_PATH, JSON_OUTPUT_PATH)
