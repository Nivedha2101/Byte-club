import os
import pandas as pd
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
from datasets import load_dataset, Dataset, concatenate_datasets
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch
import numpy as np
import evaluate


def pdf_to_wordlevel_df(pdf_path):
    model = ocr_predictor(
        det_arch='linknet_resnet18',
        reco_arch='sar_resnet31',
        pretrained=True,
        assume_straight_pages=False,
        straighten_pages=True,
        detect_orientation=True
    )
    doc = DocumentFile.from_pdf(pdf_path)
    result = model(doc)

    records = []
    for page_num, page in enumerate(result.pages, start=1):
        for block in page.blocks:
            for line in block.lines:
                for word in line.words:
                    x0, y0 = word.geometry[0]
                    x1, y1 = word.geometry[1]
                    font_size = y1 - y0
                    records.append({
                        "page": page_num,
                        "text": word.value,
                        "bbox_x0": x0,
                        "bbox_y0": y0,
                        "bbox_x1": x1,
                        "bbox_y1": y1,
                        "confidence": word.confidence,
                        "font_size": font_size,
                        "label": ""   # empty, to be predicted during inference
                    })
    return pd.DataFrame(records)


def group_words_to_sentences(df, labels_to_group=None):
    """
    Group consecutive words sharing the same label on the same page
    into sentences. Returns grouped sentence-level DataFrame.
    """
    if labels_to_group is None:
        labels_to_group = {'title', 'H1', 'H2', 'H3'}

    grouped_sentences = []
    current_page = None
    current_label = None
    current_words = []

    # Sort by page and bbox_x0 (horizontal reading order)
    for idx, row in df.sort_values(by=['page', 'bbox_x0']).iterrows():
        page = row['page']
        label = row['label'].strip() if pd.notna(row['label']) else ''
        text = str(row['text']).strip()

        # Only group target labels, flush otherwise
        if label not in labels_to_group:
            if current_words and current_label is not None:
                grouped_sentences.append({
                    'page': current_page,
                    'label': current_label,
                    'text': ' '.join(current_words)
                })
            current_words = []
            current_label = None
            current_page = None
            continue

        # Group if same page and label; else flush and start new
        if (page != current_page) or (label != current_label):
            if current_words:
                grouped_sentences.append({
                    'page': current_page,
                    'label': current_label,
                    'text': ' '.join(current_words)
                })
            current_words = [text]
            current_label = label
            current_page = page
        else:
            current_words.append(text)

    if current_words:
        grouped_sentences.append({
            'page': current_page,
            'label': current_label,
            'text': ' '.join(current_words)
        })

    return pd.DataFrame(grouped_sentences)


def load_and_combine_csv_datasets(csv_files):
    """
    Load multiple CSV files and combine into one Hugging Face Dataset.
    Each csv should have at least 'text' and 'label' columns.
    """
    datasets = []
    for f in csv_files:
        df = pd.read_csv(f)
        # Remove rows with empty or missing label
        df['label'] = df['label'].fillna('').astype(str).str.strip()
        df = df[df['label'] != '']

        # Optionally group words to sentences if word-level
        # but assuming CSVs are already sentence-level labeled here
        ds = Dataset.from_pandas(df.reset_index(drop=True))
        datasets.append(ds)

    combined = concatenate_datasets(datasets)
    return combined


def train_classifier(dataset, model_name='distilbert-base-uncased', output_dir='./bert-best-model'):
    """
    Fine-tune DistilBERT model for sequence classification on input dataset.
    """

    # Extract label list and mappings
    label_list = sorted(set(dataset['label']))
    label2id = {l: i for i, l in enumerate(label_list)}
    id2label = {i: l for l, i in label2id.items()}

    # Encode labels
    def encode_labels(example):
        example['label'] = label2id.get(example['label'], -1)
        return example

    dataset = dataset.map(encode_labels)
    dataset = dataset.filter(lambda x: x['label'] != -1)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_fn(examples):
        return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=128)

    tokenized_ds = dataset.map(tokenize_fn, batched=True)

    # Split
    if 'test' not in tokenized_ds:
        tokenized_ds = tokenized_ds.train_test_split(test_size=0.1)

    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(label_list),
        id2label=id2label,
        label2id=label2id,
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy='epoch',
        save_strategy='epoch',
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model='accuracy',
        report_to=[]  # disable wandb if undesired
    )

    metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return metric.compute(predictions=preds, references=labels)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds['train'],
        eval_dataset=tokenized_ds['test'],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.train()

    return model, tokenizer


def predict_structure_labels_for_pdf(pdf_path, model, tokenizer, labels_to_group={'title', 'H1', 'H2', 'H3'}):
    """
    OCR -> group words to sentences -> classify each sentence.
    Returns list of dicts with page, text, label.
    """
    df_words = pdf_to_wordlevel_df(pdf_path)
    df_words['label'] = ''  # no labels during prediction

    # Group words on page into sentences (max 20 words heuristic)
    sentences = []
    current_page = None
    current_words = []
    max_words_per_sentence = 20

    for idx, row in df_words.sort_values(['page', 'bbox_x0']).iterrows():
        page = row['page']
        text = str(row['text']).strip()

        if page != current_page or len(current_words) >= max_words_per_sentence:
            if current_words:
                sentences.append({'page': current_page, 'text': ' '.join(current_words)})
            current_words = [text]
            current_page = page
        else:
            current_words.append(text)

    if current_words:
        sentences.append({'page': current_page, 'text': ' '.join(current_words)})

    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    predicted_output = []
    for sent in sentences:
        inputs = tokenizer(
            sent['text'], return_tensors='pt', padding='max_length', truncation=True, max_length=128
        ).to(device)

        with torch.no_grad():
            logits = model(**inputs).logits
            pred_id = logits.argmax(-1).item()
            pred_label = model.config.id2label.get(pred_id, 'UNKNOWN')

        predicted_output.append({
            'page': sent['page'],
            'text': sent['text'],
            'label': pred_label
        })

    return predicted_output


def assemble_outline_json(predicted_output):
    """
    Converts classification output to final JSON format.
    """
    title_texts = [x['text'] for x in predicted_output if x['label'].lower() == 'title']
    outline_items = [
        {
            "level": x['label'],
            "text": x['text'],
            "page": x['page']
        }
        for x in predicted_output if x['label'].upper() in {'H1', 'H2', 'H3'}
    ]
    return {
        "title": " ".join(title_texts).strip() if title_texts else "",
        "outline": outline_items
    }


# === Example Usage ===

# 1. List your CSV and PDF files (upload them to Colab and use paths here)
csv_training_files = ['1.csv', '2.csv', '3.csv','4.csv','file01.csv','file02.csv','file03.csv' ,'file04.csv']               # Add all your CSV files here
pdf_test_files = ['1.pdf', '2.pdf', '3.pdf','4.pdf','file01.pdf','file02.pdf','file03.pdf' ,'file04.pdf']         # And PDFs to predict on

# 2. Combine all datasets for training
combined_dataset = load_and_combine_csv_datasets(csv_training_files)

# 3. Train model
model, tokenizer = train_classifier(combined_dataset)

# 4. Run prediction and output JSON for each PDF
for pdf_path in pdf_test_files:
    print(f"\nProcessing {pdf_path}")
    predicted = predict_structure_labels_for_pdf(pdf_path, model, tokenizer)
    final_json = assemble_outline_json(predicted)
    print(final_json)