import pandas as pd
import numpy as np
import re
from unidecode import unidecode
from setfit import SetFitModel, Trainer, TrainingArguments
from datasets import Dataset
from sklearn.model_selection import train_test_split
import linktransformer as lt
from sentence_transformers import SentenceTransformer
from Levenshtein import distance as levenshtein_distance
from jarowinkler import jarowinkler_similarity
from thefuzz import fuzz
from supervised.automl import AutoML
from sklearn.metrics import accuracy_score, roc_auc_score
import logging
import sys
import os
import warnings
from datetime import datetime
from tqdm import tqdm


for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

# Set up logging
logging.basicConfig(filename='litigation_log.txt', 
                    filemode='a',
                    level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logging.info("Logging initialized successfully")
print(f"Log file created at: {os.path.abspath('litigation_log.txt')}")



console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
logging.getLogger().addHandler(console_handler)


warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  

# Constants
REPLACEMENTS = {r'\bET AL\b': "", r'\bETAL\b': ""}

def replace_all(text, dic):
    for i, j in dic.items():
        text = re.sub(i, j, text)
    return text

def clean_litigation_data(file_path):
    logging.info("Starting litigation data cleaning")
    df = pd.read_csv(file_path, sep="\t", encoding = 'iso-8859-1', low_memory = False)
    df = df[df['NOS'] == 820]   # copyright
    df['plt'] = df['PLT'].str.strip()
    df['def'] = df['DEF'].str.strip()
    df['tapeyear'] = df['TAPEYEAR']

    result_plt = df.groupby(['plt']).agg(
        n=('plt', 'size'),
        min_tapeyear=('tapeyear', 'min'),
        max_tapeyear=('tapeyear', 'max'),
    ).reset_index().rename(columns={"plt": "company"})

    result_def = df.groupby(['def']).agg(
        n=('def', 'size'),
        min_tapeyear=('tapeyear', 'min'),
        max_tapeyear=('tapeyear', 'max'),
    ).reset_index().rename(columns={"def": "company"})

    result = pd.concat([result_plt, result_def], axis=0)
    result['words'] = result['company'].str.count(' ') + 1
    result = result.loc[result['company']!='',:]
    result['clean_company'] = result['company'].apply(lambda x: replace_all(x, REPLACEMENTS))
    result.drop_duplicates(inplace=True)
    result.reset_index(drop=True, inplace=True)
    result['Id'] = result.index + 1
    
    logging.info(f"Litigation data cleaning complete. Rows: {len(result)}")
    return result



def load_setfit_training_data():
    """Load and preprocess the SetFit training data for litigation."""
    logging.info("Loading SetFit training data for litigation")
    df = pd.read_csv('training-data/litigation_classifier_training_data.csv')
    df = df[['company', 'label']]
    df['company'] = df['company'].str.lower()
    result = df.dropna().drop_duplicates()
    logging.info(f"SetFit training data loaded. Rows: {len(result)}")
    return result

def train_setfit_model(df):
    """Train a SetFit model for company type classification."""
    logging.info("Starting SetFit model training")
    unique_labels = df['label'].unique().tolist()
    logging.info(f"Unique labels: {unique_labels}")


    train_val, test = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=423)

    train, val = train_test_split(train_val, test_size=0.25, stratify=train_val['label'], random_state=423)

    logging.info(f"Data split: train ({len(train)}), validation ({len(val)}), test ({len(test)})")

    train_dataset = Dataset.from_pandas(train)
    val_dataset = Dataset.from_pandas(val)

    model = SetFitModel.from_pretrained(
        "BAAI/bge-large-en-v1.5",
        labels=unique_labels,
    )

    args = TrainingArguments(
        batch_size=16,
        num_epochs=1,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        column_mapping={"company": "text", "label": "label"}
    )

    trainer.train()
    
    # Evaluate on validation set
    val_metrics = trainer.evaluate(val_dataset)
    logging.info(f"SetFit Model Validation Metrics: {val_metrics}")

    # Evaluate on test set
    test_dataset = Dataset.from_pandas(test)
    test_metrics = trainer.evaluate(test_dataset)
    logging.info(f"SetFit Model Test Metrics: {test_metrics}")

    return model


def predict_in_batches(model, data, batch_size=32):
    all_predictions = []
    total_items = len(data)
    batch_size = min(batch_size, total_items)
    num_batches = (total_items + batch_size - 1) // batch_size
    
    for i in tqdm(range(0, total_items, batch_size), total=num_batches, desc="Predicting labels"):
        batch = data[i:i+batch_size].tolist()
        predictions = model.predict(batch)
        all_predictions.extend(predictions)
    
    return all_predictions

def dedupe_companies(df):
    logging.info("Starting company deduplication")
    df = df.query('label == "organization"')
    df['cluster_input'] = "The company name is " + df['clean_company']

    df_dedup = lt.cluster_rows(df,
                               model='BAAI/bge-large-en-v1.5',
                               on="cluster_input",
                               cluster_type='agglomerative',
                               cluster_params={'threshold': 0.085,
                                               'metric': 'cosine',
                                               "clustering linkage": 'single'})

    max_n_per_cluster = df_dedup.groupby('cluster')['n'].transform(max)
    mask = df_dedup['n'] == max_n_per_cluster
    cluster_to_max_company = df_dedup[mask].groupby('cluster')['clean_company'].first().reset_index()
    df_dedupe = df_dedup.merge(cluster_to_max_company, on='cluster', suffixes=('', '_max'))

    df_dedupe.rename(columns={"Id": "litigation_id", "cluster": "cluster_id", "clean_company_max": "cluster_name"}, inplace=True)
    
    logging.info(f"Deduplication complete. Resulting clusters: {df_dedupe['cluster_id'].nunique()}")
    return df_dedupe

def merge_with_crsp(crsp_df,litigation_df):
    logging.info("Starting merge with CRSP data")
    crsp_df['company_name_merge'] = "The company name is " + crsp_df['conm_clean']
    litigation_df['company_name_merge'] = "The company name is " + litigation_df['cluster_name']

    merged = lt.merge_knn(crsp_df, litigation_df, on="company_name_merge",
                          model='BAAI/bge-large-en-v1.5',
                          k=200,
                          drop_sim_threshold=0.8)
    
    logging.info(f"Merge complete. Resulting rows: {len(merged)}")
    return merged

def preprocess_for_training(df):
    logging.info("Preprocessing data for AutoML training")
    df['conm_x'] = df['conm_clean'].astype('string')
    df['conm_y'] = df['cluster_name'].astype('string')
    df['levenshtein'] = df.apply(lambda x: levenshtein_distance(x.conm_x.lower(), x.conm_y.lower()), axis=1)
    df['jaro'] = df.apply(lambda x: jarowinkler_similarity(x.conm_x.lower(), x.conm_y.lower()), axis=1)
    df['partial_ratio'] = df.apply(lambda x: fuzz.partial_ratio(x.conm_x.lower(), x.conm_y.lower()), axis=1)
    df['token_set_ratio'] = df.apply(lambda x: fuzz.token_set_ratio(x.conm_x.lower(), x.conm_y.lower()), axis=1)
    df['len_x'] = df['conm_x'].apply(len)
    df['nwords_x'] = df['conm_x'].apply(lambda x: len(x.split()))
    df['len_y'] = df['conm_y'].apply(len)
    df['nwords_y'] = df['conm_y'].apply(lambda x: len(x.split()))

    models = {
        'BGE': 'BAAI/bge-large-en-v1.5',
        'EMBER': 'llmrails/ember-v1',
        'LINKTRANSFORMERS': 'dell-research-harvard/lt-wikidata-comp-en'
    }

    for name, model_path in models.items():
        model = SentenceTransformer(model_path)
        embeddings_1 = model.encode(df['conm_x'].tolist(), normalize_embeddings=True)
        embeddings_2 = model.encode(df['conm_y'].tolist(), normalize_embeddings=True)
        similarity = np.sum(embeddings_1 * embeddings_2, axis=1)
        df[f'score{name}'] = similarity.tolist()

    logging.info("Preprocessing for AutoML complete")
    return df

def train_automl_model(df):
    logging.info("Starting AutoML model training")
    X = df[['scoreBGE', 'scoreEMBER', 'scoreLINKTRANSFORMERS', 'jaro', 'levenshtein', 'partial_ratio', 'token_set_ratio', 'len_x', 'len_y', 'nwords_x', 'nwords_y']]
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=0)

    automl = AutoML(mode="Compete", eval_metric="auc")
    automl.fit(X_train, y_train)

    y_pred = automl.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred)
    logging.info(f'AutoML model accuracy score: {accuracy:.4f}')
    logging.info(f'AutoML model AUC: {auc:.4f}')

    return automl

def main():
    logging.info("Starting litigation data processing")

    # Step 1: Clean litigation data
    litigation_data = clean_litigation_data("raw-data/cv88on.txt")

    # Step 2: Train SetFit model and label companies
    # setfit_training_data = load_setfit_training_data()
    setfit_training_data = pd.read_csv('training-data/litigation_setfit_training.csv')
    setfit_model = train_setfit_model(setfit_training_data)
    
    logging.info("Starting SetFit model predictions")

    litigation_data['label'] = setfit_model.predict(litigation_data['company'].astype('string').tolist())
    logging.info("SetFit model predictions complete")

    # Step 3: Deduplicate companies
    litigation_data = litigation_data[litigation_data['label'] == 'organization'] # limit linking to organizations
    deduped_litigation = dedupe_companies(litigation_data)
    deduped_litigation.to_csv("deduped_litigation.csv", index = False)

    # Step 4: Merge with CRSP data
    crsp_df = pd.read_csv("raw-data/name_permno_gvkey_CLEANED.csv")
    unique_clusters = deduped_litigation[['cluster_name', 'cluster_id']].drop_duplicates()
    merged_df = merge_with_crsp(crsp_df, unique_clusters)

    # Step 5: Train AutoML model
    training_data = pd.read_csv('training-data/litigation_classifier_training_data.csv')
    training_data = preprocess_for_training(training_data)
    automl_model = train_automl_model(training_data)

    # Apply AutoML model to merged data
    merged_df = preprocess_for_training(merged_df)
    X_data = merged_df[['scoreBGE', 'scoreEMBER', 'scoreLINKTRANSFORMERS', 'jaro', 'levenshtein', 'partial_ratio', 'token_set_ratio', 'len_x', 'len_y', 'nwords_x', 'nwords_y']]
    merged_df['prediction'] = automl_model.predict(X_data)
    merged_df['probability'] = automl_model.predict_proba(X_data)[:, 1]
    logging.info("AutoML model predictions applied to merged data")


    # Merged in all claimants
    merged_df = merged_df.merge(deduped_litigation, 
                          how='left',
                          left_on='cluster_name',
                          right_on='cluster_name')

    # Save final results
    merged_df.to_csv("predictions_litigation.csv", index=False)
    logging.info("Final results saved to litigation_predictions_usco.csv")

    logging.info("Litigation data processing complete")

if __name__ == "__main__":
    main()
