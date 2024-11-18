import os
import warnings
import re
import pandas as pd
import numpy as np
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
from tqdm import tqdm


for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

# Set up logging
logging.basicConfig(filename='usco_log.txt', 
                    filemode='a',
                    level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logging.info("Logging initialized successfully")
print(f"Log file created at: {os.path.abspath('usco_log.txt')}")


# Add a stream handler to also print to console
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
logging.getLogger().addHandler(console_handler)

# Suppress warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logging

# Constants and configurations
SUFFIXES = ['inc.', 'llc', 'ltd.', 'inc', 'llc.', 'lp', 'sa', 'llp', 's.a.', 'gmbh', 'l.l.c.', 'corp', 'c.', 'p.c.', 'b.v.', 'ltd', '(us)', '(uk)']
REPLACEMENTS = {
    r'\bibm\b': "international business machines",
    r'\batv\b': "associated television",
    r'\bemi\b': "electric and musical industries",
    r'\bumg\b': "universal music group",
    r'\babc\b': "american broadcasting company",
    r'\bcbs\b': "columbia broadcasting system",
    r'\bbmi\b': "broadcast music, inc.",
    r'\bwb\b': "warner brothers",
    r'\bmls\b': "multiple listing service",
    r'\bieee\b': "institute of electrical and electronics engineers",
    r'\bespn\b': "entertainment and sports programming network",
    r'\bmgm\b': "metro-goldwyn-mayer",
    r'\brca\b': "radio corporation of america"
}

def preprocess(column):
    """Clean data using Unidecode and Regex."""
    column = unidecode(column)
    column = re.sub('  +', ' ', column)
    column = re.sub('\n', ' ', column)
    return column.strip()

def replace_all(text, dic):
    """Replace all occurrences of the keys in dic with their corresponding values in text."""
    for i, j in dic.items():
        text = re.sub(i, j, text)
    return text

def remove_suffixes(x):
    """Remove common company suffixes from the end of a string."""
    x_list = x.split()
    return ' '.join([word for index, word in enumerate(x_list) if not (word.lower() in SUFFIXES and index == len(x_list) - 1)])

def clean_and_standardize(df):
    """Clean and standardize company names in the dataframe."""
    logging.info("Starting data cleaning and standardization")
    df['corp'] = df['claimant_name']
    df['corp_id'] = df.groupby(['corp']).ngroup()
    df['conm'] = df['corp'].apply(preprocess)
    df['conm_std'] = df['conm'].apply(lambda x: replace_all(x, REPLACEMENTS))
    df['conm_std'] = df['conm_std'].apply(remove_suffixes)
    df['conm_std'] = df['conm_std'].str.strip().apply(lambda x: x.rstrip(',')).str.strip()
    df['conm_std_id'] = df.groupby(['conm_std']).ngroup()
    result = df.dropna(subset=['conm_std']).loc[df['conm_std'] != '', :]
    logging.info(f"Data cleaning complete. Rows remaining: {len(result)}")
    return result

def train_setfit_model(df):
    """Train a SetFit model for company type classification."""
    logging.info("Starting SetFit model training")
    unique_labels = df['label'].unique().tolist()
    logging.info(f"Unique labels: {unique_labels}")

    # First, split off the test set
    train_val, test = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=423)

    # Then split the remaining data into train and validation sets
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
        column_mapping={"conm": "text", "label": "label"}
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
    """
    Predict labels for data in param.
    
    :batches model: SetFitModel to use for predictions
    :param data: list or Series of texts to predict
    :param batch_size: size of batches to use for prediction
    :return: list of predicted labels
    """
    all_predictions = []
    total_items = len(data)
    
    # Adjust batch_size if it's larger than the total number of items
    batch_size = min(batch_size, total_items)
    
    # Calculate number of batches
    num_batches = (total_items + batch_size - 1) // batch_size
    
    for i in tqdm(range(0, total_items, batch_size), total=num_batches, desc="Predicting labels"):
        batch = data[i:i+batch_size].tolist()
        predictions = model.predict(batch)
        all_predictions.extend(predictions)
    
    return all_predictions

def dedupe_companies(df):
    """Deduplicate company names using LinkTransformer."""
    logging.info("Starting company deduplication")
    df['pre_conm'] = 'The company name is ' + df['conm_std'].astype(str)
    df_dedupe = lt.cluster_rows(df,
                                model='BAAI/bge-large-en-v1.5',
                                on="pre_conm",
                                cluster_type='SLINK',
                                cluster_params={'min cluster size': 2,
                                                'threshold': 0.05,
                                                "metric": "cosine"})

    df_dedupe.sort_values(by=['cluster'], inplace=True)
    n_clusters = df_dedupe['cluster'].max()
    df_dedupe['cluster'] = df_dedupe.apply(lambda row: row.name + n_clusters if row['cluster'] == -1 else row['cluster'], axis=1)

    max_count_per_cluster = df_dedupe.groupby('cluster')['count'].transform(max)
    mask = df_dedupe['count'] == max_count_per_cluster
    cluster_to_max_company = df_dedupe[mask].groupby('cluster')['conm_std'].first().reset_index()
    df_dedupe = df_dedupe.merge(cluster_to_max_company, on='cluster', suffixes=('', '_max'))

    df_dedupe.rename(columns={"conm_std_id": "reg_id", "cluster": "cluster_id", "conm_std_max": "cluster_name"}, inplace=True)
    logging.info(f"Deduplication complete. Resulting clusters: {df_dedupe['cluster_id'].nunique()}")
    return df_dedupe

def merge_with_crsp(crsp_df, usco_df):
    """Merge USCO data with CRSP data using LinkTransformer."""
    logging.info("Starting merge with CRSP data")
    crsp_df['company_name_merge'] = "The company name is " + crsp_df['conm_clean']
    usco_df['company_name_merge'] = "The company name is " + usco_df['cluster_name']

    merged = lt.merge_knn(crsp_df, usco_df, on="company_name_merge",
                          model='BAAI/bge-large-en-v1.5',
                          k=200,
                          drop_sim_threshold=0.8)
    logging.info(f"Merge complete. Resulting rows: {len(merged)}")
    return merged

def preprocess_for_training(df):
    """Preprocess data for training the AutoML model."""
    logging.info("Preprocessing data for AutoML training")
    df['conm_x'] = df['conm_clean'].astype('string')
    df['conm_y'] = df['cluster_name'].astype('string')
    df['levenshtein_dist'] = df.apply(lambda x: levenshtein_distance(x.conm_x.lower(), x.conm_y.lower()), axis=1)
    df['jaro_dist'] = df.apply(lambda x: jarowinkler_similarity(x.conm_x.lower(), x.conm_y.lower()), axis=1)
    df['partial_ratio_dist'] = df.apply(lambda x: fuzz.partial_ratio(x.conm_x.lower(), x.conm_y.lower()), axis=1)
    df['token_set_ratio_dist'] = df.apply(lambda x: fuzz.token_set_ratio(x.conm_x.lower(), x.conm_y.lower()), axis=1)
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
    """Train an AutoML model for company matching."""
    logging.info("Starting AutoML model training")
    X = df[['scoreBGE', 'scoreEMBER', 'scoreLINKTRANSFORMERS', 'jaro_dist', 'levenshtein_dist', 'partial_ratio_dist', 'token_set_ratio_dist', 'len_x', 'len_y', 'nwords_x', 'nwords_y']]
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
    logging.info("Starting USCO data processing")

    # Step 1: Clean and standardize USCO data. USCO data version: 02.20.2024.
    usco_df = pd.read_csv('claimant_name_counts.csv')
    usco_df = clean_and_standardize(usco_df)

    # Step 2: Train SetFit model and label companies
    setfit_training_data = pd.read_csv('training-data/registration_training_setfit.csv')
    setfit_model = train_setfit_model(setfit_training_data)
    
    logging.info("Starting SetFit model predictions")

    usco_df['label'] = setfit_model.predict(usco_df['conm'].astype('string').tolist())
    logging.info("SetFit model predictions complete")

    # Step 3: Deduplicate companies
    usco_df = usco_df[usco_df['label'] == 'business'] # limit linking to businesses
    usco_df = dedupe_companies(usco_df[['claimant_name','conm_std', 'conm_std_id', 'count', 'label']])
    usco_df.to_csv("deduped_clustered_usco.csv", index = False)

    # Step 4: Merge with CRSP data
    crsp_df = pd.read_csv("raw-data/name_permno_gvkey_CLEANED.csv")
    unique_clusters = usco_df[['cluster_name', 'cluster_id']].drop_duplicates()
    merged_df = merge_with_crsp(crsp_df, unique_clusters)

    # Step 5: Train AutoML model
    training_data = pd.read_csv('training-data/registration_classifier_training_data.csv')
    training_data = preprocess_for_training(training_data)
    automl_model = train_automl_model(training_data)

    # Apply AutoML model to merged data
    merged_df = preprocess_for_training(merged_df)
    X_data = merged_df[['scoreBGE', 'scoreEMBER', 'scoreLINKTRANSFORMERS', 'jaro_dist', 'levenshtein_dist', 'partial_ratio_dist', 'token_set_ratio_dist', 'len_x', 'len_y', 'nwords_x', 'nwords_y']]
    merged_df['prediction'] = automl_model.predict(X_data)
    merged_df['probability'] = automl_model.predict_proba(X_data)[:, 1]
    logging.info("AutoML model predictions applied to merged data")

    # Merged in all claimants
    merged_df = merged_df.merge(usco_df, 
                          how='left',
                          left_on='cluster_name',
                          right_on='cluster_name')

    # Save final results
    merged_df.to_csv("predictions_usco.csv", index=False)
    logging.info("Final results saved to predictions_usco.csv")

    logging.info("USCO data processing complete")

if __name__ == "__main__":
    main()
