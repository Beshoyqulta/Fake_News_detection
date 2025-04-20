# 1. **`data.csv`** (Dataset)

- **Purpose**: Contains a list of websites with their trustworthiness labels.
- **Columns**:
  - `source`: Name of the website.
  - `domain`: Domain of the website.
  - `field`: The category of the website (e.g., General News, Science and Technology).
  - `trusted`: Boolean indicating whether the website is trusted (`True`) or not (`False`).

---

# 2. **`create_model.ipynb`** (Model Training)

- **Purpose**: Prepares the dataset, trains the model, and evaluates its performance.
- **Steps**:
  1. **Data Preparation**:
     - Reads `data.csv`.
     - Cleans the data by keeping only the `domain` and `trusted` columns.
     - Converts `True`/`False` labels to `1`/`0`.
     - Splits the data into training (80%) and validation (20%) sets.
  2. **Tokenization**:
     - Uses the `BertTokenizer` to tokenize domain names.
  3. **Dataset Creation**:
     - Creates PyTorch datasets for training and validation.
  4. **Model Training**:
     - Fine-tunes a `BertForSequenceClassification` model on the dataset.
     - Saves the trained model in the `results/` directory.

---

# 3. **`model_usage.py`** (Model Inference)

- **Purpose**: Uses the trained model to classify new websites as trusted or untrusted.
- **Key Functions**:
  1. **`retrieve_domains(query: str)`**:
     - Searches the web using DuckDuckGo for websites related to a query.
     - Extracts and returns the domains of the search results.
  2. **`classify_domains(domains: list)`**:
     - Uses the trained model to classify a list of domains.
     - Returns a list of tuples with the domain and its trustworthiness (`True`/`False`).
- **Example Usage**:
  - Searches for websites related to "cristiano ronaldo" and classifies them:
    ```python
    print(classify_domains(retrieve_domains('cristiano ronaldo')))
    ```
  - Example Output:
    ```
    [('ronaldo.com', True), ('newsoccer.org', False)]
    ```

---

# 4. **`results/`** (Model Checkpoints)

- **Purpose**: Stores the trained model and related files.
- **Contents**:
  - [`results/`](https://drive.google.com/drive/folders/1QTgm4MAO9mORFuK3OOG4iRukxtVlFrPZ?usp=drive_link)
  - `checkpoint-500/`: Directory containing the model checkpoint after training.
    - `config.json`: Model configuration.
    - `model.safetensors`: Model weights.
    - `trainer_state.json`: Training state information.
    - `training_args.bin`: Training arguments.
    ---

## How It Works

### Using the Model

1. **Load the Model**:
   - Load the trained model and tokenizer from the `results/` directory.

2. **Retrieve Domains**:
   - Use the `retrieve_domains` function to search for domains related to a query.

3. **Classify Domains**:
   - Use the `classify_domains` function to classify the retrieved domains as trusted or untrusted.


---

## Example Workflow
**Classify New Domains**:
   - Use `model_usage.py` to classify new domains:
     ```python
     print(classify_domains(retrieve_domains('cristiano ronaldo')))
     ```
