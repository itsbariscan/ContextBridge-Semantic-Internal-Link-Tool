# ContextBridge-Semantic-Internal-Link-Tool

ContextBridge-Semantic-Internal-Link-Tool is an advanced Python script designed to enhance website structure and user experience by identifying and suggesting intelligent internal linking opportunities. By leveraging the power of natural language processing and machine learning, this tool analyzes web page content, semantics, and context to recommend highly relevant internal links.

## Features

- Utilizes GPT-3.5 for accurate product classification into categories and subcategories
- Employs OpenAI's text-embedding-ada-002 model to generate high-quality content embeddings
- Calculates multi-dimensional similarity scores using cosine similarity and TF-IDF
- Identifies potential internal linking opportunities based on content relevance, semantic similarity, and category coherence
- Generates context-aware anchor texts for suggested links using GPT-3.5
- Outputs comprehensive results to an Excel file for easy analysis and implementation

## Prerequisites

- Python 3.7+
- OpenAI API key

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/ContextBridge-Semantic-Internal-Link-Tool.git
   cd ContextBridge-Semantic-Internal-Link-Tool
   ```

2. Install the required packages:
   ```bash
   pip install pandas numpy openai scikit-learn nltk openpyxl
   ```

3. Set up your OpenAI API key:
   - Replace `'your-api-key-here'` in the script with your actual OpenAI API key.

## Usage

1. Prepare your input data:
   - Create an Excel file named `page_metadata.xlsx` with columns 'url' and 'h1' containing your page URLs and H1 titles.

2. Run the script:
   ```bash
   python script.py
   ```

3. The script will process your data and generate an Excel file named `internal_linking_results.xlsx` with the suggested internal linking opportunities.

## How It Works

1. The script loads the page data from the Excel file.
2. It uses GPT-3.5 to classify each product into a main category and subcategory.
3. Embeddings are generated for each page's content using OpenAI's embedding model.
4. The script calculates similarities between pages using these embeddings and text-based similarity metrics.
5. It then identifies potential linking opportunities based on content similarity, embedding similarity, and category relevance.
6. For each potential link, it generates an appropriate anchor text using GPT-3.5.
7. Finally, it saves all the results, including source and target URLs, categories, similarity scores, and suggested anchor texts, to an Excel file.

## Note

This script makes multiple API calls to OpenAI, which may incur costs. Make sure you understand the pricing and have appropriate usage limits set up in your OpenAI account.

## Contributing

Contributions, issues, and feature requests are welcome. Feel free to check [issues page](https://github.com/yourusername/ContextBridge-Semantic-Internal-Link-Tool/issues) if you want to contribute.

## License

[MIT](https://choosealicense.com/licenses/mit/)
