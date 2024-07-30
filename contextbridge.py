import pandas as pd
import numpy as np
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import time
import re
import json
from nltk.util import ngrams
import nltk
from collections import defaultdict

# Download necessary NLTK data
nltk.download('punkt', quiet=True)

# OpenAI API setup
client = OpenAI(api_key='your-api-key-here')

def classify_product(url, h1):
    prompt = f"""
    URL: {url}
    H1: {h1}

    Based on the URL and H1 provided, classify this product into a specific category and subcategory.
    Provide your answer in the following JSON format:
    {{
        "main_category": "main category name",
        "sub_category": "subcategory name",
        "keywords": ["keyword1", "keyword2", "keyword3"]
    }}
    """

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an e-commerce categorization expert."},
                {"role": "user", "content": prompt}
            ]
        )
        content = response.choices[0].message.content.strip()
        
        match = re.search(r'\{.*\}', content, re.DOTALL)
        if match:
            json_str = re.sub(r'\}\s*\}$', '}', match.group(0))
            try:
                return json.loads(json_str)
            except json.JSONDecodeError as e:
                print(f"JSON parsing error: {str(e)}")
                print(f"Faulty JSON: {json_str}")
        else:
            print(f"No JSON format response received. Response: {content}")
        
        return {"main_category": "other", "sub_category": "other", "keywords": []}
    
    except Exception as e:
        print(f"Product classification error: {str(e)}")
        return {"main_category": "other", "sub_category": "other", "keywords": []}

def load_data(file_path):
    data = pd.read_excel(file_path, engine='openpyxl', usecols=['url', 'h1'])
    data = data.dropna().astype(str)
    
    classifications = [classify_product(row['url'], row['h1']) for _, row in data.iterrows()]
    
    data['main_category'] = [c.get('main_category', 'other') for c in classifications]
    data['sub_category'] = [c.get('sub_category', 'other') for c in classifications]
    data['keywords'] = [' '.join(c.get('keywords', [])) for c in classifications]
    
    return data

def generate_adaptive_ngrams(text):
    tokens = nltk.word_tokenize(text.lower())
    if len(tokens) <= 2:
        return [' '.join(tokens)]
    return [' '.join(gram) for gram in ngrams(tokens, 3)] or [' '.join(tokens)]

def generate_embeddings(texts, batch_size=100):
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        adaptive_ngram_texts = [' '.join(generate_adaptive_ngrams(text)) for text in batch]
        try:
            response = client.embeddings.create(
                model="text-embedding-ada-002",
                input=adaptive_ngram_texts
            )
            embeddings.extend([data.embedding for data in response.data])
        except Exception as e:
            print(f"Error creating embedding vectors (batch {i//batch_size + 1}): {str(e)}")
            embeddings.extend([[0] * 1536 for _ in range(len(batch))])
        print(f"Processed {min(i+batch_size, len(texts))} rows...")
        time.sleep(0.1)  # Rate limiting
    return np.array(embeddings)

def calculate_text_similarity(text1, text2):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([text1, text2])
    return cosine_similarity(tfidf_matrix)[0][1]

def calculate_category_similarity_batch(category_pairs, batch_size=50):
    similarities = {}
    for i in range(0, len(category_pairs), batch_size):
        batch = list(category_pairs)[i:i+batch_size]
        prompt = "Calculate the similarity between the following category pairs on a scale of 0 to 1:\n\n"
        prompt += "\n".join(f"{j+1}. Category 1: {cat1}\n   Category 2: {cat2}" for j, (cat1, cat2) in enumerate(batch))
        prompt += "\n\nProvide the numerical scores as a JSON array, e.g., [0.8, 0.3, 0.9, ...]"

        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an e-commerce categorization expert."},
                    {"role": "user", "content": prompt}
                ]
            )
            scores = json.loads(response.choices[0].message.content.strip())
            similarities.update({pair: score for pair, score in zip(batch, scores)})
        except Exception as e:
            print(f"Error calculating category similarity: {str(e)}")
            similarities.update({pair: 0.0 for pair in batch})
    return similarities

def generate_anchor_text(source_h1, target_h1):
    prompt = f"""Source Page H1: {source_h1}
Target Page H1: {target_h1}

Create a natural and relevant anchor text (2-5 words) suitable for linking from the source page to the target page:"""
    
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an assistant that generates relevant anchor texts for internal linking."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error generating anchor text: {str(e)}")
        return ""

def find_linking_opportunities(embeddings, data, embedding_threshold=0.7, text_similarity_threshold=0.3, category_similarity_threshold=0.5, max_links_per_page=3):
    print("Calculating similarities...")
    embedding_similarities = cosine_similarity(embeddings)
    opportunities = defaultdict(list)
    
    print("Identifying linking opportunities...")
    category_pairs = set()
    potential_links = []

    for i in range(len(data)):
        similar_pages = np.argsort(embedding_similarities[i])[::-1][1:21]  # Top 20 similar pages
        
        for j in similar_pages:
            embedding_similarity = embedding_similarities[i][j]
            if embedding_similarity > embedding_threshold:
                text_similarity = calculate_text_similarity(data['h1'].iloc[i] + ' ' + data['keywords'].iloc[i],
                                                            data['h1'].iloc[j] + ' ' + data['keywords'].iloc[j])
                if text_similarity > text_similarity_threshold:
                    cat1, cat2 = data['sub_category'].iloc[i], data['sub_category'].iloc[j]
                    category_pairs.add((cat1, cat2) if cat1 < cat2 else (cat2, cat1))
                    potential_links.append((i, j, embedding_similarity, text_similarity))

        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1} pages...")

    print("Calculating category similarities...")
    category_similarities = calculate_category_similarity_batch(category_pairs)

    print("Finalizing linking opportunities...")
    for i, j, embedding_similarity, text_similarity in potential_links:
        cat1, cat2 = data['sub_category'].iloc[i], data['sub_category'].iloc[j]
        category_similarity = category_similarities.get((cat1, cat2) if cat1 < cat2 else (cat2, cat1), 0.0)
        
        if category_similarity > category_similarity_threshold:
            overall_similarity = (embedding_similarity + text_similarity + category_similarity) / 3
            
            if len(opportunities[data['url'].iloc[i]]) < max_links_per_page:
                opportunities[data['url'].iloc[i]].append({
                    "source_url": data['url'].iloc[i],
                    "target_url": data['url'].iloc[j],
                    "embedding_similarity": embedding_similarity,
                    "text_similarity": text_similarity,
                    "category_similarity": category_similarity,
                    "overall_similarity": overall_similarity,
                    "source_index": i,
                    "target_index": j
                })

    return opportunities

def main():
    try:
        print("\nLoading Excel file and classifying products...")
        data = load_data('page_metadata.xlsx')
        
        if data.empty:
            print("Warning: Dataset is empty. Please check your Excel file.")
            return
        
        print(f"Loaded and classified a total of {len(data)} valid rows.")
        
        print("Creating embedding vectors for H1s and keywords...")
        combined_text = data['h1'] + ' ' + data['keywords']
        embeddings = generate_embeddings(combined_text.tolist())
        
        print("Finding linking opportunities...")
        opportunities = find_linking_opportunities(embeddings, data)
        
        print("\nGenerating anchor texts...")
        results = []
        for source_url, links in opportunities.items():
            for link in links:
                source_h1 = data.iloc[link['source_index']]['h1']
                target_h1 = data.iloc[link['target_index']]['h1']
                source_category = data.iloc[link['source_index']]['sub_category']
                target_category = data.iloc[link['target_index']]['sub_category']
                anchor_text = generate_anchor_text(source_h1, target_h1)
                results.append({
                    "Source URL": source_url,
                    "Source H1": source_h1,
                    "Source Category": source_category,
                    "Target URL": link['target_url'],
                    "Target H1": target_h1,
                    "Target Category": target_category,
                    "Anchor Text": anchor_text,
                    "Embedding Similarity": link['embedding_similarity'],
                    "Text Similarity": link['text_similarity'],
                    "Category Similarity": link['category_similarity'],
                    "Overall Similarity": link['overall_similarity']
                })
                print(f"Generated anchor text: {source_url} -> {link['target_url']}")
        
        # Save results to an Excel file
        print("\nSaving results to Excel file...")
        results_df = pd.DataFrame(results)
        results_df.to_excel('internal_linking_results.xlsx', index=False)
        print("Results saved to 'internal_linking_results.xlsx'.")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
