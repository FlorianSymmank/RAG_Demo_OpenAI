# Imports
from pdf2image import convert_from_path
from pdf2image.exceptions import (
    PDFInfoNotInstalledError,
    PDFPageCountError,
    PDFSyntaxError
)
from pdfminer.high_level import extract_text
import base64
from io import BytesIO
import os
import concurrent
from tqdm import tqdm
from openai import OpenAI
import re
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import json
import numpy as np
from rich import print
from ast import literal_eval
from dotenv import load_dotenv

load_dotenv()


def convert_doc_to_images(path):
    images = convert_from_path(path)
    return images


def extract_text_from_doc(path):
    text = extract_text(path)
    page_text = []
    return text


def get_img_uri(img):
    buffer = BytesIO()
    img.save(buffer, format="jpeg")
    base64_image = base64.b64encode(buffer.getvalue()).decode("utf-8")
    data_uri = f"data:image/jpeg;base64,{base64_image}"
    return data_uri


system_prompt = '''
You will be provided with an image of a pdf page or a slide. Your goal is to talk about the content that you see, in technical terms, as if you were delivering a presentation.

If there are diagrams, describe the diagrams and explain their meaning.
For example: if there is a diagram describing a process flow, say something like "the process flow starts with X then we have Y and Z..."

If there are tables, describe logically the content in the tables
For example: if there is a table listing items and prices, say something like "the prices are the following: A for X, B for Y..."

DO NOT include terms referring to the content format
DO NOT mention the content type - DO focus on the content itself
For example: if there is a diagram/chart and text on the image, talk about both without mentioning that one is a chart and the other is text.
Simply describe what you see in the diagram and what you understand from the text.

You should keep it concise, but keep in mind your audience cannot see the image so be exhaustive in describing the content.

Exclude elements that are not relevant to the content:
DO NOT mention page numbers or the position of the elements on the image.

------

If there is an identifiable title, identify the title to give the output in the following format:

{TITLE}

{Content description}

If there is no clear title, simply return the content description.

'''


def analyze_image(img_url):
    response = client.chat.completions.create(
        model="gpt-4o",
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": img_url
                        }
                    },
                ],
            }
        ],
        max_tokens=300,
        top_p=0.1
    )

    return response.choices[0].message.content


def analyze_doc_image(img):
    img_uri = get_img_uri(img)
    data = analyze_image(img_uri)
    return data


embeddings_model = "text-embedding-3-large"


def get_embeddings(text):
    embeddings = client.embeddings.create(
        model="text-embedding-3-small",
        input=text,
        encoding_format="float"
    )
    return embeddings.data[0].embedding


def search_content(df, input_text, top_k):
    embedded_value = get_embeddings(input_text)
    df["similarity"] = df.embeddings.apply(lambda x: cosine_similarity(
        np.array(x).reshape(1, -1), np.array(embedded_value).reshape(1, -1)))
    res = df.sort_values('similarity', ascending=False).head(top_k)
    return res


def get_similarity(row):
    similarity_score = row['similarity']
    if isinstance(similarity_score, np.ndarray):
        similarity_score = similarity_score[0][0]
    return similarity_score


system_prompt = '''
    You will be provided with an input prompt and content as context that can be used to reply to the prompt.
    
    You will do 2 things:
    
    1. First, you will internally assess whether the content provided is relevant to reply to the input prompt. 
    
    2a. If that is the case, answer directly using this content. If the content is relevant, use elements found in the content to craft a reply to the input prompt.

    2b. If the content is not relevant, use your own knowledge to reply or say that you don't know how to respond if your knowledge is not sufficient to answer.
    
    Stay concise with your answer, replying specifically to the input prompt without mentioning additional information provided in the context content.
'''


def generate_output(input_prompt, similar_content, threshold=0.5):

    content = similar_content.iloc[0]['content']

    # Adding more matching content if the similarity is above threshold
    if len(similar_content) > 1:
        for i, row in similar_content.iterrows():
            similarity_score = get_similarity(row)
            if similarity_score > threshold:
                content += f"\n\n{row['content']}"

    prompt = f"INPUT PROMPT:\n{input_prompt}\n-------\nCONTENT:\n{content}"

    completion = client.chat.completions.create(
        model="gpt-4o",
        temperature=0.5,
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return completion.choices[0].message.content


client = OpenAI()

files_path = "data/example_pdfs"

all_items = os.listdir(files_path)
files = [item for item in all_items if os.path.isfile(
    os.path.join(files_path, item))]

docs = []

for f in files:

    path = f"{files_path}/{f}"
    doc = {
        "filename": f
    }
    text = extract_text_from_doc(path)
    doc['text'] = text
    imgs = convert_doc_to_images(path)
    pages_description = []

    print(f"Analyzing pages for doc {f}")

    # Concurrent execution
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:

        futures = [
            executor.submit(analyze_doc_image, img)
            for img in imgs
        ]

        with tqdm(total=len(imgs)-1) as pbar:
            for _ in concurrent.futures.as_completed(futures):
                pbar.update(1)

        for f in futures:
            res = f.result()
            pages_description.append(res)

    doc['pages_description'] = pages_description
    docs.append(doc)

json_path = "data/parsed_pdf_docs.json"

# Saving result to file for later
with open(json_path, 'w') as f:
    json.dump(docs, f)

# Optional: load content from the saved file
with open(json_path, 'r') as f:
    docs = json.load(f)

# Chunking content by page and merging together slides text & description if applicable
content = []
for doc in docs:
    text = doc['text'].split('\f')
    description = doc['pages_description']
    description_indexes = []
    for i in range(len(text)):
        slide_content = text[i] + '\n'
        # Trying to find matching slide description
        slide_title = text[i].split('\n')[0]
        for j in range(len(description)):
            description_title = description[j].split('\n')[0]
            if slide_title.lower() == description_title.lower():
                slide_content += description[j].replace(description_title, '')
                # Keeping track of the descriptions added
                description_indexes.append(j)
        # Adding the slide content + matching slide description to the content pieces
        content.append(slide_content)
    # Adding the slides descriptions that weren't used
    for j in range(len(description)):
        if j not in description_indexes:
            content.append(description[j])

# Cleaning up content
# Removing trailing spaces, additional line breaks, page numbers and references to the content being a slide
clean_content = []
for c in content:
    text = c.replace(' \n', '').replace(
        '\n\n', '\n').replace('\n\n\n', '\n').strip()
    text = re.sub(r"(?<=\n)\d{1,2}", "", text)
    text = re.sub(r"\b(?:the|this)\s*slide\s*\w+\b",
                  "", text, flags=re.IGNORECASE)
    clean_content.append(text)

for c in clean_content:
    print(c)
    print("\n\n-------------------------------\n\n")

# Creating the embeddings
# We'll save to a csv file here for testing purposes but this is where you should load content in your vectorDB.
df = pd.DataFrame(clean_content, columns=['content'])
print(df.shape)
df.head()

df['embeddings'] = df['content'].apply(lambda x: get_embeddings(x))
df.head()

data_path = "data/parsed_pdf_docs_with_embeddings.csv"

# Saving result to file for later
df.to_csv(data_path, index=False)

# Optional: load content from the saved file
df = pd.read_csv(data_path)
df["embeddings"] = df.embeddings.apply(literal_eval).apply(np.array)

example_inputs = [
    "Was sind wichtige Prüfaspekte?", # S. 7
    "Wie sollten studentische Stellungnahmen aufgebaut sein? ", # S. 8
    "Kann ich als Biologie Chemie Lehramt Student Mitglied der Fachschaft sein? Mein Kernfach ist Biologie. Schau mal in der Satzung nach.", # S. 7 
    "Welche Studiengänge dürfen laut Satzung in der FSI sein?",
    "Was ist der Institutsrat?", # S. 16
    "Nenne mir wichtige Punkte zum Thema Moodle?", # S. 19

]

# Running the RAG pipeline on each example
for ex in example_inputs:
    print(f"[deep_pink4][bold]QUERY:[/bold] {ex}[/deep_pink4]\n\n")
    matching_content = search_content(df, ex, 3)
    print(f"[grey37][b]Matching content:[/b][/grey37]\n")
    for i, match in matching_content.iterrows():
        print(
            f"[grey37][i]Similarity: {get_similarity(match):.2f}[/i][/grey37]")
        print(
            f"[grey37]{match['content'][:100]}{'...' if len(match['content']) > 100 else ''}[/[grey37]]\n\n")
    reply = generate_output(ex, matching_content)
    print(
        f"[turquoise4][b]REPLY:[/b][/turquoise4]\n\n[spring_green4]{reply}[/spring_green4]\n\n--------------\n\n")
