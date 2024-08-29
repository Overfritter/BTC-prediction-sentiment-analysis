import pandas as pd
import os
import json
from openai import OpenAI
import time
from dotenv import load_dotenv

# Replace with your OpenAI API key
load_dotenv()
os.environ.get("OPENAI_API_KEY")

client = OpenAI()

#create batches from the intermediate.csv file
df = pd.read_csv("tweet_delete_16M.csv") #temporary test file

sentiment_prompt = "You are a financial expert in Bitcoin and have to classify Twitter posts about Bitcoin as either positive, negative or neutral. Positive => 0.01-0.99, Neutral => 0.0, Negative => -0.99--0.01. Positive tweet represents a bullish sentiment and might mean an increase in the price of Bitcoin, Neutral tweet represents a neutral sentiment and indicates no change in price. Negative tweet represents a bearish sentiment and might mean a decrease in the price of Bitcoin. Only return corresponding number."

#create the column 'id'
#df['id'] = df.index

#split the dataset into batches
def create_batches(df, batch_size):
    for i in range(0, len(df), batch_size): 
        yield df[i:i + batch_size]

#create a jsonl-file for every batch
def generate_tasks(df):
    tasks = []
    for index, row in df.iterrows():
        post = row["text"]
        task = {
            "custom_id": "request-" + str(index + 1),
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "gpt-4o-mini",
                "messages": [
                    {
                        "role": "system",
                        "content": sentiment_prompt
                    },
                    {
                        "role": "user",
                        "content": post
                    }
                ],
                "max_tokens": 4
            }
        }
        tasks.append(task)
    return tasks

def save_tasks_to_jsonl(tasks, file_name):
    with open(file_name, 'w', encoding='utf-8') as file:
        for task in tasks:
            file.write(json.dumps(task, ensure_ascii=False) + '\n')

#Delete the brackets at the beginning and end of each jsonl object
def process_jsonl_file(input_file_path, output_file_path):
    with open(input_file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    processed_lines = []
    for line in lines:
        stripped_line = line.strip()
        if stripped_line.startswith('[') and stripped_line.endswith(']'):
            processed_line = stripped_line[1:-1]
            processed_lines.append(processed_line)
        else:
            processed_lines.append(stripped_line)

    with open(output_file_path, 'w', encoding='utf-8') as file:
        for line in processed_lines:
            file.write(f"{line}\n")

def send_batch(jsonl_file):
    batch_file = client.files.create(
        file=open(jsonl_file, "rb"),
        purpose='batch'
    )
    print(batch_file)
    batch_file_id = batch_file.id

    batch_job = client.batches.create(
        input_file_id = batch_file_id,
        endpoint = "/v1/chat/completions",  
        completion_window = "24h",
        metadata = {
            "description": "nightly eval job",
        }
    )
    batch = client.batches.list(limit=1) #get the latest batch information to extract id from
    #print(batch)

    # data = batch['data']
    last_batch_id = batch.first_id
    print(last_batch_id)
    return last_batch_id

def check_batch_status(batch_id):
    response = client.batches.retrieve(batch_id)
    print(response.status)
    return response.status

#get results and extract the sentiment score
def retrieve_results(batch_id):
    response = client.batches.retrieve(batch_id)
    output_file_id = response.output_file_id
    file_response = client.files.content(output_file_id)
    file_content = file_response.read()
    data = []
    for line in file_content.splitlines():
        record = json.loads(line)
        if 'custom_id' in record and 'response' in record:
            response = record['response']
            if 'body' in response and 'choices' in response['body']:
                choices = response['body']['choices']
                if len(choices) > 0 and 'message' in choices[0]:
                    message = choices[0]['message']
                    if 'content' in message:
                        data.append({'line_id': record['custom_id'], 'content': message['content']})      
    return pd.DataFrame(data)

def append_results(df, results): #add the results to the original dataframe starting from the first
    return pd.concat([df, results], ignore_index=True)

def process_batches(df, batch_size):
    #results_df = pd.DataFrame(columns=['line_id', 'content'])
    batches = create_batches(df, batch_size)
    for batch in batches:
        tasks = generate_tasks(batch)
        raw_jsonl_file = "batch_input.jsonl"
        processed_jsonl_file = "processed_batch_input.jsonl"
        save_tasks_to_jsonl(tasks, raw_jsonl_file)
        
        # Process the JSONL file to remove brackets
        process_jsonl_file(raw_jsonl_file, processed_jsonl_file)
        
        batch_id = send_batch(processed_jsonl_file)
        while True:
            status = check_batch_status(batch_id)
            if status == 'completed':
                results = retrieve_results(batch_id)
                results.to_csv('only_results.csv', index=False)
                df = append_results(df, results)
                df['line_id'] = df['line_id'].str.replace('request-', '')
                break
            elif status == 'failed':
                print('Batch failed')
                break
            time.sleep(60)  # Wait for 10 minutes -> 600
    return df 

# Processing batches of 2200 requests
result_df = process_batches(df, 2200)

# Save the results
result_df.to_csv('results_test_16M.csv', index=False)
