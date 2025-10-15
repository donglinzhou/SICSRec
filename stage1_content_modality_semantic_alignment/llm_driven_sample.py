
import os
import pandas as pd
import re

# Please install OpenAI SDK first: `pip3 install openai`

from openai import OpenAI


if __name__ == "__main__":
    category = "Bili_Dance" # Dance
    file = "Downstream_datasets"
    data_directory = f'./{file}/{category}/'
    meta_file_path = os.path.join(data_directory, f'{category}_item_sort.csv')

    sequential_data_path = f'./{file}/{category}/sequential_data.txt'
    output_id_path = f'./{file}/{category}/item_pairs_id_ds_v2.txt'
    output_cn_path = f'./{file}/{category}/item_pairs_cn_ds_v2.txt'
    output_en_path = f'./{file}/{category}/item_pairs_en_ds_v2.txt'

    column_names = ['item_id', 'chinese_title', 'english_title']
    df = pd.read_csv(meta_file_path, header=None, names=column_names, encoding='utf-8')
    df.sort_values(by='item_id', inplace=True)
    df.reset_index(drop=True, inplace=True)
    client = OpenAI(api_key="", base_url="")

    with open(sequential_data_path, 'r') as file:
        lines = file.readlines()

    with open(output_id_path, 'w', encoding='utf-8') as id_file, open(output_cn_path, 'w',
                                                                      encoding='utf-8') as cn_file, open(output_en_path,
                                                                                                         'w',
                                                                                                         encoding='utf-8') as en_file:
        for line in lines:
            data = list(map(int, line.strip().split()))
            user_id = data[0]
            print(f"用户：{user_id}")
            item_sequence = data[1:]
            target_item = item_sequence[-1]
            candidate_items = item_sequence[:-1]

            target_cn_title = df[df['item_id'] == target_item]['chinese_title'].values[0]
            target_en_title = df[df['item_id'] == target_item]['english_title'].values[0]

            candidate_cn_titles = df[df['item_id'].isin(candidate_items)]['chinese_title'].tolist()
            candidate_en_titles = df[df['item_id'].isin(candidate_items)]['english_title'].tolist()

            candidate_item_descriptions = ', '.join(
                [f"{item_id}-{title}" for item_id, title in zip(candidate_items, candidate_cn_titles)]
            )

            query = (
                f"我会给你一个目标视频和一组候选视频，请帮我找出候选视频中与目标视频最相似的一个。"
                f"目标视频是 {target_item}-{target_cn_title}，候选视频列表是 {candidate_item_descriptions}。"
                "请你在候选视频中找到与目标视频最相似的，并按以下格式输出: 目标视频ID-最相似视频ID。"
                "例如: 123-456。如果候选视频中没有相似的视频，请直接输出 -1,-1。请确保格式正确，其他形式的输出都是非法的。"
            )

            response = client.chat.completions.create(
                model="",
                messages=[
                    {"role": "system", "content": "你是一个视频相似度判断助手。"},
                    {"role": "user", "content": query},
                ],
                stream=False
            )

            print(response.choices[0].message.content)
            content = response.choices[0].message.content

            pattern = r"^(-?\d+)-(-?\d+)$"
            match = re.match(pattern, content.strip())

            if match:
                target_out_id = match.group(1).strip()
                similar_out_id = match.group(2).strip()

                try:
                    target_out_id = int(target_out_id)
                    similar_out_id = int(similar_out_id)
                except ValueError:
                    print(f"  Invalid ID format in output: {content}. Skipping.")
                    continue

                if target_out_id == -1 and similar_out_id == -1:
                    print(f"  No similar video found for target {target_item}.")
                    continue  
                elif target_out_id != target_item:
                    print(f"  Mismatched target ID in output: Expected {target_item}, got {target_out_id}. Skipping.")
                    continue  
                elif similar_out_id not in candidate_items:
                    print(f"  Similar ID {similar_out_id} not in candidate list {candidate_items}. Skipping.")
                    continue  
                elif similar_out_id == -1:
                    print(f"  Found target {target_item}, but no similar candidate found by LLM.")
                    continue  
                else:
                    print(f"  Valid output: Target ID={target_out_id}, Similar ID={similar_out_id}")

                   
                    try:
                        similar_cn_title = df.loc[df['item_id'] == similar_out_id, 'chinese_title'].squeeze()
                        similar_en_title = df.loc[df['item_id'] == similar_out_id, 'english_title'].squeeze()
                       
                        if pd.isna(similar_cn_title) or pd.isna(similar_en_title):
                            print(f"  Missing title for item ID {similar_out_id}. Skipping.")
                            continue
                    except Exception as e:
                        print(f"  Error fetching titles for ID {similar_out_id}: {e}. Skipping.")
                        continue

                 
                    id_file.write(f"{target_out_id},{similar_out_id}\n")
                    cn_file.write(f"{target_cn_title},{similar_cn_title}\n")
                    en_file.write(f"{target_en_title},{similar_en_title}\n")
                    print(f"  Successfully wrote pair: {target_out_id} -> {similar_out_id}\n")

            else:
                print(f"  Illegal output format: {content}. Skipping.")
                continue

        print("Processing finished!")
