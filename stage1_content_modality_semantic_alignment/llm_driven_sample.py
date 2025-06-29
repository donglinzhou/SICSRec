import torch
import os
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
import re

if __name__ == "__main__":
    category = "Bili_Movie"
    file = "Downstream_datasets"
    data_directory = f'./{file}/{category}/'
    meta_file_path = os.path.join(data_directory, f'{category}_item_sort.csv')

    sequential_data_path = f'./{file}/{category}/sequential_data.txt'
    output_id_path = f'./{file}/{category}/item_pairs_id.txt'
    output_cn_path = f'./{file}/{category}/item_pairs_cn.txt'
    output_en_path = f'./{file}/{category}/item_pairs_en.txt'

    column_names = ['item_id', 'chinese_title', 'english_title']
    df = pd.read_csv(meta_file_path, header=None, names=column_names, encoding='utf-8')
    df.sort_values(by='item_id', inplace=True)
    df.reset_index(drop=True, inplace=True)

    device = "cuda:1"

    path = './glm-4-chat/'
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)  #
    model = AutoModelForCausalLM.from_pretrained(
        path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    ).to(device).eval()

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
                f"你是一个视频相似度判断助手。我会给你一个目标视频和一组候选视频，请帮我找出候选视频中与目标视频最相似的一个。"
                f"目标视频是 {target_item}-{target_cn_title}，候选视频列表是 {candidate_item_descriptions}。"
                "请你在候选视频中找到与目标视频最相似的，并按以下格式输出: 相似视频ID-相似视频标题。"
                "例如: 3-花园酒店。如果没有相似的视频，请直接输出-1。请确保格式正确，其他形式的输出都是非法的。")

            inputs = tokenizer.apply_chat_template([{"role": "user", "content": query}],
                                                   add_generation_prompt=True,
                                                   tokenize=True,
                                                   return_tensors="pt",
                                                   return_dict=True
                                                   )

            inputs = inputs.to(device)
            gen_kwargs = {"max_length": 15000, "do_sample": True, "top_k": 1}
            with torch.no_grad():
                outputs = model.generate(**inputs, **gen_kwargs)
                outputs = outputs[:, inputs['input_ids'].shape[1]:]
                response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                print(response)

            pattern = r"^(\d+)-([^,]+)$"
            match = re.match(pattern, response.strip())

            if match:
                most_similar_item = match.group(1).strip()
                if most_similar_item == '-1':
                    most_similar_item = -1
                    most_similar_title = None
                else:
                    most_similar_item = int(most_similar_item)
                    most_similar_title = match.group(2).strip()

                if most_similar_item != -1:
                    print(f"目标视频对：目标视频ID={target_item},目标视频标题={target_cn_title}")
                    print(f"合法输出: 最相似视频ID={most_similar_item}, 最相似视频标题={most_similar_title}")
                else:
                    print("没有找到相似视频")
                    continue
            else:
                print("非法输出，跳过处理")
                continue

            if most_similar_item != -1:
                similar_cn_title = df.loc[df['item_id'] == most_similar_item, 'chinese_title'].squeeze() or "无"
                similar_en_title = df.loc[df['item_id'] == most_similar_item, 'english_title'].squeeze() or "None"
                id_file.write(f"{target_item},{most_similar_item}\n")
                cn_file.write(f"{target_cn_title},{similar_cn_title}\n")
                en_file.write(f"{target_en_title},{similar_en_title}\n")

    print("finish!")