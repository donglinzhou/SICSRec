import os
from collections import defaultdict




def data_partition(data_dir, dataset_name):

    User = defaultdict(list)
    train_data = defaultdict(list)
    valid_data = defaultdict(list)
    test_data = defaultdict(list)

    input_file = os.path.join(data_dir, dataset_name, "sequential_data.txt")
    train_file = os.path.join(data_dir, dataset_name, f"{dataset_name}.train.inter")
    test_file = os.path.join(data_dir, dataset_name, f"{dataset_name}.test.inter")
    valid_file = os.path.join(data_dir, dataset_name, f"{dataset_name}.valid.inter")

    with open(input_file, "r") as f_in:
        for line in f_in:
            parts = list(map(int, line.strip().split()))
            user_id = parts[0]
            item_ids = parts[1:]
            for i in item_ids:
                User[user_id+1].append(i+1)

    uid_list = list(User.keys())
    uid_list.sort(key=lambda t: int(t))

    for user in User:
        nfeedback = len(User[user])
        if nfeedback < 3:
            train_data[user] = User[user]
            valid_data[user] = []
            test_data[user] = []
        else:
            train_data[user] = User[user][:-2]
            valid_data[user] = []
            valid_data[user].append(User[user][-2])
            test_data[user] = []
            test_data[user].append(User[user][-1])

    with open(train_file, "w", newline='') as file:
        file.write('user_id:token\titem_id_list:token_seq\titem_id:token\n')
        for uid in uid_list:
            item_seq = train_data[uid]
            seq_len = len(item_seq)
            for target_idx in range(1, seq_len):
                target_item = item_seq[-target_idx]
                seq = item_seq[:-target_idx][-50:]
                seq = [str(item) for item in seq]
                file.write(f'{uid}\t{" ".join(seq)}\t{target_item}\n')

    with open(valid_file, "w", newline='') as file:
        file.write('user_id:token\titem_id_list:token_seq\titem_id:token\n')
        for uid in uid_list:
            item_seq = train_data[uid][-50:]
            target_item = valid_data[uid][0]
            item_seq = [str(item) for item in item_seq]
            file.write(f'{uid}\t{" ".join(item_seq)}\t{target_item}\n')

    with open(test_file, "w", newline='') as file:
        file.write('user_id:token\titem_id_list:token_seq\titem_id:token\n')
        for uid in uid_list:
            item_seq = (train_data[uid] + valid_data[uid])[-50:]
            target_item = test_data[uid][0]
            item_seq = [str(item) for item in item_seq]
            file.write(f'{uid}\t{" ".join(item_seq)}\t{target_item}\n')

    print('success')


if __name__ == '__main__':
    data_dir = "dataset"
    dataset_name = "DY"
    data_partition(data_dir, dataset_name)


