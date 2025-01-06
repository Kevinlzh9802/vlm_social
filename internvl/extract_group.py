import os.path
import re
import json
import pandas as pd
from metrics import compute_HIC, get_all_elems, HIC_stats
import numpy as np
import matplotlib.pyplot as plt

def extract_image_path(section):
    """Extract the image path from the section."""
    image_path_match = re.search(r'Image Path: (.+)', section)
    return image_path_match.group(1) if image_path_match else None


def extract_file_name(image_path):
    """Extract the file name from the image path."""
    return image_path.split('/')[-1] if image_path else None


def extract_response(section):
    """Extract the response content from the section."""
    response_match = re.search(r'Response:(.*)', section, re.S)
    return response_match.group(1).strip() if response_match else None


def remove_group_numbers(response):
    """Remove numbers that are trailing the word 'group' (case insensitive)."""
    return re.sub(r'(?i)group\s*\d+', 'group', response)


def remove_list_style_numbers(response):
    """Remove numbers followed by a dot if the contents after the dot are not empty."""
    return re.sub(r'\b\d+\.\s+\S+', '', response)


def extract_group_in_response(s):
    # Step 1: Remove any number before a dot ('.'), except if the dot is at the end
    s = remove_list_style_numbers(s)

    # Step 2: Search for 'group' pattern and handle accordingly
    group_pattern = re.compile(r'group\s*\d*', re.IGNORECASE)
    matches = list(group_pattern.finditer(s))

    if matches:
        sections = []
        for i in range(len(matches)):
            start = matches[i].end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(s)
            group_content = s[start:end]
            # Extract numbers from the group content
            numbers = list(map(int, re.findall(r'\d+', group_content)))
            sections.append(tuple(numbers))
        return sections

    # Step 3: Search for numbers inside parentheses
    paren_pattern = re.compile(r'\(([^)]*?)\)|\[(.*?)\]')
    paren_matches = paren_pattern.findall(s)

    if paren_matches:
        result = []
        for match in paren_matches:
            content = match[0] or match[1]
            numbers = list(map(int, re.findall(r'\d+', content)))
            result.append(tuple(numbers))
        return result

    # Step 4: Return all numbers as a list
    numbers = [tuple(map(int, re.findall(r'\d+', s)))]
    return numbers

def clear_grouping(group: list):
    return [list(set(tup)) for tup in group if tup]

def extract_info(input_file):
    with open(input_file, 'r', encoding='latin1') as file:
        content = file.read()

    # Split the content by the dotted line separators
    sections = re.split(r'\n-+\n', content)

    results = {}

    for section in sections:
        image_path = extract_image_path(section)
        file_name = extract_file_name(image_path)
        response = extract_response(section)

        if image_path and response:
            # Extract numbers and parentheses
            grouped_numbers = extract_group_in_response(response)
            cleared_groups = clear_grouping(grouped_numbers)
            # Append to results
            num = file_name.split('_')[0][-4:]
            results[num] = {
                "file_name": file_name,
                "image_path": image_path,
                # "extraction_results": matches,
                "grouped_numbers": grouped_numbers,
                "cleared_groups": cleared_groups
            }

    # Write to the output JSON file
    # with open(output_file, 'w') as json_file:
    #     json.dump(results, json_file, indent=4)

    return results

def deal_annotation_xlsx(gt_file, output_file, tab='Conversation group'):
    df = pd.read_excel(gt_file, sheet_name=tab)
    df = pd.DataFrame(df.values[3:])
    # Step 1: Drop the column before "Num"
    df = df.loc[:, df.columns[1:]]

    # Step 2: Replace the current row index with row 0 values
    df.columns = df.iloc[0]  # Set the first row as column headers
    df = df[1:]  # Drop the first row

    # Step 3: Ensure all values are strings
    df = df.applymap(lambda x: str(x) if not pd.isna(x) else x)

    # Step 4: Create the dictionary with values turned into tuples of integers
    result_dict = {}
    for index, row in df.iterrows():
        key = row['Num']  # Get the "Num" column value
        groups = [
            tuple(map(int, group.split(',')))  # Convert group string to tuple of integers
            for group in row[1:]  # Skip the "Num" column
            if pd.notna(group) and isinstance(group, str)  # Ignore NaN values and ensure strings
        ]
        result_dict[key] = groups

    # Print the resulting dictionary
    # print(result_dict)
    # Add singletons
    result_dict = add_singletons(result_dict, 'C:\\Users\\Zonghuan Li\\Downloads\\gallery')

    with open(output_file, 'w') as json_file:
        json.dump(result_dict, json_file, indent=4)
    return result_dict

def add_singletons(input_dict, img_folder):
    # Resulting dictionary to store updated values
    updated_dict = {}

    for key, value in input_dict.items():
        # 1. Turn the key into a 6-digit string
        six_digit_key = key.zfill(6)

        # 2. Find the subfolder that starts with the six-digit key
        subfolder = None
        for folder in os.listdir(img_folder):
            if folder.startswith(six_digit_key):
                subfolder = os.path.join(img_folder, folder)
                break

        if not subfolder or not os.path.isdir(subfolder):
            print(f"Subfolder for {six_digit_key} not found. Skipping...")
            continue

        # 3. Get all image filenames as a list of numbers
        all_people = []
        for file in os.listdir(subfolder):
            f_name = file.split('.')[0]
            if f_name.isdigit():  # Ensure file name is numeric
                all_people.append(int(f_name))

        # 4. Update the list of tuples in the dictionary
        existing_numbers = {num for tup in value for num in tup}  # Flatten existing tuples
        new_tuples = value.copy()

        for num in all_people:
            if num not in existing_numbers:
                new_tuples.append((num,))  # Append as a tuple
        updated_dict[key] = new_tuples

    return updated_dict

def evaluate_groups(result_folder, file_name):
    input_file_path = os.path.join(result_folder, file_name)
    if 'cgroup' in file_name:
        # gt_file = 'D:\\Desktop\\results\\conflab_cgroup.json'
        gt_file = '/home/zonghuan/tudelft/projects/vlm_social/conflab_cgroup.json'
    elif 'fform' in file_name:
        # gt_file = 'D:\\Desktop\\results\\conflab_fform.json'
        gt_file = '/home/zonghuan/tudelft/projects/vlm_social/conflab_fform.json'
    else:
        return

    results = extract_info(input_file_path)
    # group_stats(results)

    with open(gt_file, 'r') as f:
        gt = json.load(f)
    common_keys = list(results.keys() & gt.keys())
    common_keys.sort()
    results = {x: results[x]['cleared_groups'] for x in common_keys}
    gt = {x: gt[x] for x in common_keys}
    all_detected = get_all_elems(list(results.values()))
    hic = compute_HIC(common_keys, all_detected, results, gt)
    hic_stats = HIC_stats(hic)
    plot_hic_stats(result_folder, file_name, hic_stats)
    c = 9
    #
    # plt.clf()
    # plt.imshow(hic, cmap='viridis', interpolation='nearest')
    # # Add a color bar
    # plt.colorbar(label='Value')
    #
    # # Add labels for clarity
    # plt.title('HIC matrix' + file_name.split('.')[0])
    # plt.xlabel('Detected group cardinality')
    # plt.ylabel('GT group cardinality')
    #
    # plt.savefig(os.path.join(result_folder, file_name.split('.')[0] + '.png'), dpi=300, bbox_inches='tight')
    # plt.close()
    # # Show the plot
    # # plt.show()

def plot_hic_stats(result_folder, file_name, hic_stats):
    # Create a figure and subplots
    data1 = hic_stats['precision']
    data2 = hic_stats['recall']
    data3 = hic_stats['f1']
    fig, axes = plt.subplots(3, 1, figsize=(15, 5), constrained_layout=True)

    # Find the common color range across all heatmaps
    vmin = min(data1.min(), data2.min(), data3.min())
    vmax = max(data1.max(), data2.max(), data3.max())

    # Plot each heatmap
    heatmap1 = axes[0].imshow(data1, cmap='viridis', vmin=vmin, vmax=vmax)
    axes[0].set_title("precision")

    heatmap2 = axes[1].imshow(data2, cmap='viridis', vmin=vmin, vmax=vmax)
    axes[1].set_title("recall")

    heatmap3 = axes[2].imshow(data3, cmap='viridis', vmin=vmin, vmax=vmax)
    axes[2].set_title("f1")

    # Add a single shared colorbar
    cbar = fig.colorbar(heatmap1, ax=axes, orientation='vertical', fraction=0.02, pad=0.02)
    cbar.set_label("Shared Colorbar")
    plt.savefig(os.path.join(result_folder, file_name.split('.')[0] + '.png'), dpi=300, bbox_inches='tight')


def group_stats(results):
    all_groups = []
    for x in results.values():
        all_groups += x['grouped_numbers']
    long_groups = len([1 for x in all_groups if len(x) > 20])
    dup_groups = len([1 for x in all_groups if len(x) > len(set(x))])
    empty_groups = len([1 for x in all_groups if not x])
    max_card = max(len(x) for x in all_groups)
    print(f'long groups: {long_groups} / {len(all_groups)}, {long_groups/len(all_groups)}')
    print(f'duplicate groups: {dup_groups} / {len(all_groups)}, {dup_groups/len(all_groups)}')
    print(f'empty groups: {empty_groups} / {len(all_groups)}, {empty_groups/len(all_groups)}')
    print(f'max card: {max_card}')

def main():
    # gt_original ='D:\\Desktop\\results\\Conflab.xlsx'
    # deal_annotation_xlsx(gt_original, 'D:\\Desktop\\results\\conflab_cgroup2.json',
    #                      tab='Conversation group')
    # deal_annotation_xlsx(gt_original, 'D:\\Desktop\\results\\conflab_fform2.json',
    #                      tab='F-formation')

    # result_folder = 'D:\\Desktop\\results\\'
    result_folder = '/home/zonghuan/tudelft/projects/vlm_social/internvl/experiments/cluster_raw/results/'
    for file in os.listdir(result_folder):
        if file.endswith('.txt'):
            print(file)
            try:
                evaluate_groups(result_folder, file)
            except Exception as e:
                print(e)

    # # file = 'InternVL2-4B_conflab_gallery_fform_2024-12-13-21-15-48.txt'
    # file = 'InternVL2_5-1B_conflab_concat_cgroup_2024-12-13-21-15-48.txt'
    # # file = 'InternVL2_5-1B_conflab_concat_fform_2024-12-13-21-15-48.txt'
    # try:
    #     evaluate_groups(result_folder, file)
    # except Exception as e:
    #     print(e)
    #     print(file)
    # single_text_file = f"InternVL2-2B_conflab_gallery_cgroup_2024-12-13-21-15-48.txt"
    # output_file_path = os.path.join(result_folder, 'group_extraction', single_text_file.split('.')[0] + '.json')

    c = 9

if __name__ == '__main__':
    main()
