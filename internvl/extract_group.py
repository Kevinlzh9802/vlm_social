import os.path
import re
import json
import pandas as pd

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


# def group_numbers_by_group(response):
#     """Group numbers between each occurrence of 'group', ignoring numbers after 'group' and list-style numbers, and remove duplicates."""
#     # response = remove_list_style_numbers(response)
#     # response = remove_group_numbers(response)
#
#     # Find all 'group' sections and the content that follows until the next 'group' or the end of the response
#     group_matches = re.finditer(r'(?i)^\\s*group\\s*\\d*[:\\n](.*?)(?=^\\s*group\\s*\\d*[:\\n]|\\Z)', response,
#                                 re.S | re.M)
#     grouped_numbers = []
#
#     for match in group_matches:
#         group_content = match.group(1)
#         numbers = re.findall(r'\\d+', group_content)
#         unique_numbers = sorted(set(numbers), key=numbers.index)
#         if unique_numbers:
#             grouped_numbers.append(unique_numbers)
#
#     return grouped_numbers
#
#
# def extract_numbers_and_parentheses(response):
#     """Extract numbers and parentheses, ignoring numbers after 'group' and list-style numbers."""
#     # response = remove_group_numbers(response)
#     response = remove_list_style_numbers(response)
#     return re.findall(r'\d+|[()]', response)

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
    numbers = list(map(int, re.findall(r'\d+', s)))
    return numbers


def extract_info(input_file, output_file):
    with open(input_file, 'r') as file:
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
            # matches = extract_numbers_and_parentheses(response)
            grouped_numbers = extract_group_in_response(response)

            # Append to results
            num = file_name.split('_')[0][-4:]
            results[num] = {
                "file_name": file_name,
                "image_path": image_path,
                # "extraction_results": matches,
                "grouped_numbers": grouped_numbers
            }

    # Write to the output JSON file
    # with open(output_file, 'w') as json_file:
    #     json.dump(results, json_file, indent=4)

    return results

def deal_annotation_xlsx(input_file, output_file, tab='Conversation group'):
    gt_file = '/home/zonghuan/tudelft/projects/vlm_social/Conflab.xlsx'

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
    with open(output_file, 'w') as json_file:
        json.dump(result_dict, json_file, indent=4)
    return result_dict


def main():


    result_folder = '/home/zonghuan/tudelft/projects/vlm_social/internvl/experiments/cluster_raw/results/'
    # result_path = '/home/zonghuan/tudelft/projects/vlm_social/internvl/experiments/results/'
    single_text_file = f"InternVL2-2B_conflab_gallery_cgroup_2024-12-13 21:15:48.txt"
    input_file_path = os.path.join(result_folder, single_text_file)  # Replace with the path to your document
    output_file_path = os.path.join(result_folder, 'group_extraction', single_text_file.split('.')[0] + '.json')

    if 'cgroup' in single_text_file:
        gt_file = '/home/zonghuan/tudelft/projects/vlm_social/conflab_cgroup.json'
    elif 'fform' in single_text_file:
        gt_file = '/home/zonghuan/tudelft/projects/vlm_social/conflab_fform.json'
    else:
        gt_file = None
    a = extract_info(input_file_path, output_file_path)
    with open(gt_file, 'r') as f:
        b = json.load(f)
    # deal_annotation_xlsx(gt_file, '/home/zonghuan/tudelft/projects/vlm_social/conflab_cgroup.json',
    #                      tab='Conversation group')
    #
    # deal_annotation_xlsx(gt_file, '/home/zonghuan/tudelft/projects/vlm_social/conflab_fform.json',
    #                      tab='F-formation')
    # files = [f for f in os.listdir(result_folder) if f.endswith('.txt')]

    # file_name = "InternVL2-2B_conflab_gallery_fform.txt"
    # result_path = 'D:\\Desktop\\'

    c = 9

if __name__ == '__main__':
    main()
