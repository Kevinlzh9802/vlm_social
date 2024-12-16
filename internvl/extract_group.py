import os.path
import re
import json


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


def group_numbers_by_group(response):
    """Group numbers between each occurrence of 'group', ignoring numbers after 'group' and list-style numbers, and remove duplicates."""
    # response = remove_list_style_numbers(response)
    # response = remove_group_numbers(response)

    # Find all 'group' sections and the content that follows until the next 'group' or the end of the response
    group_matches = re.finditer(r'(?i)^\\s*group\\s*\\d*[:\\n](.*?)(?=^\\s*group\\s*\\d*[:\\n]|\\Z)', response,
                                re.S | re.M)
    grouped_numbers = []

    for match in group_matches:
        group_content = match.group(1)
        numbers = re.findall(r'\\d+', group_content)
        unique_numbers = sorted(set(numbers), key=numbers.index)
        if unique_numbers:
            grouped_numbers.append(unique_numbers)

    return grouped_numbers


def extract_numbers_and_parentheses(response):
    """Extract numbers and parentheses, ignoring numbers after 'group' and list-style numbers."""
    # response = remove_group_numbers(response)
    response = remove_list_style_numbers(response)
    return re.findall(r'\d+|[()]', response)


def extract_info(input_file, output_file):
    with open(input_file, 'r') as file:
        content = file.read()

    # Split the content by the dotted line separators
    sections = re.split(r'\n-+\n', content)

    results = []

    for section in sections:
        image_path = extract_image_path(section)
        file_name = extract_file_name(image_path)
        response = extract_response(section)

        if image_path and response:
            # Extract numbers and parentheses
            matches = extract_numbers_and_parentheses(response)
            grouped_numbers = group_numbers_by_group(response)

            # Append to results
            results.append({
                "file_name": file_name,
                "image_path": image_path,
                "extraction_results": matches,
                "grouped_numbers": grouped_numbers
            })

    # Write to the output JSON file
    with open(output_file, 'w') as json_file:
        json.dump(results, json_file, indent=4)

def main():
    file_name = 'InternVL2-4B_conflab_gallery_cgroup_2024-12-13 21:15:48.txt'
    result_path = '/home/zonghuan/tudelft/projects/vlm_social/internvl/experiments/results/'
    input_file_path = os.path.join(result_path, file_name)  # Replace with the path to your document
    output_file_path = os.path.join(result_path, 'group_extraction', file_name.split('.')[0] + '.json')
    extract_info(input_file_path, output_file_path)

if __name__ == '__main__':
    main()
