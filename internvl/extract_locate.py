import re
import json

def read_file(file_path):
    """Reads the file and splits it into blocks based on the separator."""
    with open(file_path, 'r') as f:
        content = f.read()
    # Split the content into blocks separated by '---'
    blocks = content.split('-' * 80)
    # Remove any empty blocks
    return [block.strip() for block in blocks if block.strip()]

def extract_response(block):
    """Extracts the 'Response' content from a block."""
    response_match = re.search(r'Response:\s*(.*)', block)
    return response_match.group(1).strip() if response_match else None

def extract_image_name(block):
    """Extracts the image file name from a block."""
    path_match = re.search(r'Image Path:\s*/.*/(.*\.jpg)', block)
    return path_match.group(1) if path_match else None

def match_position(response, pos):
    """Matches the response content to the given position using string matching or capital letters."""
    pos_map = {
        "top-left": "A",
        "top-right": "B",
        "bottom-left": "C",
        "bottom-right": "D"
    }
    # Check for string matching
    pos_keywords = pos.split('-')  # e.g., ["top", "left"]
    if all(keyword in response.lower() for keyword in pos_keywords):
        return True

    # Check for capital letter matching
    if pos_map[pos] in response:
        return True

    return False

def process_blocks(file_path, position_dict):
    """Processes the blocks, extracts required information, and evaluates matches."""
    blocks = read_file(file_path)
    correct = 0

    for block in blocks:
        response = extract_response(block)
        image_name = extract_image_name(block)

        if not response or not image_name:
            continue  # Skip blocks without required information

        if image_name not in position_dict:
            continue  # Skip if image name not in the provided dictionary

        pos = position_dict[image_name]["position"]

        if match_position(response, pos):
            correct += 1

    return correct, len(blocks)

def calculate_accuracy(correct, total):
    """Calculates accuracy as correct / total."""
    return correct / total if total > 0 else 0

def evaluate_locat(file_path, json_path):
    # Paths to the files
    # file_path = 'input.txt'  # Replace with your file path
    # json_path = 'positions.json'  # Replace with your JSON file path

    # Read the JSON file
    with open(json_path, 'r') as f:
        position_dict = json.load(f)

    # Process the blocks and get the results
    correct, total = process_blocks(file_path, position_dict)

    # Calculate and print the accuracy
    accuracy = calculate_accuracy(correct, total)
    print(f"Correct: {correct}")
    print(f"Total: {total}")
    print(f"Accuracy: {accuracy:.2%}")

def main():

    for file in os.listdir(result_folder):
        if file.endswith('.txt'):
            print(file)
            try:
                evaluate_groups(result_folder, file)
            except Exception as e:
                print(e)


if __name__ == "__main__":
    main()