import json
import random

def create_qa_chat_files(input_path, output_directory, number_of_files=5):
    """
    Generate a number of text files with Q&A pairs in a chat format.

    :param input_path: Path to the JSON file containing Q&A data.
    :param output_directory: Directory where the output files will be saved.
    :param number_of_files: Number of text files to generate.
    :return: List of paths to the created files.
    """
    try:
        # Load the dataset from the provided JSON file
        with open(input_path, 'r') as file:
            data = json.load(file)

        created_files = []

        for file_number in range(1, number_of_files + 1):
            # Select 2-5 random objects from the dataset
            selected_qa_pairs = random.sample(data, random.randint(2, 5))

            # Construct the chat text
            chat_text = '[START]S: You are a chatbot created by Peyton to answer simple questions about his resume. ' \
                             'If you do not know an answer, reply with \'I am not trained on this information, ' \
                             'you may contact Peyton to find out more.\' \n'
            for qa in selected_qa_pairs:
                chat_text += 'Q: ' + qa['question'] + '\nA: ' + qa['answer'] + '\n'
            chat_text += '[END]'

            # Write the text to a file
            filename = f'{output_directory}/chat_{file_number}.txt'
            with open(filename, 'w') as file:
                file.write(chat_text)

            created_files.append(filename)

        return created_files

    except Exception as e:
        return f"An error occurred: {e}"

# Example usage
input_path = 'datasets/qa.json'  # Replace with the actual input path
output_directory = 'generatedchats'  # Replace with the desired output directory
number_of_files = 100  # Number of text files to create

# Call the function
# The paths to the created files will be stored in 'created_files'
created_files = create_qa_chat_files(input_path, output_directory, number_of_files)