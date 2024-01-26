import json

def convert_json_to_txt(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as file:
        qa_data = json.load(file)

    with open(output_file, 'w', encoding='utf-8') as file:
        for qa in qa_data:
            question = qa.get("question", "")
            answer = qa.get("answer", "")
            formatted_qa = f"[START]S: You are a chatbot created by Peyton to answer simple questions about his resume. If you do not know an answer, reply with 'I am not trained on this information, you may contact Peyton to find out more.'Q: {question}A: {answer}[END]\n"
            file.write(formatted_qa)

def main():
    input_file = 'datasets/qa.json'
    output_file = 'qaquestions.txt'
    convert_json_to_txt(input_file, output_file)
    print(f"QA pairs have been successfully converted and saved in {output_file}")

if __name__ == "__main__":
    main()