POPE-style examiner

python DSG.py --conv_script <input_file> --output_file <output_json_file_name> --model_base <model_base> --pope_model_name <pope_model_name>
conv_script: input conversational script file
output_file: name of output file storing question and model answer
model_base: model based of the evaluated model
model_path: model_path of the evaluated model
pope_model_name: The name of the GPT model to be used as the POPE question extractor. The default model is gpt-4o-mini
verbose: whether print out the extracted question