# from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
#
# model_name_or_path = "llm/Mistral-7B-Instruct-v0.2-GPTQ"
# # To use a different branch, change revision
# # For example: revision="gptq-4bit-32g-actorder_True"
# model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
#                                              device_map="auto",
#                                              trust_remote_code=False,
#                                              revision="main")
#
# tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
#
# prompt = "Parle nous de l'IA:"
# prompt_template=f'''<s>[INST] {prompt} [/INST]
# '''
#
# print("\n\n*** Generate:")
#
# input_ids = tokenizer(prompt_template, return_tensors='pt').input_ids.cuda()
# output = model.generate(inputs=input_ids, temperature=0.7, do_sample=True, top_p=0.95, top_k=40, max_new_tokens=512)
# print(tokenizer.decode(output[0]))


from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

model_name_or_path = "llm/Vigogne-2-7B-Instruct-GPTQ"
# To use a different branch, change revision
# For example: revision="main"
model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                             device_map="auto",
                                             trust_remote_code=False,
                                             revision="main")

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)

prompt = "Tell me about AI"
prompt_template=f'''Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{prompt}

### Response:

'''

print("\n\n*** Generate:")

input_ids = tokenizer(prompt_template, return_tensors='pt').input_ids.cuda()
output = model.generate(inputs=input_ids, temperature=0.7, do_sample=True, top_p=0.95, top_k=40, max_new_tokens=512)
print(tokenizer.decode(output[0]))
