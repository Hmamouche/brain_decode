from huggingface_hub import snapshot_download
#snapshot_download(repo_id="TheBloke/Mistral-7B-Instruct-v0.2-GPTQ", local_dir = "llm/Mistral-7B-Instruct-v0.2-GPTQ", local_dir_use_symlinks=False)

#snapshot_download(repo_id="TheBloke/Vigogne-2-7B-Instruct-GPTQ", local_dir = "llm/Vigogne-2-7B-Instruct-GPTQ", local_dir_use_symlinks=False)


snapshot_download(repo_id="bofenghuang/vigogne-2-7b-instruct", local_dir = "/home/youssef.hmamouche/lustre/pt_cloud-muhqxqc6fxo/users/youssef.hmamouche/vigogne-2-7b-instruct", local_dir_use_symlinks=False)
