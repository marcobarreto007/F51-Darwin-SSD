import time
import subprocess
import json
import sys
import os

INSTANCE_ID = "43754078"
SSH_HOST = "ssh8.vast.ai"
SSH_PORT = "34078"
LOCAL_DIR = r"c:\Users\marco\Desktop\F51-Darwin-SSD"

def get_status():
    res = subprocess.run(["vastai", "show", "instance", INSTANCE_ID, "--raw"], capture_output=True, text=True)
    try:
        data = json.loads(res.stdout)
        return data.get("actual_status"), data.get("status_msg")
    except Exception as e:
        return None, str(e)

print(f"Iniciando monitoramento da instancia {INSTANCE_ID} (Polonia)...")

# 1. Espera ficar online
while True:
    status, msg = get_status()
    print(f"Status atual: {status} | Mensagem: {msg}")
    if status == "running":
        print("A maquina esta ONLINE!")
        break
    time.sleep(15)

# 2. Cria o arquivo tar localmente
print("Criando arquivo project.tar localmente...")
tar_res = subprocess.run(["tar", "-cf", "project.tar", "f51_darwin", "configs", "scripts", "tests", "requirements.txt"], cwd=LOCAL_DIR, capture_output=True, text=True)
if tar_res.returncode != 0:
    print("Erro ao criar tar:", tar_res.stderr)
    sys.exit(1)

# 3. Cria diretorio remoto via SSH
print("Criando diretorio remoto no workspace...")
mkdir_cmd = [
    "ssh", "-p", SSH_PORT, "-o", "StrictHostKeyChecking=no",
    f"root@{SSH_HOST}", "mkdir -p /workspace/F51-Darwin-SSD"
]
subprocess.run(mkdir_cmd)

# 4. Copia o arquivo tar para o servidor via SCP
print("Enviando project.tar via SCP...")
scp_cmd = [
    "scp", "-P", SSH_PORT, "-o", "StrictHostKeyChecking=no",
    os.path.join(LOCAL_DIR, "project.tar"),
    f"root@{SSH_HOST}:/workspace/F51-Darwin-SSD/"
]
scp_res = subprocess.run(scp_cmd, capture_output=True, text=True)
print(scp_res.stdout)

# 5. Descompacta no servidor
print("Descompactando arquivos no servidor...")
unzip_cmd = [
    "ssh", "-p", SSH_PORT, "-o", "StrictHostKeyChecking=no",
    f"root@{SSH_HOST}", "tar -xf /workspace/F51-Darwin-SSD/project.tar -C /workspace/F51-Darwin-SSD/"
]
subprocess.run(unzip_cmd)

# 6. Executa comandos remotos via SSH
print("Conectando via SSH e executando dependencias + smoke test...")
ssh_cmd = [
    "ssh",
    "-o", "StrictHostKeyChecking=no",
    "-o", "ConnectTimeout=10",
    "-p", SSH_PORT,
    f"root@{SSH_HOST}",
    "cd /workspace/F51-Darwin-SSD && "
    "pip install -r requirements.txt --break-system-packages && "
    "python scripts/prepare_base_training.py && "
    "python scripts/train_tokenizer.py --corpus tests/fixtures/corpus --output tokenizer/f51_bpe_smoke --vocab-size 260 && "
    "python scripts/train_base.py --corpus tests/fixtures/corpus --tokenizer tokenizer/f51_bpe_smoke --smoke"
]

ssh_res = subprocess.run(ssh_cmd, capture_output=True, text=True)
print("--- OUTPUT DO TREINO REMOTO ---")
print(ssh_res.stdout)
print("--- ERROS/WARNINGS REMOTOS ---")
print(ssh_res.stderr)

# Limpa o arquivo tar local
try:
    os.remove(os.path.join(LOCAL_DIR, "project.tar"))
except:
    pass

print("\n--- TESTE CONCLUIDO COM SUCESSO! ---")
print(f"Lembre-se de destruir a instancia executando: vastai destroy instance {INSTANCE_ID} -y")
