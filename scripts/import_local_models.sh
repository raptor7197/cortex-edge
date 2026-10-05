#!/bin/bash
# Import the GGUF weights that already exist on this machine into Ollama.
# No downloads: every model below is built FROM a local .gguf file.
#
#   bash scripts/import_local_models.sh
#
# Sources (see docs/LOCAL_LATENCY.md):
#   /home/krxsna/dev/caamas/models/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf
#   /home/krxsna/dev/caamas/models/gemma-2-2b-it-Q4_K_M.gguf
#   /home/krxsna/dev/caamas/models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf
#   /home/krxsna/dev/caamas/models/nomic-embed-text-v1.5.Q4_K_M.gguf
set -u

MODEL_DIR="${MODEL_DIR:-/home/krxsna/dev/caamas/models}"
WORK="${HOME}/.cortexedge-modelfiles"
mkdir -p "${WORK}"

create_from_gguf() {
  local name="$1" gguf="$2" template="$3" extra="$4"
  if [ ! -f "${gguf}" ]; then
    echo "skip ${name}: missing ${gguf}"
    return 0
  fi
  if ollama list 2>/dev/null | awk '{print $1}' | grep -qx "${name}"; then
    echo "exists ${name}"
    return 0
  fi
  local mf="${WORK}/Modelfile.${name//[:\/]/_}"
  {
    echo "FROM ${gguf}"
    if [ -n "${template}" ]; then
      printf 'TEMPLATE """%s"""\n' "${template}"
    fi
    echo 'PARAMETER temperature 0.2'
    echo 'PARAMETER num_ctx 4096'
    if [ -n "${extra}" ]; then
      echo "${extra}"
    fi
  } > "${mf}"
  echo "creating ${name} from ${gguf}"
  ollama create "${name}" -f "${mf}"
}

QWEN_TMPL='{{ if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{ if .Prompt }}<|im_start|>user
{{ .Prompt }}<|im_end|>
{{ end }}<|im_start|>assistant
'
GEMMA_TMPL='{{ if .System }}<start_of_turn>user
{{ .System }}<end_of_turn>
{{ end }}{{ if .Prompt }}<start_of_turn>user
{{ .Prompt }}<end_of_turn>
{{ end }}<start_of_turn>model
'
LLAMA_TMPL='{{ if .System }}<|start_header_id|>system<|end_header_id|>

{{ .System }}<|eot_id|>{{ end }}{{ if .Prompt }}<|start_header_id|>user<|end_header_id|>

{{ .Prompt }}<|eot_id|>{{ end }}<|start_header_id|>assistant<|end_header_id|>

'

create_from_gguf "qwen2.5:1.5b"  "${MODEL_DIR}/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf" "${QWEN_TMPL}"   'PARAMETER stop "<|im_end|>"'
create_from_gguf "gemma2:2b"     "${MODEL_DIR}/gemma-2-2b-it-Q4_K_M.gguf"         "${GEMMA_TMPL}"  'PARAMETER stop "<end_of_turn>"'
create_from_gguf "llama3.1:8b"   "${MODEL_DIR}/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf" "${LLAMA_TMPL}" 'PARAMETER stop "<|eot_id|>"'
create_from_gguf "nomic-embed-text" "${MODEL_DIR}/nomic-embed-text-v1.5.Q4_K_M.gguf" "" ""

echo "--- ollama list ---"
ollama list
