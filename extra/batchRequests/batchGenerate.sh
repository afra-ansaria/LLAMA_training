#!/usr/bin/env bash
set -euo pipefail

NODE_IP="10.7.3.248"
PORT=8000
URL="http://${NODE_IP}:${PORT}/generate"

prompts=(
  "Explain quantum computing to a 10-year-old in ≤150 words."
  "Give three practical tips to optimize Python for numerical workloads."
  "What’s the difference between data, tensor, and pipeline parallelism? Keep it concise."
  "Summarize LayerNorm vs. RMSNorm and when to prefer each."
  "Describe a simple caching strategy for high-throughput LLM inference APIs."
)

echo 'id,prompt,response_text,response_json,ts' > results.csv

i=1
for p in "${prompts[@]}"; do
  payload=$(jq -n --arg prompt "$p" --argjson max_tokens 200 '{prompt:$prompt, max_tokens:$max_tokens}')
  resp=$(curl -sS -X POST "$URL" -H 'Content-Type: application/json' -d "$payload")

  # Try several JSON shapes to find a text field
  text=$(echo "$resp" | jq -r '
    if type=="object" and has("text") then .text
    elif type=="object" and has("response") then .response
    elif type=="object" and has("choices") and (.choices|type=="array") and (.choices|length>0) then
      ( .choices[0].text // .choices[0].message.content // (.choices[0]|tostring) )
    elif type=="object" and has("data") then
      ( ( .data.text // .data ) | tostring )
    else tostring end
  ' | tr -d '\n' )

  ts=$(date '+%Y-%m-%d %H:%M:%S')

  # Escape quotes for CSV
  esc_prompt=$(printf '%s' "$p" | sed 's/"/""/g')
  esc_text=$(printf '%s' "$text" | sed 's/"/""/g')
  esc_resp=$(printf '%s' "$resp" | tr -d '\n' | sed 's/"/""/g')

  mkdir -p results
  echo "$i,\"$esc_prompt\",\"$esc_text\",\"$esc_resp\",\"$ts\"" >> results/results.csv
  echo "[$i/${#prompts[@]}] OK"
  ((i++))
done

echo "Wrote results.csv"
