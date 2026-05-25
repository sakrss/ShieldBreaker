# Examples

Run commands from the repository root.

## Build Endpoint Corpus

```powershell
python src\leakedr\preprocess\extract_edr_processes.py --input_dir runs\baseline --pattern drakvuf.json --recursive --output_file datasets\main\baseline\elastic_edr_only.jsonl --processes elastic-endpoint.exe elastic-agent.exe filebeat.exe metricbeat.exe

python src\leakedr\preprocess\build_vocab.py --input_path datasets\main\baseline\elastic_edr_only.jsonl --out_tokens_path datasets\main\baseline\elastic_endpoint_tokens.jsonl --out_vocab_path datasets\main\baseline\elastic_endpoint_vocab.json --processes elastic-endpoint.exe

python src\leakedr\training\make_tokenizer_and_corpus.py --tokens_path datasets\main\baseline\elastic_endpoint_tokens.jsonl --vocab_path datasets\main\baseline\elastic_endpoint_vocab.json --out_dir datasets\lm_data_elastic_endpoint
```

## Train GPT Baseline

```powershell
python src\leakedr\training\train_gpt2.py --model_name_or_path gpt2 --tokenizer_name datasets\lm_data_elastic_endpoint --train_file datasets\lm_data_elastic_endpoint\train.txt --validation_file datasets\lm_data_elastic_endpoint\valid.txt --do_train --do_eval --output_dir models\out_elastic_endpoint_gpt2
```

## Export Event NLL Sequence

```powershell
python src\leakedr\analysis\export_endpoint_window_nll_matrix.py --root datasets\main\win10.elastic --model_dir models\out_elastic_endpoint_gpt2 --vocab_path datasets\main\baseline\elastic_endpoint_vocab.json --valid_txt_path datasets\lm_data_elastic_endpoint\valid.txt --prefix gpt2_medium
```

## Localize Multi-Scale Anomalies

```powershell
python src\leakedr\analysis\localize_multiscale_anomalies.py --product elastic --baseline datasets\main\baseline_sample\gpt2_medium_event_nll_sequence.json --stimulated datasets\main\stimulated_sample\gpt2_medium_event_nll_sequence.json --output_json runs\generated\elastic_multiscale_anomalies.json
```

