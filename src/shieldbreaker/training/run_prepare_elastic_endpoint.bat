@echo off
setlocal

set ROOT=%~dp0..\..\..
set SRC=%ROOT%\src\shieldbreaker
set PYTHON=python

set RAW_BASELINE_DIR=%ROOT%\runs\baseline
set DERIVED_DIR=%ROOT%\datasets\main\baseline
set EDR_JSONL=%DERIVED_DIR%\elastic_edr_only.jsonl
set ENDPOINT_TOKENS=%DERIVED_DIR%\elastic_endpoint_tokens.jsonl
set ENDPOINT_VOCAB=%DERIVED_DIR%\elastic_endpoint_vocab.json
set CORPUS_DIR=%ROOT%\datasets\lm_data_elastic_endpoint

if not exist "%DERIVED_DIR%" mkdir "%DERIVED_DIR%"

echo [1/3] Extract the four Elastic processes from full baseline logs
%PYTHON% "%SRC%\preprocess\extract_edr_processes.py" ^
  --input_dir "%RAW_BASELINE_DIR%" ^
  --pattern drakvuf.json ^
  --recursive ^
  --output_file "%EDR_JSONL%" ^
  --processes elastic-endpoint.exe elastic-agent.exe filebeat.exe metricbeat.exe
if errorlevel 1 exit /b 1

echo [2/3] Build endpoint-only tokens/vocab from the four-process extraction
%PYTHON% "%SRC%\preprocess\build_vocab.py" ^
  --input_path "%EDR_JSONL%" ^
  --out_tokens_path "%ENDPOINT_TOKENS%" ^
  --out_vocab_path "%ENDPOINT_VOCAB%" ^
  --processes elastic-endpoint.exe
if errorlevel 1 exit /b 1

echo [3/3] Build tokenizer and train/valid corpus
%PYTHON% "%SRC%\training\make_tokenizer_and_corpus.py" ^
  --tokens_path "%ENDPOINT_TOKENS%" ^
  --vocab_path "%ENDPOINT_VOCAB%" ^
  --out_dir "%CORPUS_DIR%" ^
  --segment_length 4096
if errorlevel 1 exit /b 1

echo.
echo Prepared endpoint-only corpus from four-process Elastic extraction.
echo Raw input:  %RAW_BASELINE_DIR%
echo 4-proc file:%EDR_JSONL%
echo Corpus dir: %CORPUS_DIR%
echo Vocab:      %ENDPOINT_VOCAB%

endlocal
