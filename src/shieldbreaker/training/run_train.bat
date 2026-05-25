@echo off
REM ==========================================
REM GPT-2 / GPT-2-XL Perplexity Curve Training
REM Records:
REM  - Validation Perplexity vs Step
REM  - Training Time
REM  - GPU Memory Usage
REM ==========================================

set ROOT=%~dp0..\..\..
python "%~dp0train_gpt2.py" ^
  --model_name_or_path "%ROOT%\models\pretrained\gpt2-xl" ^
  --tokenizer_name "%ROOT%\datasets\lm_data" ^
  --train_file "%ROOT%\datasets\lm_data\train.txt" ^
  --validation_file "%ROOT%\datasets\lm_data\valid.txt" ^
  --do_train ^
  --do_eval ^
  --block_size 256 ^
  --per_device_train_batch_size 1 ^
  --per_device_eval_batch_size 1 ^
  --gradient_accumulation_steps 16 ^
  --learning_rate 2e-5 ^
  --num_train_epochs 5 ^
  --eval_steps 50 ^
  --logging_steps 100 ^
  --save_steps 1000 ^
  --fp16 ^
  --output_dir "%ROOT%\models\out_gpt2_xl"

pause
