$ErrorActionPreference = "Stop"

$python = (Get-Command python).Source
$trainDir = $PSScriptRoot
$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..\..")
$trainScript = Join-Path $trainDir "train_gpt2.py"
$tokenizerDir = Join-Path $repoRoot "datasets\lm_data_elastic_endpoint"
$trainFile = Join-Path $tokenizerDir "train.txt"
$validFile = Join-Path $tokenizerDir "valid.txt"
$modelCheckpoint = Join-Path $repoRoot "models\out_elastic_endpoint_gpt2_xl_lowmem_v2_strictpath\checkpoint-2300"
$outputDir = Join-Path $repoRoot "models\out_elastic_endpoint_gpt2_xl_lowmem_v3_modelonly_continue"

New-Item -ItemType Directory -Force -Path $outputDir | Out-Null

$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$logPath = Join-Path $outputDir "scheduled_run_$stamp.log"
$latestStatusPath = Join-Path $outputDir "scheduled_run_latest.json"

$env:PYTHONUNBUFFERED = "1"

$args = @(
    $trainScript,
    "--model_name_or_path", $modelCheckpoint,
    "--tokenizer_name", $tokenizerDir,
    "--train_file", $trainFile,
    "--validation_file", $validFile,
    "--do_train",
    "--do_eval",
    "--block_size", "96",
    "--per_device_train_batch_size", "1",
    "--per_device_eval_batch_size", "1",
    "--gradient_accumulation_steps", "32",
    "--learning_rate", "2e-5",
    "--num_train_epochs", "20",
    "--eval_steps", "100",
    "--logging_steps", "50",
    "--save_steps", "100",
    "--fp16",
    "--optim", "adafactor",
    "--gradient_checkpointing",
    "--attn_implementation", "sdpa",
    "--early_stopping_patience", "3",
    "--early_stopping_threshold", "5e-4",
    "--output_dir", $outputDir
)

$startedAt = Get-Date
@{
    started_at = $startedAt.ToString("o")
    status = "running"
    log_path = $logPath
    output_dir = $outputDir
    model_checkpoint = $modelCheckpoint
} | ConvertTo-Json | Set-Content -Path $latestStatusPath -Encoding UTF8

"[$($startedAt.ToString('o'))] Starting GPT2-XL model-only continuation run" | Tee-Object -FilePath $logPath
"[$($startedAt.ToString('o'))] Output directory: $outputDir" | Tee-Object -FilePath $logPath -Append

& $python @args 2>&1 | Tee-Object -FilePath $logPath -Append
$exitCode = $LASTEXITCODE
$finishedAt = Get-Date

@{
    started_at = $startedAt.ToString("o")
    finished_at = $finishedAt.ToString("o")
    status = if ($exitCode -eq 0) { "finished" } else { "failed" }
    exit_code = $exitCode
    log_path = $logPath
    output_dir = $outputDir
    model_checkpoint = $modelCheckpoint
} | ConvertTo-Json | Set-Content -Path $latestStatusPath -Encoding UTF8

if ($exitCode -ne 0) {
    throw "GPT2-XL continuation run failed with exit code $exitCode"
}
