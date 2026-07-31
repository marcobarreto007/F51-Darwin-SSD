@echo off
cd /d C:\Users\marco\Desktop\F51-Darwin-SSD
echo Iniciando treino F51 Darwin v2...
echo Data: %date% %time%
echo.

python -u scripts/train_cloud.py ^
  --device cuda ^
  --batch-size 22 ^
  --block-size 512 ^
  --steps 100000 ^
  --save-every 2500 ^
  --eval-every 500 ^
  --lr 3e-4 ^
  --corpus data/corpus ^
  --tokenizer tokenizer/f51_bpe ^
  --checkpoint-dir checkpoints/organism ^
  --resume checkpoints/organism/organism_247/step_0010000.pt ^
  --no-weights ^
  > logs\train_organism_v2.log 2>&1

echo Treino finalizado ou erro. Verifique logs\train_organism_v2.log
pause
