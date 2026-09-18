set -uo pipefail
cd /root/yslfalcon
export PYTHONPATH=$PWD/src HF_HOME=/root/.hf
PY=/root/benv/bin/python
echo pytest > status/run2
python3 -m uv pip install -q --python $PY pytest > logs/pytest_install.log 2>&1
$PY -m pytest -q tests/test_night.py > logs/pytest.log 2>&1 || echo pytest >> status/run2_errors
echo equal > status/run2
$PY /root/check_equal.py /root/bench_out/check_equal.json > logs/check_equal.log 2>&1 || echo equal >> status/run2_errors
echo benchmark > status/run2
nvidia-smi dmon -s pc -d 1 > logs/dmon3.log 2>&1 &
D=$!
$PY -m reid.benchmark --data-config /root/bench_data.yaml --config configs/exp043_soup_exp042_seeds.yaml --checkpoint /root/speed_ckpt/siglip2.pt --out /root/bench_out/after_fp16.json > logs/bench_after.log 2>&1 || echo benchmark >> status/run2_errors
kill $D
echo done > status/run2
