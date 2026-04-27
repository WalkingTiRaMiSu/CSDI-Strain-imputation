CSDI Strain Global Mask Experiment
==================================

목적
----
- 1~18번 event로 학습한다.
- 입력 feature는 기본값 기준 [XTa, dXTa, ddXTa, STr]이다.
- 복원 target은 STr 하나만이다.
- XTa, dXTa, ddXTa는 결측시키지 않고 조건 정보로 사용한다.
- STr은 event 절대 시간축 기준 global 연속 block으로 결측시킨다.
- 그 다음 sliding window를 만든다. 즉 window마다 친절하게 center block을 새로 만드는 방식이 아니다.

실행 방법
---------
1) zip 압축을 풀어서 기존 csdi260427 폴더에 파일을 넣는다.
   기존 baseline 파일은 덮어쓰지 않는다.

2) 현재 폴더에 아래 파일들이 있어야 한다.
   - resp_total_re_05.mat
   - main_model.py
   - diff_models.py
   - exe_strain_global.py
   - dataset_strain_global.py
   - utils_strain_global.py
   - main_model_strain_global.py
   - base_strain_global.yaml

3) 실행:
   python exe_strain_global.py --config base_strain_global.yaml --device cuda:0 --epochs 700 --nsample 500

   더 오래/강하게:
   python exe_strain_global.py --config base_strain_global.yaml --device cuda:0 --epochs 900 --nsample 700

저장 결과
---------
- save/strain_global_YYYYMMDD_HHMMSS/
  - config_used.yaml
  - dataset_info_global.json
  - dataset_summary_global.json
  - train_scenarios_global.csv
  - valid_scenarios_global.csv
  - test_scenarios_global.csv
  - loss_history.csv
  - loss_curve.png
  - reconstruction_metrics_global.csv
  - plots_reconstruction_global/*.png
  - model.pth
  - model_best.pth

정규화
------
- feature별 z-score 정규화이다.
- mean/std는 1~18번 train event의 XTa peak 기반 active interval에서만 계산한다.
- 19~20번 test event의 통계값은 mean/std 계산에 사용하지 않는다.
- test의 STr 정답값은 CSDI 입력으로 들어가지 않고, metric/plot 비교용으로만 사용된다.
