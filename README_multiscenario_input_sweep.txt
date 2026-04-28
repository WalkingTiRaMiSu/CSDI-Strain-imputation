CSDI STr 복원 - multi-scenario input sweep 버전

1) 기존 baseline/global mask 코드는 덮어쓰지 않습니다.
   이 폴더 안의 새 파일명은 모두 *_multiscenario.py, base_input_*.yaml 형식입니다.

2) 실행 전 확인
   - resp_total_re_05.mat 파일이 현재 작업 폴더에 있어야 합니다.
   - 기존 CSDI 파일 main_model.py, diff_models.py가 같은 폴더에 있어야 합니다.
   - 이 zip 안의 파일들을 기존 폴더에 복사해도 기존 exe_strain_global.py 등은 건드리지 않습니다.

3) 데이터 구성
   - 훈련: EQ 1~18
   - 테스트: EQ 19~20
   - target: STr
   - 결측은 STr에만 적용합니다.
   - acc_g / XTa / 미분항 / time_norm은 조건 입력으로 관측된 상태입니다.
   - scaling은 train event의 XTa peak 기준 active interval에서만 mean/std를 계산하는 feature별 z-score입니다.

4) 학습 결측 시나리오
   - peak 근처 연속 block: peak_before, peak_center, peak_after
   - peak에서 먼 연속 block: far_before, far_after
   - active interval 내부 위치별 block: active_left, active_middle, active_right
   - 두 번으로 나누어진 block: double_around_peak, double_far, double_random
   - 점식 random missing: random_point
   - missing ratio: active interval 대비 5%, 10%, 15%
   - jitter repeat: 2회
   - 전체 input-output pair 수는 실행 시 [INFO] samples train/valid/test에 표시됩니다. 목표는 약 4000개 이상입니다.

5) 입력 feature 조합
   A. base_input_accg_daccg_xta_dxta_str.yaml
      acc_g, dacc_g, XTa, dXTa, time_norm, STr

   B. base_input_accg_daccg_xta_str.yaml
      acc_g, dacc_g, XTa, time_norm, STr

   C. base_input_accg_daccg_ddaccg_xta_dxta_ddxta_str.yaml
      acc_g, dacc_g, ddacc_g, XTa, dXTa, ddXTa, time_norm, STr

   include_time_feature: true이므로 time_norm은 STr 앞에 자동 추가됩니다.
   CSDI 자체 time embedding도 window sample index를 통해 그대로 사용됩니다.

6) 빠른 데이터셋 개수 확인
   python exe_strain_multiscenario.py --config base_input_accg_daccg_xta_dxta_str.yaml --device cuda:0 --dryrun_dataset

7) 한 개 조합 실행
   python exe_strain_multiscenario.py --config base_input_accg_daccg_xta_dxta_str.yaml --device cuda:0 --epochs 50 --nsample 50

8) 세 개 조합 순차 실행
   run_three_input_scenarios.bat 더블클릭 또는 프롬프트에서 실행

9) 결과 저장 위치
   save/strain_multi_<run_name>_<timestamp>/
   - loss_curve.png
   - loss_history.csv
   - dataset_info.json
   - dataset_summary.json
   - train_scenarios.csv / valid_scenarios.csv / test_scenarios.csv
   - reconstruction_metrics_multiscenario.csv
   - plots_reconstruction_multiscenario/

10) 그림 출력
   기본값은 그림이 너무 많이 생기지 않도록 EQ19/EQ20의 peak_center, ratio=0.10만 그림으로 저장합니다.
   모든 테스트 시나리오 그림이 필요하면 yaml에서 eval.plot_all_test_scenarios: true 로 바꾸면 됩니다.
