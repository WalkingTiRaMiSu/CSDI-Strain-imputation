CSDI STr reconstruction - multi-scenario training + fixed peak-center 15% test

핵심 의도
- 학습: 1~18번 지진 데이터에 대해 다양한 결측 시나리오를 모두 섞어서 학습한다.
  * peak 근처 연속결측: peak_before / peak_center / peak_after
  * peak에서 먼 연속결측: far_before / far_after
  * active interval 내부 위치별 연속결측: active_left / active_middle / active_right
  * 두 덩어리 연속결측: double_around_peak / double_far / double_random
  * 점식 랜덤결측: random_point
  * 결측 비율: active interval 기준 5%, 10%, 15%
  * jitter repeat 적용으로 데이터셋 수를 늘림

- 테스트/최종 추론: 19~20번 지진 데이터에 대해 peak_center 15% 연속결측만 고정으로 평가한다.
  * test_missing_ratios: [0.15]
  * test_missing_scenarios: single_block / peak_center
  * 발표에서는 "학습은 다양한 결측 상황에 대응하도록 구성했고, 최종 추론 평가는 피크 중심 15% 연속결측에 대해 수행했다"고 설명하면 됨.

입력 시나리오 3개
1) base_input_accg_daccg_xta_dxta_str.yaml
   feature_set: acc_g, dacc_g, XTa, dXTa, STr + time_norm

2) base_input_accg_daccg_xta_str.yaml
   feature_set: acc_g, dacc_g, XTa, STr + time_norm

3) base_input_accg_daccg_ddaccg_xta_dxta_ddxta_str.yaml
   feature_set: acc_g, dacc_g, ddacc_g, XTa, dXTa, ddXTa, STr + time_norm

실행 예시
python exe_strain_multiscenario.py --config base_input_accg_daccg_xta_dxta_str.yaml --device cuda:0 --epochs 50 --nsample 50
python exe_strain_multiscenario.py --config base_input_accg_daccg_xta_str.yaml --device cuda:0 --epochs 50 --nsample 50
python exe_strain_multiscenario.py --config base_input_accg_daccg_ddaccg_xta_dxta_ddxta_str.yaml --device cuda:0 --epochs 50 --nsample 50

먼저 데이터셋 개수만 확인
python exe_strain_multiscenario.py --config base_input_accg_daccg_xta_dxta_str.yaml --device cuda:0 --dryrun_dataset

결과 저장
save/strain_multi_<run_name>_<timestamp>/

주의
- STr의 결측 구간 정답은 모델 입력에 들어가지 않음.
- 정답 STr은 학습 loss 계산 및 테스트 후 metric/plot 비교에만 사용됨.
- CSDI 모델 구조는 유지하고, 입력 feature와 masking scenario/data 구성만 바꾼 버전임.
