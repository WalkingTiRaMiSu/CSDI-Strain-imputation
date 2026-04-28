# main_model_strain_multiscenario.py
# Vanilla CSDI_Physio를 그대로 사용하되, target_dim만 feature 수에 맞게 받는 얇은 wrapper입니다.
# 모델 구조/forward/impute/loss는 main_model.py의 CSDI_Physio 그대로입니다.

from main_model import CSDI_Physio


class CSDI_Str_MultiScenario(CSDI_Physio):
    def __init__(self, config, device, target_dim):
        super().__init__(config, device, target_dim=target_dim)
