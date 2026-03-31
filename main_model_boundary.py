import torch
from main_model import CSDI_Custom


class CSDI_Custom_Boundary(CSDI_Custom):
    """
    CSDI_Custom 기반 커스텀 모델
    - 학습 시 gt_mask(= ch8 block missing)를 그대로 cond_mask로 사용
    - 평가 시 생성된 샘플의 결측 블록 시작/끝을 관측 경계값에 정확히 맞춤
    """

    def __init__(self, config, device, target_dim=1):
        super().__init__(config, device, target_dim=target_dim)

    def forward(self, batch, is_train=1):
        (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            _,
            cut_length,
        ) = self.process_data(batch)

        # 핵심: train/valid 모두 gt_mask를 그대로 사용
        # 즉 random masking이 아니라 ch8 block missing만 타깃으로 학습
        cond_mask = gt_mask

        side_info = self.get_side_info(observed_tp, cond_mask)

        loss_func = self.calc_loss if is_train == 1 else self.calc_loss_valid
        return loss_func(observed_data, cond_mask, observed_mask, side_info, is_train)

    def _find_segments(self, mask_1d):
        """
        mask_1d: bool tensor/list, True = missing 위치
        contiguous segment 리스트 반환
        """
        segs = []
        idx = torch.where(mask_1d)[0]
        if len(idx) == 0:
            return segs

        start = idx[0].item()
        prev = idx[0].item()

        for i in idx[1:]:
            cur = i.item()
            if cur == prev + 1:
                prev = cur
            else:
                segs.append((start, prev))
                start = cur
                prev = cur
        segs.append((start, prev))
        return segs

    def _anchor_missing_blocks(self, samples, observed_data, gt_mask, observed_mask):
        """
        samples: (B, n_samples, K, L)
        observed_data: (B, K, L)    # normalized full data
        gt_mask: (B, K, L)          # 1=given, 0=missing
        observed_mask: (B, K, L)    # 여기선 거의 전부 1

        목적:
        - 결측 블록의 첫/끝 점을 경계 관측값과 정확히 일치시키기
        - 블록 내부는 affine transform으로 부드럽게 이동
        """
        B, N, K, L = samples.shape
        target_mask = (observed_mask - gt_mask) > 0  # True = missing

        anchored = samples.clone()

        for b in range(B):
            for k in range(K):
                miss_mask_1d = target_mask[b, k]  # (L,)
                segments = self._find_segments(miss_mask_1d)

                for (s, e) in segments:
                    left_exists = s - 1 >= 0 and gt_mask[b, k, s - 1] > 0
                    right_exists = e + 1 < L and gt_mask[b, k, e + 1] > 0

                    left_val = observed_data[b, k, s - 1] if left_exists else None
                    right_val = observed_data[b, k, e + 1] if right_exists else None

                    for n in range(N):
                        block = anchored[b, n, k, s:e + 1]  # (len,)

                        if len(block) == 0:
                            continue

                        # case 1: 양쪽 경계 다 있으면 시작/끝 정확히 맞춤
                        if left_exists and right_exists:
                            old_start = block[0]
                            old_end = block[-1]

                            # block이 거의 평평하면 선형보간으로 대체
                            if torch.abs(old_end - old_start) < 1e-8:
                                new_block = torch.linspace(
                                    left_val.item(),
                                    right_val.item(),
                                    steps=len(block),
                                    device=block.device,
                                    dtype=block.dtype,
                                )
                            else:
                                # affine transform: new = a*old + b
                                a = (right_val - left_val) / (old_end - old_start)
                                b_shift = left_val - a * old_start
                                new_block = a * block + b_shift

                            # 시작/끝 완전 강제 일치
                            new_block[0] = left_val
                            new_block[-1] = right_val
                            anchored[b, n, k, s:e + 1] = new_block

                        # case 2: 왼쪽 경계만 있으면 시작값만 맞춤
                        elif left_exists and not right_exists:
                            shift = left_val - block[0]
                            new_block = block + shift
                            new_block[0] = left_val
                            anchored[b, n, k, s:e + 1] = new_block

                        # case 3: 오른쪽 경계만 있으면 끝값만 맞춤
                        elif right_exists and not left_exists:
                            shift = right_val - block[-1]
                            new_block = block + shift
                            new_block[-1] = right_val
                            anchored[b, n, k, s:e + 1] = new_block

                        # case 4: 양쪽 경계 모두 없으면 그대로 둠
                        else:
                            pass

        return anchored

    def evaluate(self, batch, n_samples):
        (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            _,
            cut_length,
        ) = self.process_data(batch)

        with torch.no_grad():
            cond_mask = gt_mask
            target_mask = observed_mask - cond_mask

            side_info = self.get_side_info(observed_tp, cond_mask)
            samples = self.impute(observed_data, cond_mask, side_info, n_samples)

            # 핵심: 생성된 샘플을 경계에 정확히 붙이도록 후처리
            samples = self._anchor_missing_blocks(
                samples=samples,
                observed_data=observed_data,
                gt_mask=gt_mask,
                observed_mask=observed_mask,
            )

            for i in range(len(cut_length)):
                target_mask[i, ..., 0:cut_length[i].item()] = 0

        return samples, observed_data, target_mask, observed_mask, observed_tp