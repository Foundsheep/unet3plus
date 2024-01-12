import torch


def iou_loss(pred, target):
    assert len(pred.shape) == 4
    batch_size = pred.shape[0]

    # pred is now probabilities
    pred = torch.round(pred)

    intersection = pred * target
    union = pred + target - intersection
    iou_over_batches = torch.sum(intersection, [1, 2, 3]) / torch.sum(union, [1, 2, 3])
    #     iou = torch.mean(iou_over_batches)
    #     return iou
    return iou_over_batches


def ms_ssim_loss(pred, target, beta_m=1, gamma_m=1):
    assert len(pred.shape) == 4
    batch_size = pred.shape[0]

    # pred is now probabilities
    pred = torch.round(pred)

    pred = pred.squeeze(1)
    target = target.squeeze(1)

    mu_p = torch.mean(pred, dim=[1, 2])
    mu_g = torch.mean(target, dim=[1, 2])
    sigma_p = torch.std(pred, dim=[1, 2])
    sigma_g = torch.std(target, dim=[1, 2])

    # get covariance - first flatten the patch to pass on to torch.cov() as a matrix of two variables
    flatten_p = torch.flatten(pred, start_dim=1, end_dim=2)  # (B, -1)
    flatten_g = torch.flatten(target, start_dim=1, end_dim=2)  # (B, -1)
    cov_input = torch.stack([flatten_p, flatten_g], dim=1)  # (B, 2, -1)

    covs = []
    for inp in cov_input:
        cov_matrix = torch.cov(inp)
        cov = cov_matrix[0, 1]  # extract covariance from 2 X 2 matrix
        covs.append(cov)
    sigma_pg = torch.stack(covs)

    C1 = torch.Tensor([0.01 ** 2])
    C2 = torch.Tensor([0.03 ** 2])

    # multiply for batches
    first_component = ((2 * mu_p * mu_g + C1) / (mu_p ** 2 + mu_g ** 2 + C1)) ** beta_m
    second_component = ((2 * sigma_pg + C2) / (sigma_p ** 2 + sigma_g ** 2 + C2)) ** gamma_m
    result = first_component * second_component

    return result


def total_loss(pred_list, target_list, beta_m_list=[1] * 5, gamma_m_list=[1] * 5):
    loss_1 = 0.0
    loss_2 = 1.0
    for pred, target, beta_m, gamma_m in zip(pred_list, target_list, beta_m_list, gamma_m_list):
        loss_1 += iou_loss(pred, target)
        loss_2 *= ms_ssim_loss(pred, target, beta_m, gamma_m)

    loss_2 = 1 - loss_2
    loss_1_over_batches = torch.sum(loss_1)
    loss_2_over_batches = torch.sum(loss_2)

    batch_size = len(loss_1)
    result = (loss_1_over_batches + loss_2_over_batches) / batch_size
    result = result / len(pred_list)
    return result
