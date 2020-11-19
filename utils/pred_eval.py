from typing import Dict, List, Optional, Tuple
import math
import numpy as np
def get_ade(forecasted_trajectory: np.ndarray, gt_trajectory: np.ndarray) -> float:
    """Compute Average Displacement Error.

    Args:
        forecasted_trajectory: Predicted trajectory with shape (N x pred_len x 2)
        gt_trajectory: Ground truth trajectory with shape (N x pred_len x 2)

    Returns:
        ade: Average Displacement Error

    """
    forecasted_trajectory = forecasted_trajectory.reshape(-1,2)
    gt_trajectory = gt_trajectory.reshape(-1,2)

    error = (forecasted_trajectory - gt_trajectory)**2
    error = np.sum(error, axis = 1)
    error = np.mean(error**0.5)
    return error

def get_fde(forecasted_trajectory: np.ndarray, gt_trajectory: np.ndarray) -> float:
    """Compute Final Displacement Error.

    Args:
        forecasted_trajectory: Predicted trajectory with shape (N x pred_len x 2)
        gt_trajectory: Ground truth trajectory with shape (N x pred_len x 2)

    Returns:
        fde: Final Displacement Error

    """
    forecasted_trajectory = forecasted_trajectory[:,-1,:]
    gt_trajectory = gt_trajectory[:, -1, :]

    error = (forecasted_trajectory - gt_trajectory)**2
    error = np.sum(error, axis = 1)
    error = np.mean(error**0.5)
    return error



def get_displacement_errors(forecasted_trajectory, gt_trajectory):
    """Compute Final Displacement Error.

    Args:
        forecasted_trajectory: Predicted trajectory with shape (bt_sz x pred_len x 2)
        gt_trajectory: Ground truth trajectory with shape (bt_sz x pred_len x 2)

    Returns:
        fde: Final Displacement Error
        ade 
    """    

    ade = get_ade(forecasted_trajectory, gt_trajectory)
    fde = get_fde(forecasted_trajectory, gt_trajectory)

    return ade, fde
