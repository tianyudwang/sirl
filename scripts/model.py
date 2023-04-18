import numpy as np  

import torch
import torch.nn as nn
import torch.nn.functional as F

import astar_pybind as pyAstar   # A*

from segnet import SegNet

class Astar(torch.autograd.Function):

    @staticmethod
    def forward(ctx, cost_batch, start_batch, goal_batch, batch_size, grid_size, num_controls):
        """
        Inputs: 
            cost_batch  (B, num_of_controls, map_size, map_size), a batch of cost funtion c(x, u)
            start_batch (B, 2), a batch of start location in 2D
            goal_batch  (B, 2), a batch of goal location in 2D
        Outputs: 
            q_val_batch (B, num_of_controls), Q values of the start, Q(start, u) 
        ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method
        """

        # Solve in batch
        # Initialize Q values and gradient dQdc
        
        Q_batch = np.zeros((batch_size, num_controls), dtype=np.float32)
        dQdc_batch = np.zeros((batch_size, num_controls, num_controls, 
                               grid_size, grid_size), dtype=np.float32)
        L = 1e6
        g_batch = np.ones((batch_size, grid_size, grid_size), dtype=np.float32) * L

        # move pytorch tensors to numpy arrays
        cost_batch = cost_batch.cpu().numpy()
        start_batch = start_batch.cpu().numpy()
        goal_batch = goal_batch.cpu().numpy()

        pyAstar.planBatch2DGrid(cost_batch, start_batch, goal_batch, Q_batch, dQdc_batch, g_batch)

        # move numpy arrays to pytorch tensors
        Q_batch = torch.from_numpy(Q_batch).cuda()
        dQdc_batch = torch.from_numpy(dQdc_batch).cuda()
        g_batch = torch.from_numpy(g_batch).cuda()
        ctx.save_for_backward(dQdc_batch)
        
        return Q_batch

    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        Gradients w.r.t. start and goal are None since we do not require them.
        """
        dQdc_batch, = ctx.saved_tensors
        return torch.einsum('ij,ijklm->iklm', grad_output, dQdc_batch), None, None, None, None, None



class SemIRLModel(nn.Module):
    def __init__(self, num_classes=4, num_controls=4, grid_size=16, batch_size=128):
        super().__init__()
        self.dtype = torch.float
        self.grid_size = grid_size
        self.batch_size = batch_size
        self.num_controls = num_controls

        # map encoder parameters
        self.theta = nn.Parameter(
            1*torch.ones((num_classes, grid_size, grid_size),
            dtype=self.dtype, device='cuda'))

        # cost encoder parameters
        self.cost_encoder = SegNet(num_classes, num_controls, grid_size)


    def map_encoder(self, grid_cnt_batch):
        """
        Encodes the lidar grid count to semantic map probability
        Inputs:
            grid_cnt_batch: (B x num_classes x grid_size x grid_size)
        Outputs:
            sem_prob_batch: (B x num_classes x grid_size x grid_size)
        """
        
        log_odds_batch = grid_cnt_batch * self.theta
        sem_prob_batch = F.softmax(log_odds_batch, dim=1)

        return sem_prob_batch 


    def astar(self, cost_batch, loc_batch, goal_batch):
        """
        A* planning  
        forward pass implemented in c++ and
        returns the optimal paths for analytic expression in custom backward function 
        Inputs:
            cost_batch: (B, num_controls, map_size, map_size), a batch of cost function c(x, u)
            start_batch: (B, 2), a batch of start coordinates in 2D
            goal_batch: (B, 2), a batch of goal coordinates in 2D 
        """
        return Astar.apply(cost_batch, loc_batch, goal_batch,
            self.batch_size, self.grid_size, self.num_controls)

    def forward(self, grid_cnt_batch, loc_batch, goal_batch):
        """
        grid_cnt_batch: (B x num_classes x grid_size x grid_size)
        loc_batch: (B x 2)
        goal_batch: (B x 2)
        
        """
        # map encoder
        sem_prob_batch = self.map_encoder(grid_cnt_batch)
        
        # cost encoder, add small constant to avoid negative cycles in planning
        cost_batch = self.cost_encoder(sem_prob_batch) + 1e-6

        # planner
        assert ((loc_batch >= 1).all() and (loc_batch < self.grid_size-1).all())
        assert ((goal_batch >= 1).all() and (goal_batch < self.grid_size-1).all())

        Q_batch = self.astar(cost_batch, loc_batch, goal_batch)
        logit = -Q_batch
        return logit, F.softmax(logit, dim=1)
