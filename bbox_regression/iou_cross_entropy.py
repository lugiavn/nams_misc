
# Notes

##########################################################################################
# faster-rcnn.pytorch/lib/model/utils/net_utils.py
# you want to switch from _smooth_l1_loss
# to nams_regression_loss and nams_regression_loss4rnn
##########################################################################################

def _smooth_l1_loss(bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights, sigma=1.0, dim=[1]):
    
    sigma_2 = sigma ** 2
    box_diff = bbox_pred - bbox_targets
    in_box_diff = bbox_inside_weights * box_diff
    abs_in_box_diff = torch.abs(in_box_diff)
    smoothL1_sign = (abs_in_box_diff < 1. / sigma_2).detach().float()
    in_loss_box = torch.pow(in_box_diff, 2) * (sigma_2 / 2.) * smoothL1_sign \
                  + (abs_in_box_diff - (0.5 / sigma_2)) * (1. - smoothL1_sign)
    out_loss_box = bbox_outside_weights * in_loss_box
    loss_box = out_loss_box
    for i in sorted(dim, reverse=True):
      loss_box = loss_box.sum(i)
    loss_box = loss_box.mean()
    return loss_box

def nams_regression_loss(bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights, sigma=1.0, dim=[1]):
    
    y = _smooth_l1_loss(bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights, sigma, dim)
    #print(bbox_pred.shape,bbox_inside_weights.shape)
    
    bbox_inside_weights_ = bbox_inside_weights.reshape(bbox_pred.shape[0], 9, 4, bbox_pred.shape[2], bbox_pred.shape[3])
    bbox_outside_weights_ = bbox_outside_weights.reshape(bbox_pred.shape[0], 9, 4, bbox_pred.shape[2], bbox_pred.shape[3])
    bbox_pred_ = bbox_pred.reshape(bbox_pred.shape[0], 9, 4, bbox_pred.shape[2], bbox_pred.shape[3])
    bbox_targets_ = bbox_targets.reshape(bbox_pred.shape[0], 9, 4, bbox_pred.shape[2], bbox_pred.shape[3])
    
    bbox_pred_x = bbox_pred_[:,:,0,:,:]
    bbox_pred_y = bbox_pred_[:,:,1,:,:]
    bbox_pred_w = torch.exp(bbox_pred_[:,:,2,:,:])
    bbox_pred_h = torch.exp(bbox_pred_[:,:,3,:,:])
    bbox_pred_top = bbox_pred_y - bbox_pred_h / 2.0
    bbox_pred_bottom = bbox_pred_y + bbox_pred_h / 2.0
    bbox_pred_left = bbox_pred_x - bbox_pred_w / 2.0
    bbox_pred_right = bbox_pred_x + bbox_pred_w / 2.0
    bbox_pred_area = bbox_pred_w * bbox_pred_h

    bbox_targets_x = bbox_targets_[:,:,0,:,:]
    bbox_targets_y = bbox_targets_[:,:,1,:,:]
    bbox_targets_w = torch.exp(bbox_targets_[:,:,2,:,:])
    bbox_targets_h = torch.exp(bbox_targets_[:,:,3,:,:])
    bbox_targets_top = bbox_targets_y - bbox_targets_h / 2.0
    bbox_targets_bottom = bbox_targets_y + bbox_targets_h / 2.0
    bbox_targets_left = bbox_targets_x - bbox_targets_w / 2.0
    bbox_targets_right = bbox_targets_x + bbox_targets_w / 2.0
    bbox_targets_area = bbox_targets_w * bbox_targets_h

    intersect_top = torch.max(bbox_targets_top, bbox_pred_top)
    intersect_bottom = torch.min(bbox_targets_bottom, bbox_pred_bottom)
    intersect_left = torch.max(bbox_targets_left, bbox_pred_left)
    intersect_right = torch.min(bbox_targets_right, bbox_pred_right)
    intersect_area = (intersect_bottom - intersect_top) * (intersect_right - intersect_left)

    iou = intersect_area / (bbox_pred_area + bbox_targets_area - intersect_area)
    iou = torch.max(iou, (iou * 0 + 0.1).detach())
    logiou = -torch.log(iou)
    x = logiou * bbox_inside_weights_[:,:,0,:,:]
    x = torch.sum(x / torch.sum(bbox_inside_weights_[:,:,0,:,:]))
    #print(iou[bbox_inside_weights_[:,:,0,:,:] > 0])
    #print float(x), float(y)
    return (x, y)




def nams_regression_loss4rnn(bbox_pred, rois_target, rois_inside_ws, rois_outside_ws, sigma=1.0, dim=[1]):
    
    y = _smooth_l1_loss(bbox_pred, rois_target, rois_inside_ws, rois_outside_ws, sigma, dim)
    #print(bbox_pred.shape,bbox_inside_weights.shape)
    
    if 1: # unnormalize
      rois_target /= 10
      rois_target[:,2:] *= 2
      bbox_pred /= 10
      bbox_pred[:,2:] *= 2
    
    bbox_pred_x = bbox_pred[:,0]
    bbox_pred_y = bbox_pred[:,1]
    bbox_pred_w = torch.exp(bbox_pred[:,2])
    bbox_pred_h = torch.exp(bbox_pred[:,3])
    bbox_pred_top = bbox_pred_y - bbox_pred_h / 2.0
    bbox_pred_bottom = bbox_pred_y + bbox_pred_h / 2.0
    bbox_pred_left = bbox_pred_x - bbox_pred_w / 2.0
    bbox_pred_right = bbox_pred_x + bbox_pred_w / 2.0
    bbox_pred_area = bbox_pred_w * bbox_pred_h

    rois_target_x = rois_target[:,0]
    rois_target_y = rois_target[:,1]
    rois_target_w = torch.exp(rois_target[:,2])
    rois_target_h = torch.exp(rois_target[:,3])
    rois_target_top = rois_target_y - rois_target_h / 2.0
    rois_target_bottom = rois_target_y + rois_target_h / 2.0
    rois_target_left = rois_target_x - rois_target_w / 2.0
    rois_target_right = rois_target_x + rois_target_w / 2.0
    rois_target_area = rois_target_w * rois_target_h


    intersect_top = torch.max(rois_target_top, bbox_pred_top)
    intersect_bottom = torch.min(rois_target_bottom, bbox_pred_bottom)
    intersect_left = torch.max(rois_target_left, bbox_pred_left)
    intersect_right = torch.min(rois_target_right, bbox_pred_right)
    intersect_area = (intersect_bottom - intersect_top) * (intersect_right - intersect_left)

    iou = intersect_area / (bbox_pred_area + rois_target_area - intersect_area)
    iou = torch.max(iou, (iou * 0 + 0.1).detach())
    logiou = -torch.log(iou)
    x = logiou * rois_inside_ws[:,0]
    x = torch.sum(x / torch.sum(rois_inside_ws[:,0]))

    return (x, y)
 
##########################################################################################
# change the composition of loss in trainval_net.py
##########################################################################################


      rpn_loss_box_nam, rpn_loss_box = rpn_loss_box
      RCNN_loss_bbox_nam, RCNN_loss_bbox = RCNN_loss_bbox
    
      if 1: # nams code
          losses = []
          losses += [('rpn_loss_cls', 1.0, rpn_loss_cls.mean())]
          losses += [('rpn_loss_box', 1.0, rpn_loss_box.mean())]
          losses += [('rpn_loss_box_nam', 0.0, rpn_loss_box_nam.mean())]
          losses += [('RCNN_loss_cls', 1.0, RCNN_loss_cls.mean())]
          losses += [('RCNN_loss_bbox', 0.0, RCNN_loss_bbox.mean())]
          losses += [('RCNN_loss_bbox_nam', 1.0, RCNN_loss_bbox_nam.mean())]

          loss = sum([loss_weight * loss_value for loss_name, loss_weight, loss_value in losses])
          losses += [('total training loss', None, loss)]
        
          # track losses
          for loss_name, loss_weight, loss_value in losses:
                if not losses_tracking.has_key(loss_name):
                    losses_tracking[loss_name] = []
                losses_tracking[loss_name].append(float(loss_value))
