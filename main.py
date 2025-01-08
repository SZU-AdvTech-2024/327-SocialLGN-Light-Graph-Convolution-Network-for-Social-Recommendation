import time
from os.path import join

import torch

import Procedure
import register
import utils
import world
from register import dataset
from torch.utils.tensorboard import SummaryWriter 
from tqdm import tqdm
import os 


# ==============================
utils.set_seed(world.seed)
print(">>SEED:", world.seed)
# ==============================
torch.autograd.set_detect_anomaly(True)
Recmodel = register.MODELS[world.model_name](world.config, dataset)
Recmodel = Recmodel.to(world.device)
bpr = utils.BPRLoss(Recmodel, world.config)

# weight_file = utils.getFileName()
# print(f"load and save to {weight_file}")
# if world.LOAD:
#     try:
#         Recmodel.load_state_dict(torch.load(weight_file, map_location=torch.device('cpu')))
#         print(f"loaded model weights from {weight_file}")
#     except FileNotFoundError:
#         print(f"{weight_file} not exists, start from beginning")

# try to use tensorboard 
writer : SummaryWriter = SummaryWriter()

best_ndcg, best_recall, best_pre = 0, 0, 0
best_ndcg_cold, best_recall_cold, best_pre_cold = 0, 0, 0
low_count, low_count_cold = 0, 0

# ========================= methods to load and save checkpoints 
def save_checkpoint_with_bprloss(model, bpr_loss, epoch,best_ndcg, filepath):
    """保存模型和 BPRLoss（包含优化器状态）"""
    torch.save({
        'epoch': epoch,
        'model_state': model.state_dict(),
        'optimizer_state': bpr_loss.opt.state_dict(),
        'weight_decay': bpr_loss.weight_decay,
        'lr': bpr_loss.lr,
        'best_ndcg': best_ndcg
    }, filepath)
    # print(f"Checkpoint saved to {filepath}")
    

def load_checkpoint_with_bprloss(filepath, model, bpr_loss):
    """加载模型和 BPRLoss（包含优化器状态）"""
    if os.path.exists(filepath):
        checkpoint = torch.load(filepath)
        model.load_state_dict(checkpoint['model_state'])
        bpr_loss.opt.load_state_dict(checkpoint['optimizer_state'])
        
        # 恢复 BPRLoss 的属性
        bpr_loss.weight_decay = checkpoint.get('weight_decay', bpr_loss.weight_decay)
        bpr_loss.lr = checkpoint.get('lr', bpr_loss.lr)
        best_metric = checkpoint.get('best_ndcg')
        print(f"!!!!!!!!!!! loaded best metric:{best_metric}")
        
        print(f"Checkpoint loaded from {filepath}")
        return checkpoint['epoch'], best_metric
    else:
        print(f"No checkpoint found at {filepath}")
        return 0,0  # 默认从头开始
# =========================

checkpoint_dir = "./checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)
latest_checkpoint_path = os.path.join(checkpoint_dir, "latest_checkpoint.pth")
best_checkpoint_path = os.path.join(checkpoint_dir, "best_checkpoint.pth")



# recover from the latest 
start_epoch = -1
if os.path.exists(latest_checkpoint_path):
    start_epoch, best_ndcg = load_checkpoint_with_bprloss(latest_checkpoint_path, Recmodel, bpr) 
    print(f"Resumed from latest checkpoint at epoch {start_epoch}.")

try:
    stop_training = False 
    for epoch in tqdm(range(start_epoch + 1, world.TRAIN_epochs + 1)):
        if stop_training:
            break 
        start = time.time()
        
        # train at least once for all metric needed to be calculated 
        
        loss = Procedure.BPR_train_original(dataset, Recmodel, bpr, epoch, writer)
        
        if epoch % 10 == 0 or epoch == world.TRAIN_epochs:
            results = Procedure.Test(dataset, Recmodel, epoch, False, writer)
            results_cold = Procedure.Test(dataset, Recmodel, epoch, True, writer)
            if results['ndcg'][0] < best_ndcg:
                # 记录 效果变差的 epoch数量 （这里不管冷启动）
                low_count += 1
                if low_count == 30: 
                    # 如果连续30次都变差
                    if epoch > 1000:
                        # 如果已经训练了超过1000轮，停止训练 
                        stop_training = True
                        # break 
                    
                    else:
                        # 否则可能是因为训练的还太少，不稳定，继续训练 
                        low_count = 0
            else:
                # 如果效果变好了，更新best 
                best_recall = results['recall'][0]
                best_ndcg = results['ndcg'][0]
                best_pre = results['precision'][0]
                low_count = 0
                
                # save best model 
                save_checkpoint_with_bprloss(Recmodel, bpr, epoch, best_ndcg, best_checkpoint_path)

            if results_cold['ndcg'][0] > best_ndcg_cold:
                best_recall_cold = results_cold['recall'][0]
                best_ndcg_cold = results_cold['ndcg'][0]
                best_pre_cold = results_cold['precision'][0]
                low_count_cold = 0
        save_checkpoint_with_bprloss(Recmodel, bpr, epoch, best_ndcg, latest_checkpoint_path)    
finally:
    print(f"\nbest recall at 10:{best_recall}")
    print(f"best ndcg at 10:{best_ndcg}")
    print(f"best precision at 10:{best_pre}")
    print(f"\ncold best recall at 10:{best_recall_cold}")
    print(f"cold best ndcg at 10:{best_ndcg_cold}")
    print(f"cold best precision at 10:{best_pre_cold}")
    if writer is not None: 
        writer.close() 
