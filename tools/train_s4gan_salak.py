"""
S4GAN Training Script for Salak Dataset
With Wandb Integration and Train/Val mIoU Tracking

Usage:
    C:\\Users\\IoT-C504-03\\miniconda3\\envs\\als4gan_env\\python.exe tools\\train_s4gan_salak.py
"""

import argparse
import os
import numpy as np
import timeit
import sys

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils import data
from torch.autograd import Variable
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import *
from model.discriminator import s4GAN_discriminator
from data.salak_dataset import SalakDataSet
from utils.loss import CrossEntropy2d
from utils.lr_scheduler import PolynomialLR
from utils.metric import scores

# Wandb for experiment tracking
import wandb

start = timeit.default_timer()

IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)

# Default hyperparameters
INPUT_SIZE = '256,256'
IGNORE_LABEL = 255
BATCH_SIZE = 4
NUM_STEPS = 50000
SAVE_PRED_EVERY = 5000
SAVE_LATEST_EVERY = 100  # Save latest model every N iterations
EVAL_EVERY = 1000
LEARNING_RATE = 2.5e-4
LEARNING_RATE_D = 1e-4
POWER = 0.9
LR_DECAY = 10
WEIGHT_DECAY = 0.0005
ITER_MAX = 20000
MOMENTUM = 0.9
RANDOM_SEED = 5000
LAMBDA_FM = 0.1
LAMBDA_ST = 1.0


def get_arguments():
    """Parse all the arguments provided from the CLI."""
    parser = argparse.ArgumentParser(description="S4GAN Training for Salak Dataset")
    
    # Dataset
    parser.add_argument("--data-root", type=str, required=True,
                        help="Path to the salak dataset root (contains salak-1-* folders)")
    parser.add_argument("--labeled-list", type=str, default=None,
                        help="Path to the file listing labeled images (optional - if not provided, loads all patches)")
    parser.add_argument("--class-mapping", type=str, required=True,
                        help="Path to class_mapping.csv")
    parser.add_argument("--num-classes", type=int, default=7,
                        help="Number of classes (including background)")
    
    # Training
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Batch size")
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS,
                        help="Number of training iterations (default: 50000)")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                        help="Learning rate for generator")
    parser.add_argument("--learning-rate-D", type=float, default=LEARNING_RATE_D,
                        help="Learning rate for discriminator")
    parser.add_argument("--power", type=float, default=POWER,
                        help="Power for polynomial learning rate decay")
    parser.add_argument("--threshold-st", type=float, default=0.2,
                        help="Self-training confidence threshold")
    parser.add_argument("--lambda-fm", type=float, default=LAMBDA_FM,
                        help="Weight for feature matching loss")
    parser.add_argument("--lambda-st", type=float, default=LAMBDA_ST,
                        help="Weight for self-training loss")
    parser.add_argument("--val-split", type=float, default=0.2,
                        help="Validation split ratio (0.0-1.0)")
    
    # Checkpoints
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints",
                        help="Directory to save checkpoints")
    parser.add_argument("--restore-from", type=str, default=None,
                        help="Path to restore checkpoint from")
    parser.add_argument("--save-pred-every", type=int, default=SAVE_PRED_EVERY,
                        help="Save checkpoint every N iterations")
    parser.add_argument("--save-latest-every", type=int, default=SAVE_LATEST_EVERY,
                        help="Save latest model every N iterations")
    parser.add_argument("--eval-every", type=int, default=EVAL_EVERY,
                        help="Evaluate on validation set every N iterations")
    
    # Wandb
    parser.add_argument("--wandb-project", type=str, default="als4gan-salak",
                        help="Wandb project name")
    parser.add_argument("--wandb-run-name", type=str, default=None,
                        help="Wandb run name")
    parser.add_argument("--no-wandb", action="store_true",
                        help="Disable wandb logging")
    
    # Other
    parser.add_argument("--input-size", type=str, default=INPUT_SIZE,
                        help="Input size (height,width)")
    parser.add_argument("--random-mirror", action="store_true",
                        help="Random horizontal flip")
    parser.add_argument("--random-scale", action="store_true",
                        help="Random scaling")
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED,
                        help="Random seed")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="Ignore label value")
    
    return parser.parse_args()


def get_device(cuda=True):
    """Get CUDA device if available"""
    cuda = cuda and torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    if cuda:
        print("Device:")
        for i in range(torch.cuda.device_count()):
            print(f"    {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("Device: CPU")
    return device


def loss_calc(pred, label, device, ignore_label=255):
    """Calculate cross-entropy loss"""
    label = Variable(label.long()).to(device)
    criterion = CrossEntropy2d(ignore_label=ignore_label).to(device)
    return criterion(pred, label)


def adjust_learning_rate_D(optimizer, i_iter, args):
    """Polynomial learning rate decay for discriminator"""
    lr = args.learning_rate_D * ((1 - float(i_iter) / args.num_steps) ** args.power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10
    return lr


def one_hot(label, num_classes):
    """Convert label to one-hot encoding"""
    label = label.numpy()
    one_hot_encoded = np.zeros((label.shape[0], num_classes, label.shape[1], label.shape[2]), dtype=label.dtype)
    for i in range(num_classes):
        one_hot_encoded[:, i, ...] = (label == i)
    return torch.FloatTensor(one_hot_encoded)


def compute_argmax_map(output):
    """Get class predictions from softmax output"""
    output = output.detach().cpu().numpy()
    output = output.transpose((1, 2, 0))
    output = np.asarray(np.argmax(output, axis=2), dtype=np.int32)
    output = torch.from_numpy(output).float()
    return output


def find_good_maps(D_outs, pred_all, threshold_st, device):
    """Find predictions above confidence threshold for self-training"""
    count = 0
    indexes = []
    for i in range(D_outs.size(0)):
        if D_outs[i] > threshold_st:
            count += 1
            indexes.append(i)
    
    if count > 0:
        pred_sel = torch.Tensor(count, pred_all.size(1), pred_all.size(2), pred_all.size(3))
        label_sel = torch.Tensor(count, pred_sel.size(2), pred_sel.size(3))
        num_sel = 0
        for j in range(D_outs.size(0)):
            if D_outs[j] > threshold_st:
                pred_sel[num_sel] = pred_all[j]
                label_sel[num_sel] = compute_argmax_map(pred_all[j])
                num_sel += 1
        return pred_sel.to(device), label_sel.to(device), count, indexes
    else:
        return None, None, 0, []


def evaluate_model(model, dataloader, device, num_classes, ignore_label=255):
    """Evaluate model on validation set and compute mIoU"""
    model.eval()
    label_trues = []
    label_preds = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            images, labels, _, _, _ = batch
            images = images.to(device)
            labels = labels.numpy()
            
            # Get original label size
            label_size = labels.shape[1:]  # (H, W)
            
            # Forward pass
            preds = model(images)
            preds = F.interpolate(preds, size=label_size, mode='bilinear', align_corners=True)
            preds = preds.detach().cpu().numpy()
            preds = np.argmax(preds, axis=1)
            
            # Filter out ignore labels
            for pred, label in zip(preds, labels):
                mask = label != ignore_label
                if mask.sum() > 0:
                    label_trues.append(label[mask])
                    label_preds.append(pred[mask])
    
    model.train()
    
    if len(label_trues) == 0:
        return 0.0
    
    # Calculate metrics
    metrics = scores(label_trues, label_preds, num_classes)
    return metrics["Mean IoU"]


def get_params(model, key):
    """Get parameters for different learning rates"""
    # For Dilated FCN
    if key == "1x":
        for m in model.named_modules():
            if "layer" in m[0]:
                if isinstance(m[1], nn.Conv2d):
                    for p in m[1].parameters():
                        yield p
    # For conv weight in the ASPP module
    if key == "10x":
        for m in model.named_modules():
            if "aspp" in m[0]:
                if isinstance(m[1], nn.Conv2d):
                    yield m[1].weight
    # For conv bias in the ASPP module
    if key == "20x":
        for m in model.named_modules():
            if "aspp" in m[0]:
                if isinstance(m[1], nn.Conv2d):
                    yield m[1].bias


def main():
    args = get_arguments()
    
    # Set random seed
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.random_seed)
    
    # Parse input size
    h, w = map(int, args.input_size.split(','))
    input_size = (h, w)
    
    # Setup device
    cudnn.enabled = True
    cudnn.benchmark = True
    device = get_device(cuda=True)
    
    # Initialize Wandb
    if not args.no_wandb:
        # Prompt for API key if not logged in
        try:
            wandb.login()
        except:
            print("\n" + "="*60)
            print("Wandb Login Required")
            print("="*60)
            print("Please enter your Wandb API key.")
            print("You can find it at: https://wandb.ai/authorize")
            api_key = input("API Key: ").strip()
            wandb.login(key=api_key)
        
        # Initialize wandb run
        run_name = args.wandb_run_name or f"s4gan_salak_{args.batch_size}bs_{args.threshold_st}st"
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config=vars(args)
        )
        print(f"\n✓ Wandb initialized: {args.wandb_project}/{run_name}\n")
    
    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Load dataset
    print("="*60)
    print("Loading Salak Dataset...")
    print("="*60)
    
    # Load ALL patches (labeled + unlabeled) for semi-supervised training
    full_dataset = SalakDataSet(
        root=args.data_root,
        list_path=None,  # None = load ALL patches (370k+ patches)
        class_mapping_csv=args.class_mapping,
        module='s4gan',
        crop_size=input_size,
        scale=args.random_scale,
        mirror=args.random_mirror,
        mean=IMG_MEAN
    )
    
    dataset_size = len(full_dataset)
    print(f"\n✓ Total patches loaded: {dataset_size}")
    
    # Count labeled vs unlabeled samples
    labeled_count = sum(1 for item in full_dataset.files if item['label'] is not None)
    unlabeled_count = dataset_size - labeled_count
    print(f"✓ Labeled patches: {labeled_count}")
    print(f"✓ Unlabeled patches: {unlabeled_count}")
    
    # Split ONLY labeled data into train and validation
    labeled_indices = [i for i, item in enumerate(full_dataset.files) if item['label'] is not None]
    unlabeled_indices = [i for i, item in enumerate(full_dataset.files) if item['label'] is None]
    
    val_size = int(args.val_split * len(labeled_indices))
    train_size_labeled = len(labeled_indices) - val_size
    
    np.random.shuffle(labeled_indices)
    
    train_indices = labeled_indices[:train_size_labeled]
    val_indices = labeled_indices[train_size_labeled:]
    
    print(f"\n📊 Training Strategy:")
    print(f"  Labeled train samples: {train_size_labeled}")
    print(f"  Labeled validation samples: {val_size}")
    print(f"  Unlabeled samples (for semi-supervised): {unlabeled_count}")
    
    # Create samplers
    train_sampler = data.sampler.SubsetRandomSampler(train_indices)
    val_sampler = data.sampler.SubsetRandomSampler(val_indices)
    
    # Create dataloaders
    trainloader = data.DataLoader(
        full_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=0,
        pin_memory=True
    )
    
    trainloader_gt = data.DataLoader(
        full_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=0,
        pin_memory=True
    )
    
    # For unlabeled data, we reuse the full dataset (unlabeled samples have label=255)
    trainloader_remain = data.DataLoader(
        full_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    
    valloader = data.DataLoader(
        full_dataset,
        batch_size=args.batch_size,
        sampler=val_sampler,
        num_workers=0,
        pin_memory=True
    )
    
    # Create model
    print("\n" + "="*60)
    print("Initializing Models...")
    print("="*60)
    
    model = DeepLabV2_ResNet101_MSC(n_classes=args.num_classes)
    
    # Load pretrained weights if provided
    start_iter = 0
    discriminator_state = None  # Initialize at module level
    
    if args.restore_from is not None:
        if os.path.isfile(args.restore_from):
            print(f"Loading checkpoint from: {args.restore_from}")
            
            # Check if it's a full checkpoint or just weights
            checkpoint = torch.load(args.restore_from)
            
            if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
                # Full checkpoint with iteration, optimizer, etc.
                print("  ✓ Loading full checkpoint (model + optimizer + iteration)")
                model.load_state_dict(checkpoint['model_state'])
                start_iter = checkpoint.get('iteration', 0)
                print(f"  ✓ Resuming from iteration: {start_iter}")
                
                # Load discriminator if available
                if 'model_D_state' in checkpoint:
                    print("  ✓ Discriminator state found in checkpoint")
                    discriminator_state = checkpoint['model_D_state']
            else:
                # Just model weights
                print("  ✓ Loading model weights only")
                saved_state_dict = checkpoint if isinstance(checkpoint, dict) else torch.load(args.restore_from)
                new_params = model.state_dict().copy()
                for name, param in new_params.items():
                    if name in saved_state_dict and param.size() == saved_state_dict[name].size():
                        new_params[name].copy_(saved_state_dict[name])
                model.load_state_dict(new_params)
        else:
            print(f"WARNING: Checkpoint file not found: {args.restore_from}")
    
    model = nn.DataParallel(model)
    model = model.to(device)
    model.train()
    
    # Initialize discriminator
    model_D = s4GAN_discriminator(num_classes=args.num_classes, dataset='custom')
    
    # Load discriminator state if resuming
    if discriminator_state is not None:
        model_D.load_state_dict(discriminator_state)
        print("  ✓ Discriminator state loaded")
    
    model_D = nn.DataParallel(model_D)
    model_D = model_D.to(device)
    model_D.train()
    
    print("✓ Models initialized")
    
    # Optimizers
    optimizer = torch.optim.SGD(
        params=[
            {"params": get_params(model.module, key="1x"), "lr": args.learning_rate, "weight_decay": WEIGHT_DECAY},
            {"params": get_params(model.module, key="10x"), "lr": 10 * args.learning_rate, "weight_decay": WEIGHT_DECAY},
            {"params": get_params(model.module, key="20x"), "lr": 20 * args.learning_rate, "weight_decay": 0.0},
        ],
        momentum=MOMENTUM,
    )
    
    optimizer_D = optim.Adam(model_D.parameters(), lr=args.learning_rate_D, betas=(0.9, 0.99))
    
    # Learning rate scheduler
    scheduler = PolynomialLR(
        optimizer=optimizer,
        step_size=LR_DECAY,
        iter_max=ITER_MAX,
        power=POWER,
    )
    
    interp = nn.Upsample(size=(input_size[0], input_size[1]), mode='bilinear', align_corners=True)
    criterion_bce = nn.BCELoss()
    
    # Training loop
    print("\n" + "="*60)
    print("Starting Training...")
    print("="*60)
    
    trainloader_iter = iter(trainloader)
    trainloader_remain_iter = iter(trainloader_remain)
    trainloader_gt_iter = iter(trainloader_gt)
    
    best_val_miou = 0.0
    
    # Progress bar for training
    pbar = tqdm(range(start_iter, args.num_steps), initial=start_iter, total=args.num_steps, desc="Training")
    
    for i_iter in pbar:
        
        # Reset losses
        loss_ce_value = 0
        loss_D_value = 0
        loss_fm_value = 0
        loss_S_value = 0
        
        optimizer.zero_grad()
        optimizer_D.zero_grad()
        lr_D = adjust_learning_rate_D(optimizer_D, i_iter, args)
        
        # =======================
        # Train Generator (Segmentation Network)
        # =======================
        for param in model_D.parameters():
            param.requires_grad = False
        
        # 1. Supervised loss on labeled data
        try:
            batch = next(trainloader_iter)
        except:
            trainloader_iter = iter(trainloader)
            batch = next(trainloader_iter)
        
        images, labels, _, _, _ = batch
        images = Variable(images).to(device)
        pred = interp(model(images))
        loss_ce = loss_calc(pred, labels, device, args.ignore_label)
        
        # 2. Self-training on unlabeled data
        try:
            batch_remain = next(trainloader_remain_iter)
        except:
            trainloader_remain_iter = iter(trainloader_remain)
            batch_remain = next(trainloader_remain_iter)
        
        images_remain, _, _, _, _ = batch_remain
        images_remain = Variable(images_remain).to(device)
        pred_remain = interp(model(images_remain))
        
        # Normalize images for discriminator
        images_remain_norm = (images_remain - torch.min(images_remain)) / (torch.max(images_remain) - torch.min(images_remain) + 1e-8)
        pred_cat = torch.cat((F.softmax(pred_remain, dim=1), images_remain_norm), dim=1)
        D_out_z, D_out_y_pred = model_D(pred_cat)
        
        # Find high-confidence predictions
        pred_sel, labels_sel, count, indexes = find_good_maps(D_out_z, pred_remain, args.threshold_st, device)
        
        # Self-training loss
        if count > 0 and i_iter > 1000:
            loss_st = loss_calc(pred_sel, labels_sel, device, args.ignore_label)
        else:
            loss_st = 0.0
        
        # 3. Feature matching loss
        try:
            batch_gt = next(trainloader_gt_iter)
        except:
            trainloader_gt_iter = iter(trainloader_gt)
            batch_gt = next(trainloader_gt_iter)
        
        images_gt, labels_gt, _, _, _ = batch_gt
        D_gt_v = Variable(one_hot(labels_gt, args.num_classes)).to(device)
        
        images_gt = images_gt.to(device)
        images_gt_norm = (images_gt - torch.min(images_gt)) / (torch.max(images_gt) - torch.min(images_gt) + 1e-8)
        D_gt_v_cat = torch.cat((D_gt_v, images_gt_norm), dim=1)
        D_out_z_gt, D_out_y_gt = model_D(D_gt_v_cat)
        
        loss_fm = torch.mean(torch.abs(torch.mean(D_out_y_gt, 0) - torch.mean(D_out_y_pred, 0)))
        
        # Total generator loss
        if count > 0 and i_iter > 1000:
            loss_S = loss_ce + args.lambda_fm * loss_fm + args.lambda_st * loss_st
        else:
            loss_S = loss_ce + args.lambda_fm * loss_fm
        
        loss_S.backward()
        
        loss_ce_value = loss_ce.item()
        loss_fm_value = args.lambda_fm * loss_fm.item()
        if isinstance(loss_st, float):
            loss_st_value = 0.0
        else:
            loss_st_value = args.lambda_st * loss_st.item()
        loss_S_value = loss_S.item()
        
        # =======================
        # Train Discriminator
        # =======================
        for param in model_D.parameters():
            param.requires_grad = True
        
        # Train with fake (predicted)
        pred_cat = pred_cat.detach()
        D_out_z, _ = model_D(pred_cat)
        y_fake = Variable(torch.zeros(D_out_z.size(0), 1).to(device))
        loss_D_fake = criterion_bce(D_out_z, y_fake)
        
        # Train with real (ground truth)
        D_out_z_gt, _ = model_D(D_gt_v_cat)
        y_real = Variable(torch.ones(D_out_z_gt.size(0), 1).to(device))
        loss_D_real = criterion_bce(D_out_z_gt, y_real)
        
        loss_D = (loss_D_fake + loss_D_real) / 2.0
        loss_D.backward()
        loss_D_value = loss_D.item()
        
        # Update optimizers
        optimizer.step()
        optimizer_D.step()
        scheduler.step(epoch=i_iter)
        
        # =======================
        # Logging
        # =======================
        if i_iter % 100 == 0:
            # Update progress bar with current losses
            pbar.set_postfix({
                'CE': f'{loss_ce_value:.3f}',
                'FM': f'{loss_fm_value:.3f}',
                'ST': f'{loss_st_value:.3f}',
                'D': f'{loss_D_value:.3f}',
                'ST_cnt': count
            })
            
            if not args.no_wandb:
                wandb.log({
                    "Training Loss/Cross Entropy": loss_ce_value,
                    "Training Loss/Feature Matching": loss_fm_value,
                    "Training Loss/Self-Training": loss_st_value,
                    "Training Loss/Discriminator": loss_D_value,
                    "Training Loss/Total Generator": loss_S_value,
                    "Self-Training/Confidence Count": count,
                    "Learning Rate/Generator": optimizer.param_groups[0]['lr'],
                    "Learning Rate/Discriminator": lr_D,
                    "Iteration": i_iter
                })
        
        # =======================
        # Validation Evaluation
        # =======================
        if i_iter % args.eval_every == 0 and i_iter > 0:
            pbar.write(f"\n{'='*60}")
            pbar.write(f"Evaluating at iteration {i_iter}...")
            pbar.write(f"{'='*60}")
            
            # Evaluate on train set (subset for speed)
            train_miou = evaluate_model(model, trainloader, device, args.num_classes, args.ignore_label)
            pbar.write(f"Training mIoU: {train_miou:.4f}")
            
            # Evaluate on validation set
            val_miou = evaluate_model(model, valloader, device, args.num_classes, args.ignore_label)
            pbar.write(f"Validation mIoU: {val_miou:.4f}")
            
            if not args.no_wandb:
                wandb.log({
                    "Metrics/Training mIoU": train_miou,
                    "Metrics/Validation mIoU": val_miou,
                    "Iteration": i_iter
                })
            
            # Save best model (comparing with previous best)
            if val_miou > best_val_miou:
                prev_best = best_val_miou
                best_val_miou = val_miou
                pbar.write(f"✓ New best validation mIoU: {best_val_miou:.4f} (previous: {prev_best:.4f})")
                pbar.write(f"  Saving best model to: {os.path.join(args.checkpoint_dir, 'best_model.pth')}")
                torch.save(model.module.state_dict(), os.path.join(args.checkpoint_dir, 'best_model.pth'))
                torch.save(model_D.module.state_dict(), os.path.join(args.checkpoint_dir, 'best_model_D.pth'))
                
                if not args.no_wandb:
                    wandb.run.summary["Best Validation mIoU"] = best_val_miou
                    wandb.run.summary["Best Model Iteration"] = i_iter
            else:
                pbar.write(f"  Current mIoU: {val_miou:.4f} (best remains: {best_val_miou:.4f})")
            
            pbar.write(f"{'='*60}\n")
        
        # =======================
        # Save Latest Model (with full checkpoint for resuming)
        # =======================
        if i_iter % args.save_latest_every == 0 and i_iter != 0:
            # Save full checkpoint for easy resuming
            checkpoint = {
                'iteration': i_iter,
                'model_state': model.module.state_dict(),
                'model_D_state': model_D.module.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'optimizer_D_state': optimizer_D.state_dict(),
                'best_val_miou': best_val_miou,
            }
            torch.save(checkpoint, os.path.join(args.checkpoint_dir, 'latest_checkpoint.pth'))
            
            # Also save model weights only (for inference)
            torch.save(model.module.state_dict(), os.path.join(args.checkpoint_dir, 'latest_model.pth'))
            torch.save(model_D.module.state_dict(), os.path.join(args.checkpoint_dir, 'latest_model_D.pth'))
        
        # =======================
        # Save Periodic Checkpoint
        # =======================
        if i_iter % args.save_pred_every == 0 and i_iter != 0:
            pbar.write(f'💾 Saving checkpoint at iteration {i_iter}...')
            
            # Save full checkpoint
            checkpoint = {
                'iteration': i_iter,
                'model_state': model.module.state_dict(),
                'model_D_state': model_D.module.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'optimizer_D_state': optimizer_D.state_dict(),
                'best_val_miou': best_val_miou,
            }
            torch.save(checkpoint, os.path.join(args.checkpoint_dir, f'checkpoint_{i_iter}.pth'))
            pbar.write(f'  ✓ Checkpoint saved: checkpoint_{i_iter}.pth')
    
    pbar.close()
    
    # =======================
    # Final Evaluation
    # =======================
    print("\n" + "="*60)
    print("Final Evaluation...")
    print("="*60)
    
    train_miou = evaluate_model(model, trainloader, device, args.num_classes, args.ignore_label)
    val_miou = evaluate_model(model, valloader, device, args.num_classes, args.ignore_label)
    
    print(f"Final Training mIoU: {train_miou:.4f}")
    print(f"Final Validation mIoU: {val_miou:.4f}")
    print(f"Best Validation mIoU: {best_val_miou:.4f}")
    
    if not args.no_wandb:
        wandb.log({
            "Metrics/Final Training mIoU": train_miou,
            "Metrics/Final Validation mIoU": val_miou,
        })
        wandb.run.summary["Final Training mIoU"] = train_miou
        wandb.run.summary["Final Validation mIoU"] = val_miou
    
    # Save final model
    torch.save(model.module.state_dict(), os.path.join(args.checkpoint_dir, 'final_model.pth'))
    torch.save(model_D.module.state_dict(), os.path.join(args.checkpoint_dir, 'final_model_D.pth'))
    
    end = timeit.default_timer()
    print(f'\nTotal training time: {(end - start)/3600:.2f} hours')
    
    if not args.no_wandb:
        wandb.finish()
    
    print("\n✓ Training completed!")


if __name__ == '__main__':
    main()
