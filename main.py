import argparse
import os
import torch.nn.parallel
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.optim import AdamW
from models.modeling.resnet import resnet101, resnet50
from dataloader.a2d_loader_diff import *
from eval import evaluate, cal_fps
from train import train
import random
import numpy as np
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from get_model import get_model_by_name


def seed_setting():
    torch.cuda.empty_cache()
    seed = 3407
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


def get_args():
    parser = argparse.ArgumentParser('cvpr22-rvos')

    # input
    parser.add_argument('--image_dim', type=int, default=320)
    parser.add_argument('--feature_dim', type=int, default=10)
    parser.add_argument('--dataroot', type=str, default='./datasets')
    parser.add_argument('--task', type=str, default='a2d', choices=['a2d', 'ytvos', 'jhmdb', 'davis'])
    parser.add_argument('--phrase_len', type=int, default=25)
    parser.add_argument('--glove_path', type=str, default='/pretrain/glove_840B_300d')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--save_path', type=str)
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--resume', type=str, default='')
    parser.add_argument('--save_result', action='store_true')
    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument('--interval', type=int, default=2)
    parser.add_argument('--dist', action='store_true')
    parser.add_argument('--fps', action='store_true')
    parser.add_argument('--num_workers', type=int, default=8)

    # backbone
    parser.add_argument('--backbone', type=str, default='resnet50', choices=['resnet50', 'resnet101', 'i3d'])

    # ddp
    parser.add_argument('--local_rank', default=-1, type=int)

    # lr
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--learning_rate_decay_rate', type=float, default=0.5)

    # vis
    parser.add_argument('--vis',  action='store_true')

    args = parser.parse_args()
    return args


def main(args):
    # ddp backend init
    local_rank = args.local_rank
    torch.cuda.set_device(local_rank)
    if args.dist:
        dist.init_process_group(backend='nccl')
        device = torch.device('cuda', local_rank)
    else:
        device = torch.device('cuda')

    if args.backbone == "resnet50":
        image_encoder = resnet50(pretrained=True)
        flow_encoder = resnet50(pretrained=True)
    elif args.backbone == "resnet101":
        image_encoder = resnet101(pretrained=True)
        flow_encoder = resnet101(pretrained=True)
    else:
        raise NotImplemented("Model not implemented")

    joint_model = get_model_by_name(
        args.model_name,
        image_encoder=image_encoder,
        flow_encoder=flow_encoder,
    ).to(device)

    if args.resume != '' and (args.dist and dist.get_rank()) == 0:
        joint_model.load_state_dict(torch.load(args.resume), strict=False)

    if args.dist:
        joint_model = DDP(joint_model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    else:
        joint_model = torch.nn.parallel.DataParallel(joint_model)

    params = list([p for p in joint_model.parameters() if p.requires_grad])
    if args.dist and dist.get_rank() == 0:
        print(f"interval: {args.interval}")
        print(f"len params training: {len(params)}.")

    optimizer = AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    to_tensor = transforms.ToTensor()
    resize = transforms.Resize((args.image_dim, args.image_dim))

    train_dataset = ReferDataset(
        data_root=args.dataroot,
        dataset=args.task,
        transform=transforms.Compose(
            [resize, to_tensor, normalize]
        ),
        transform_flow=transforms.Compose(
            [resize, to_tensor, normalize]
        ),
        transform_orig=transforms.Compose([resize, to_tensor]),
        eval_ann_transform=transforms.Compose([ResizeAnnotation(args.image_dim)]),
        split="train",
        max_query_len=args.phrase_len,
        glove_path=args.glove_path,
        interval=args.interval,
        save_result=args.save_result
    )
    val_dataset = ReferDataset(
        data_root=args.dataroot,
        dataset=args.task,
        transform=transforms.Compose([resize, to_tensor, normalize]),
        transform_flow=transforms.Compose(
            [resize, to_tensor, normalize]
        ),
        transform_orig=transforms.Compose([resize, to_tensor]),
        eval_ann_transform=transforms.Compose([ResizeAnnotation(args.image_dim)]),
        split="val",
        max_query_len=args.phrase_len,
        glove_path=args.glove_path,
        interval=args.interval,
        save_result=args.save_result
    )

    if args.dist:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)

    if args.dist:
        train_loader = DataLoader(
            train_dataset,
            sampler=train_sampler,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            shuffle=True,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            drop_last=True
        )

    # if not args.vis:
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=True
    )
    milestone = [10, 12, 14]
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestone, gamma=args.learning_rate_decay_rate)

    if args.dist and dist.get_rank() == 0:
        num_iter = len(train_loader)
        print(f"training iterations {num_iter}")

    best_val_acc = 0

    if args.fps:
        cal_fps(val_loader, joint_model, args)
        return

    for epochId in range(args.epochs):
        if args.dist:
            train_loader.sampler.set_epoch(epochId)
        if (args.task != 'jhmdb' and args.resume == '') or args.task == 'davis':
            train(train_loader, joint_model, optimizer,  epochId, args)
        val_acc = evaluate(val_loader, joint_model, epochId, args)
        if args.save_result or args.resume != '':
            break  ## for save
        lr_scheduler.step()

        if epochId >= milestone[0] - 1 or args.task == 'davis':
            best_val_acc = val_acc
            if dist.get_rank() == 0:
                print(f'Saving epoch {epochId+1} to {args.save_path}/checkpoint{epochId+1}.ckpt')
                torch.save(joint_model.module.state_dict(), os.path.join(args.save_path, f'checkpoint{epochId+1}.ckpt'))
        else:
            if dist.get_rank() == 0:
                print(f'Saving epoch {epochId + 1} to {args.save_path}/checkpoint.ckpt')
                torch.save(joint_model.module.state_dict(), os.path.join(args.save_path, f'checkpoint.ckpt'))



if __name__ == "__main__":
    seed_setting()
    args = get_args()
    main(args)
