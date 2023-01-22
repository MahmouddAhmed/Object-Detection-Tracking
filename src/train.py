import os
import time
import torch
import argparse
from losses.combined_loss import CombinedLoss
from models.models import build_model
from data.market_1501_datamanager import ImageDataManager
from utils.utils import cosine_distance,euclidean_squared_distance,compute_distance_matrix
from utils.market1501_utils import extract_features,eval_market1501,MetricMeter,AverageMeter,print_statistics



def train(model,train_loader,val_loader,criterion,optimizer,scheduler,output_dir,MAX_EPOCH,EPOCH_EVAL_FREQ,PRINT_FREQ):
    num_batches = len(train_loader)
    for epoch in range(MAX_EPOCH):
        losses = MetricMeter()
        batch_time = AverageMeter()
        end = time.time()
        model.train()
        for batch_idx, data in enumerate(train_loader):
            # Predict output.
            imgs, pids = data['img'].cuda(), data['pid'].cuda()
            logits, features = model(imgs)
            # Compute loss.
            loss, loss_summary = criterion(logits, features, pids)
        
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
            batch_time.update(time.time() - end)
            losses.update(loss_summary)
            if (batch_idx + 1) % PRINT_FREQ == 0:
                print_statistics(batch_idx, num_batches, epoch, MAX_EPOCH, batch_time, losses)
            end = time.time()

        scheduler.step()    
        if (epoch + 1) % EPOCH_EVAL_FREQ == 0 or epoch == MAX_EPOCH - 1:
            torch.save(model, output_dir)
            rank1, mAP = evaluate(model, val_loader )
            print('Epoch {0}/{1}: Rank1: {rank}, mAP: {map}'.format(
                    epoch + 1, MAX_EPOCH, rank=rank1, map=mAP))

def evaluate(model, test_loader, ranks=[1, 5, 10, 20],metric_fn=cosine_distance):
    with torch.no_grad():
        model.eval()
        print('Extracting features from query set...')
        q_feat, q_pids, q_camids = extract_features(model, test_loader['query'])
        print('Done, obtained {}-by-{} matrix'.format(q_feat.size(0), q_feat.size(1)))

        print('Extracting features from gallery set ...')
        g_feat, g_pids, g_camids = extract_features(model, test_loader['gallery'])
        print('Done, obtained {}-by-{} matrix'.format(g_feat.size(0), g_feat.size(1)))
        
        distmat = compute_distance_matrix(q_feat, g_feat, metric_fn=metric_fn)
        distmat = distmat.numpy()

        print('Computing CMC and mAP ...')
        cmc, mAP = eval_market1501(
            distmat,
            q_pids,
            g_pids,
            q_camids,
            g_camids,
            max_rank=50
        )

        print('** Results **')
        print('mAP: {:.1%}'.format(mAP))
        print('CMC curve')
        for r in ranks:
            print('Rank-{:<3}: {:.1%}'.format(r, cmc[r - 1]))
        return cmc[0], mAP

def parse_args():
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--model', type=str, default='final-reid_model.pth', help='model name')
    parser.add_argument('--output_dir', type=str, default='models/', help='output directory to save the model')
    parser.add_argument('--batch_size', type=int, default=32, help='batch Size during training')
    parser.add_argument('--epoch', default=50, type=int, help='number of epoch to run')
    parser.add_argument('--epoch_eval_freq', default=5, type=int, help='evaluate model every n epochs')
    parser.add_argument('--print_freq', type=int, default=10, help='print losses every')
    parser.add_argument('-g', '--gpu', metavar='G', type=str, default='0',help='GPU', dest='gpu')
    parser.add_argument('--data_dir', type=str, default='.', help='market data directory')
    parser.add_argument('--resume_ckpt', type=str,default=None, help='checkpoint path to resume training')
    parser.add_argument('--learning_rate', default=0.0003, type=float, help='Initial learning rate')
    parser.add_argument('--loss_margin', default=0.3, type=float, )
    parser.add_argument('--hard_negative_mining_weight', default=1, type=float, )
    parser.add_argument('--cross_entropy_weight', default=1, type=float, )


    return parser.parse_args()

def main(args):
    # Declare device
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create Dataloaders
    datamanager = ImageDataManager(root=args.data_dir, 
                               height=256,
                               width=128,
                               batch_size_train=args.batch_size, 
                               workers=2,
                               transforms=['random_flip', 'random_crop'],
                               train_sampler='RandomIdentitySampler')
    train_loader = datamanager.train_loader
    val_loader = datamanager.test_loader

    # Instantiate model
    model = build_model('resnet34', datamanager.num_train_pids, loss='triplet', pretrained=True)
    model = model.cuda()

    trainable_params = model.parameters()

    # Load model if resuming from checkpoint
    if args.resume_ckpt is not None:
        model.load_state_dict(torch.load(
            args.resume_ckpt, map_location=device))

    # Move model to specified device
    model.to(device)

    # Model Output path
    output_dir=os.path.join(args.output_dir,args.model)

    optimizer = torch.optim.Adam(trainable_params, lr=args.learning_rate, weight_decay=5e-4, amsgrad=True)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10)

    # Loss Criteria
    criterion = CombinedLoss(args.loss_margin, args.hard_negative_mining_weight, args.cross_entropy_weight)

    # Start training
    train(model,train_loader,val_loader,criterion,optimizer,scheduler,output_dir,args.epoch,args.epoch_eval_freq,args.print_freq)


if __name__ == '__main__':
    args = parse_args()
    main(args)


    