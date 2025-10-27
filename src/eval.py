import torch
import numpy as np
import time
from utils import *
from tqdm import tqdm

def predict(loader, model, args, num_e, sr2o, srt2o, logger):
    model.eval()

    with torch.no_grad():
        results = {}

        print("start evaluation")
        t1 = time.time()

        for batch in tqdm(loader):
            score = model.forward(batch, num_neighbors=args.num_neighbors)
            b_range = torch.arange(score.shape[0], dtype=torch.long, device=args.device)

            target_score = score[b_range, torch.from_numpy(batch.target_idx).long().to(args.device)]
            sub = batch.src_idx
            rel = batch.rel_idx
            
            filter_label = torch.stack([get_label(sr2o[(int(s), int(r))], num_e) for (s, r) in zip(sub, rel)], dim=0).to(args.device)
            score = torch.where(filter_label.byte(), -torch.ones_like(score) * 10000000, score)
            score[b_range, torch.from_numpy(batch.target_idx).long().to(args.device)] = target_score

            ranks = 1 + torch.argsort(torch.argsort(score, dim=1, descending=True), dim=1, descending=False)[
                b_range, torch.from_numpy(batch.target_idx).long().to(args.device)]
            ranks = ranks.float()
            results['count'] = torch.numel(ranks) + results.get('count', 0.0)
            results['mar'] = torch.sum(ranks).item() + results.get('mar', 0.0)
            results['mrr'] = torch.sum(1.0 / ranks).item() + results.get('mrr', 0.0)
            for k in range(10):
                results['hits@{}'.format(k + 1)] = torch.numel(ranks[ranks <= (k + 1)]) + results.get(
                    'hits@{}'.format(k + 1), 0.0)

        results['mar'] = round(results['mar'] / results['count'], 5)
        results['mrr'] = round(results['mrr'] / results['count'], 5)
        for k in range(10):
            results['hits@{}'.format(k + 1)] = round(results['hits@{}'.format(k + 1)] / results['count'], 5)

        t2 = time.time()
        print("evaluation time: ", t2 - t1)
        logger.info("evaluation time: {}".format(t2 - t1))

    return results

def get_label(label, num_e):
    y = np.zeros([num_e], dtype=np.float32)
    for e2 in label: y[e2] = 1.0
    return torch.FloatTensor(y)
